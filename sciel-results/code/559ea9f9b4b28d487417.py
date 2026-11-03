import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
DEADLINE = 135.0  # Leave 15s margin before 150s kill

def to_sphere(params):
    """Convert raw params to unit vectors."""
    v = np.asarray(params).reshape(-1, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return v / norms

def compute_covering_radius(points):
    """Compute the covering radius (in radians) for a set of points on the unit sphere."""
    points = np.asarray(points)
    points = to_sphere(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # worst case if hull fails (e.g. points in hemisphere)
    
    max_circum = 0.0
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        # Compute circumcenter of triangle in 3D
        ab = b - a
        ac = c - a
        abab = np.dot(ab, ab)
        acac = np.dot(ac, ac)
        abac = np.dot(ab, ac)
        
        denom = abab * acac - abac * abac
        if abs(denom) < 1e-12:
            continue
        
        u1 = (acac * abac - abab * abac) / denom
        u2 = (abab * abac - acac * abac) / denom
        circumcenter = a + u1 * ab + u2 * ac
        
        norm_cc = np.linalg.norm(circumcenter)
        if norm_cc < 1e-10:
            continue
        
        pole = circumcenter / norm_cc
        
        # Angular distance from pole to vertices
        theta_a = np.arccos(np.clip(np.dot(pole, a), -1, 1))
        theta_b = np.arccos(np.clip(np.dot(pole, b), -1, 1))
        theta_c = np.arccos(np.clip(np.dot(pole, c), -1, 1))
        
        circum_angular = max(theta_a, theta_b, theta_c)
        if circum_angular > max_circum:
            max_circum = circum_angular
            
    return max_circum

def objective_and_grad(x):
    """
    Objective function and gradient for scipy.optimize.minimize.
    x is a flat array of shape (N*3,).
    """
    points = x.reshape(N, 3)
    points = to_sphere(points)
    
    # We want to minimize the covering radius.
    # The covering radius is a piecewise differentiable function.
    # We approximate the gradient by moving points away from the "poles" of the largest facets.
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, points are likely in a hemisphere.
        # Return a large value and a gradient that pushes them apart.
        return 10.0, np.zeros_like(x)
    
    max_circum = 0.0
    grad = np.zeros((N, 3))
    
    # Store info for the largest facets to compute gradient
    facet_infos = []
    
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        a_idx, b_idx, c_idx = simplex[0], simplex[1], simplex[2]
        
        ab = b - a
        ac = c - a
        abab = np.dot(ab, ab)
        acac = np.dot(ac, ac)
        abac = np.dot(ab, ac)
        
        denom = abab * acac - abac * abac
        if abs(denom) < 1e-12:
            continue
        
        u1 = (acac * abac - abab * abac) / denom
        u2 = (abab * abac - acac * abac) / denom
        circumcenter = a + u1 * ab + u2 * ac
        
        norm_cc = np.linalg.norm(circumcenter)
        if norm_cc < 1e-10:
            continue
        
        pole = circumcenter / norm_cc
        
        theta_a = np.arccos(np.clip(np.dot(pole, a), -1, 1))
        theta_b = np.arccos(np.clip(np.dot(pole, b), -1, 1))
        theta_c = np.arccos(np.clip(np.dot(pole, c), -1, 1))
        
        circum_angular = max(theta_a, theta_b, theta_c)
        
        if circum_angular > max_circum:
            max_circum = circum_angular
            facet_infos = [(simplex, pole, a_idx, b_idx, c_idx, theta_a, theta_b, theta_c)]
        elif abs(circum_angular - max_circum) < 1e-6:
            facet_infos.append((simplex, pole, a_idx, b_idx, c_idx, theta_a, theta_b, theta_c))
            
    if not facet_infos:
        return max_circum, np.zeros_like(x)
    
    # Compute subgradient
    # The gradient of the distance from a point p to a pole q is (p - q)/||p - q|| (projected to tangent space)
    # Actually, the distance is arccos(p.q). d/dp arccos(p.q) = -q / sqrt(1 - (p.q)^2)
    # This vector points in the direction that increases the dot product p.q, thus decreasing the angle.
    # We want to DECREASE the covering radius, so we want to move points AWAY from the pole?
    # No, if the pole is the center of the empty cap, moving the vertices of the facet TOWARDS the pole
    # shrinks the cap? 
    # Let's think. The cap is centered at the pole. The vertices are on the boundary.
    # To shrink the cap, we should move the vertices towards the pole.
    # The vector from vertex to pole is pole - vertex.
    # The gradient of the angle theta = arccos(p.v) with respect to v is -p / sqrt(1 - (p.v)^2).
    # This points in the direction of increasing p.v, i.e., towards the pole.
    # So -gradient points away from the pole.
    # We want to minimize theta, so we move in direction -grad(theta).
    # Wait, minimize theta -> move towards pole.
    # grad(theta) points towards pole. So we should move in direction grad(theta)?
    # Let's check: if we move v towards p, dot product increases, arccos decreases.
    # So yes, we should move in the direction of the gradient of the angle?
    # No, standard gradient descent moves in direction -gradient of objective.
    # Objective is theta. We want to decrease theta.
    # So we move in direction -grad(theta).
    # But grad(theta) points towards the pole?
    # Let v = (1,0,0), p = (0,1,0). theta = pi/2.
    # d(theta)/dv = -p / sqrt(1 - 0) = (0, -1, 0).
    # Moving v in direction (0, -1, 0) makes v = (1, -eps, 0).
    # dot(v, p) = -eps. arccos(-eps) > pi/2. Angle increased.
    # So grad(theta) points AWAY from the pole?
    # Let's re-evaluate.
    # theta = arccos(u). d(theta) = -du / sqrt(1-u^2).
    # u = v.p. du = dv.p.
    # d(theta) = - (dv.p) / sqrt(1 - (v.p)^2).
    # Gradient wrt v is -p / sqrt(1 - (v.p)^2).
    # If v=(1,0,0), p=(0,1,0), grad = (0, -1, 0).
    # Moving in direction -grad = (0, 1, 0).
    # New v = (1, eps, 0). dot(v, p) = eps. arccos(eps) < pi/2. Angle decreased.
    # So we should move in direction -grad(theta).
    
    for simplex, pole, a_idx, b_idx, c_idx, ta, tb, tc in facet_infos:
        # Only consider the vertices that define the maximum angle for this facet?
        # Or all vertices? The cap is defined by the whole triangle.
        # Moving any vertex towards the pole shrinks the cap.
        
        # Calculate gradient contribution for each vertex
        for idx, theta_val in zip([a_idx, b_idx, c_idx], [ta, tb, tc]):
            v = points[idx]
            dot_val = np.dot(v, pole)
            denom = np.sqrt(max(1e-10, 1.0 - dot_val * dot_val))
            
            # grad_theta = -pole / denom
            # We want to decrease theta, so move in direction -grad_theta = pole / denom
            # Project onto tangent space: (pole / denom) - v * dot(v, pole / denom)
            # Actually, simpler: just use the vector pointing to the pole, projected.
            # Vector to pole: pole - v.
            # Tangent projection: (pole - v) - v * dot(pole - v, v) = pole - v - v * (dot(pole, v) - 1) = pole - v * dot(pole, v).
            
            tangent_vec = pole - v * dot_val
            # Normalize to get unit step direction? Or just sum them?
            # The magnitude of the gradient is 1/denom.
            # We can just accumulate the tangent vectors.
            grad[idx] += tangent_vec / denom
            
    # Normalize gradient to have consistent scale
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-10:
        grad /= grad_norm
        
    return max_circum, grad.flatten()

def fibonacci_spiral(n_points):
    """Fibonacci spiral on the sphere."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def random_init(n_points, rng=None):
    """Generate random points on the unit sphere."""
    if rng is None:
        rng = np.random.RandomState()
    v = rng.randn(n_points, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def local_optimize(initial_points, max_iter=100):
    """Use gradient descent to locally optimize the covering radius."""
    x0 = initial_points.flatten()
    
    # Use L-BFGS-B which handles bounds, but we don't have bounds directly.
    # We rely on the projection in objective_and_grad.
    # However, standard L-BFGS-B doesn't project.
    # We will use a custom loop with projected gradient descent.
    
    points = initial_points.copy()
    best_points = points.copy()
    best_cr = compute_covering_radius(points)
    
    step_size = 0.1
    for i in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
            
        val, grad = objective_and_grad(points.flatten())
        
        if val < best_cr:
            best_cr = val
            best_points = points.copy()
            
        # Project gradient onto tangent space for each point
        # grad is already flattened. Reshape to (N, 3)
        g = grad.reshape(N, 3)
        
        # Project g_i onto tangent space at points_i
        # g_tan = g - points * dot(points, g)
        dots = np.einsum('ij,ij->i', points, g)
        g_tan = g - points * dots[:, np.newaxis]
        
        # Adaptive step size
        # If step is too large, it might oscillate.
        # Simple backtracking line search?
        # For simplicity, fixed small step with decay.
        
        new_points = points - step_size * g_tan
        new_points = to_sphere(new_points)
        
        # Check if it improved
        new_cr = compute_covering_radius(new_points)
        
        if new_cr < val:
            points = new_points
            step_size *= 1.05 # Slightly increase if success
            step_size = min(step_size, 0.5)
        else:
            step_size *= 0.5
            if step_size < 1e-4:
                break
                
    return best_points, best_cr

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    best_points = fibonacci_spiral(N)
    best_cr = compute_covering_radius(best_points)
    
    rng = np.random.RandomState(42)
    
    # Initializations
    inits = [
        fibonacci_spiral(N),
        random_init(N, rng),
        random_init(N, rng)
    ]
    
    # Add rotated Fibonacci
    for angle in [0.1, 0.5, 1.0]:
        theta = angle * np.pi
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        inits.append(fibonacci_spiral(N) @ R.T)
        
    # Add jittered Fibonacci
    for jitter in [0.05, 0.1, 0.2]:
        init = fibonacci_spiral(N)
        noise = jitter * rng.randn(N, 3)
        inits.append(to_sphere(init + noise))
        
    for init in inits:
        if time.time() > DEADLINE - 10:
            break
            
        optimized, cr = local_optimize(init, max_iter=50)
        if cr < best_cr:
            best_cr = cr
            best_points = optimized.copy()
            
    # Final fine-tuning
    if time.time() < DEADLINE - 20:
        optimized, cr = local_optimize(best_points, max_iter=100)
        if cr < best_cr:
            best_cr = cr
            best_points = optimized
    
    # Ensure validity
    best_points = to_sphere(best_points)
    
    # Check for coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-6:
                best_points[j] = best_points[j] * 0.999 + 0.001 * rng.randn(3)
                best_points[j] /= np.linalg.norm(best_points[j])
                
    return best_points

# Keep the module-level construct as the entry point
