"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def _normalize_points(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _compute_covering_radius(points):
    """
    Compute the covering radius of the configuration in radians.
    Returns max_r, worst_facet_indices, worst_center.
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        worst_facet = None
        worst_center = None
        
        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            # Compute circumcenter of spherical triangle
            # The circumcenter is the point equidistant from all three vertices
            # and lies on the normal to the plane of the triangle
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            # Ensure normal points towards the interior (towards origin side relative to triangle)
            if np.dot(n, p1) < 0:
                n = -n
            
            cos_r = np.dot(n, p1)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            r = np.arccos(cos_r)
            
            if r > max_r:
                max_r = r
                worst_facet = simplex
                worst_center = n
        return max_r, worst_facet, worst_center
    except Exception:
        return np.pi, None, None

def _smooth_facet_surrogate(params, beta=50.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp over facet circumradii.
    """
    pts = params.reshape(N, 3)
    pts = _normalize_points(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return np.pi
        
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.pi

    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    mask = norm_n[:, 0] > 1e-10
    if not np.all(mask):
        return np.pi
        
    n = n / norm_n
    dots = np.sum(n * p1, axis=1)
    n[dots < 0] *= -1
    
    cos_r = np.sum(n * p1, axis=1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    max_val = np.max(beta * radii)
    logsumexp = max_val + np.log(np.sum(np.exp(beta * radii - max_val)))
    return logsumexp / beta

def _repulsive_energy(params, s=4.0):
    """
    Riesz s-energy with moderate exponent to encourage uniform distribution.
    Lower s (e.g., 4) avoids trapping in Tammes-like local minima.
    """
    pts = params.reshape(N, 3)
    pts = _normalize_points(pts)
    
    dots = np.dot(pts, pts.T)
    dots = np.clip(dots, -1.0, 1.0)
    
    dist_sq = 2.0 - 2.0 * dots
    
    mask = ~np.eye(N, dtype=bool)
    dist_sq = dist_sq[mask]
    dist_sq = np.maximum(dist_sq, 1e-10)
    
    return np.sum(dist_sq ** (-s / 2.0))

def _gradient_descent_phase(points, time_limit=20.0, beta=50.0, use_repulsive=False, s=4.0):
    """
    Perform gradient descent using L-BFGS-B on the smooth surrogate.
    """
    start_time = time.time()
    
    if use_repulsive:
        def objective(params):
            if time.time() - start_time > time_limit:
                return 1e6
            return _repulsive_energy(params, s=s)
    else:
        def objective(params):
            if time.time() - start_time > time_limit:
                return 1e6
            return _smooth_facet_surrogate(params, beta=beta)
    
    x0 = points.flatten()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 300})
        best_points = _normalize_points(result.x.reshape(N, 3))
        return best_points
    except Exception:
        return points

def _local_refinement(points, time_limit=30.0):
    """
    Local refinement using coordinate descent on the worst facet.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    best_points = points.copy()
    best_radius, _, _ = _compute_covering_radius(points)
    
    step = 0.02
    
    for it in range(5000):
        if time.time() > deadline - 1.0:
            break
            
        current_radius, worst_facet, worst_center = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        if worst_facet is None:
            break
            
        v0 = points[worst_facet[0]]
        v1 = points[worst_facet[1]]
        v2 = points[worst_facet[2]]
        
        def move_vertex(v, c, step):
            grad = c - np.dot(c, v) * v
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                return v
            grad = grad / grad_norm
            v_new = v + step * grad
            return _normalize(v_new)
        
        points[worst_facet[0]] = move_vertex(v0, worst_center, step)
        points[worst_facet[1]] = move_vertex(v1, worst_center, step)
        points[worst_facet[2]] = move_vertex(v2, worst_center, step)
        
        if it % 100 == 0:
            step *= 0.98
            if step < 0.0005:
                step = 0.0005

    return best_points, best_radius

def _velocity_verlet_refine(points, time_limit=20.0, s=4.0, dt=0.005, damping=0.995):
    """
    Velocity Verlet integrator with adaptive step size and damping for repulsive energy.
    This allows the system to settle into a more uniform distribution.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    pts = points.copy()
    velocities = np.zeros_like(pts)
    best_pts = pts.copy()
    best_energy = _repulsive_energy(pts, s=s)
    
    for it in range(10000):
        if time.time() > deadline - 1.0:
            break
        
        # Compute forces (negative gradient of repulsive energy)
        dots = np.dot(pts, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        dist_sq = 2.0 - 2.0 * dots
        np.fill_diagonal(dist_sq, 1.0)
        dist_sq = np.maximum(dist_sq, 1e-10)
        
        # Force: -d/dx_i sum_{j!=i} (2-2*x_i.x_j)^(-s/2)
        # = -sum_{j!=i} (-s/2)(2-2*x_i.x_j)^(-s/2-1) * (-2*x_j)
        # = -s * sum_{j!=i} (2-2*x_i.x_j)^(-s/2-1) * x_j
        # But we need to project onto tangent space
        
        weights = dist_sq ** (-s/2.0 - 1.0)
        np.fill_diagonal(weights, 0.0)
        
        forces = -s * np.dot(weights, pts)
        
        # Project forces onto tangent space
        for i in range(N):
            forces[i] -= np.dot(forces[i], pts[i]) * pts[i]
        
        # Velocity Verlet
        velocities = velocities * damping + forces * dt
        pts_new = pts + velocities * dt
        pts_new = _normalize_points(pts_new)
        
        energy = _repulsive_energy(pts_new, s=s)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_new.copy()
        
        pts = pts_new
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius, _, _ = _compute_covering_radius(points)
    
    # Phase 1: Global Gradient Descent with increasing beta
    if time.time() < deadline - 60.0:
        points = _gradient_descent_phase(points, time_limit=20.0, beta=10.0)
        current_radius, _, _ = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        if time.time() < deadline - 40.0:
            points = _gradient_descent_phase(points, time_limit=15.0, beta=30.0)
            current_radius, _, _ = _compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
                
        if time.time() < deadline - 20.0:
            points = _gradient_descent_phase(points, time_limit=10.0, beta=100.0)
            current_radius, _, _ = _compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
        
    # Phase 1.5: Velocity Verlet refinement with softer potential (s=4)
    if time.time() < deadline - 15.0:
        vv_points = _velocity_verlet_refine(best_points, time_limit=15.0, s=4.0)
        vv_radius, _, _ = _compute_covering_radius(vv_points)
        if vv_radius < best_radius:
            best_radius = vv_radius
            best_points = vv_points.copy()
            points = vv_points.copy()
            
            # Follow up with facet-based polish
            if time.time() < deadline - 5.0:
                points = _gradient_descent_phase(points, time_limit=5.0, beta=100.0)
                final_radius, _, _ = _compute_covering_radius(points)
                if final_radius < best_radius:
                    best_radius = final_radius
                    best_points = points.copy()
        
    # Phase 2: Local Refinement (High Precision Polish)
    if time.time() < deadline - 5.0:
        refined_points, refined_radius = _local_refinement(best_points, time_limit=25.0)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_points.copy()

    if np.all(np.isfinite(best_points)) and np.all(np.linalg.norm(best_points, axis=1) > 1e-9):
        return best_points
    else:
        return points
# EVOLVE-BLOCK-END
