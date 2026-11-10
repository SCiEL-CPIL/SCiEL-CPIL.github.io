"""
Sphere covering with 130 points - Smooth Energy Minimization.
Strategy:
1. Initialize with a Fibonacci lattice.
2. Define a smooth, differentiable objective function: the sum of squared 
   angular distances from a dense set of probe points on the sphere to their 
   nearest configuration point. This avoids the non-differentiability and 
   computational cost of convex hulls.
3. Use L-BFGS-B optimizer to minimize this energy. The gradient naturally 
   pushes points away from high-density coverage gaps (where probe points 
   are far from any config point) and towards balancing the load.
4. Perform multiple restarts with different initializations if time permits, 
   keeping the best result.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 130

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_exact(points):
    """
    Compute the exact covering radius in degrees using convex hull.
    This is used only for final evaluation and comparison, not for optimization.
    """
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_r = 0.0
        
        for simplex in facets:
            pts = points[simplex]
            if len(pts) < 3:
                continue
            # For a spherical triangle, the circumcenter is the projection of the origin
            # onto the plane defined by the three points? No, the circumcenter of a spherical triangle
            # is the point on the sphere equidistant from the vertices.
            # It is the normalized cross product of (b-a) and (c-a) only if we are talking about the plane normal.
            # Actually, the circumcenter of a spherical triangle ABC is the intersection of the sphere
            # with the line passing through the origin and the vector:
            # v = (b x c) x (c x a) ... no.
            
            # Easier: The circumradius of a spherical triangle is the distance from the circumcenter to a vertex.
            # The circumcenter is the normalized vector:
            # n = (a x b) x (b x c) ... no.
            
            # Let's use the property that the circumcenter C satisfies C.a = C.b = C.c = cos(R).
            # C is proportional to (b x c) x (c x a) ? No.
            # C is proportional to (a x b) + (b x c) + (c x a) ? No.
            
            # Correct formula: The circumcenter of a spherical triangle with vertices a, b, c is:
            # c_center = ( (b x c) x (c x a) ) + ... no.
            
            # Let's just find the point on the sphere equidistant to a, b, c.
            # This is the solution to:
            # |C - a|^2 = |C - b|^2 = |C - c|^2 = const
            # Since |C|=|a|=|b|=|c|=1, this simplifies to:
            # C.a = C.b = C.c
            
            # Let u = a - b, v = a - c.
            # C . u = 0 and C . v = 0.
            # So C is parallel to u x v = (a-b) x (a-c) = a x a - a x c - b x a + b x c = b x a - a x c + b x c = -a x b - a x c + b x c.
            # Let w = (a - b) x (a - c).
            # Then C = w / |w|.
            
            u = a - b
            v = a - c
            w = np.cross(u, v)
            norm_w = np.linalg.norm(w)
            if norm_w < 1e-15:
                continue
            C = w / norm_w
            
            # Check if C is on the correct side (inside the triangle)
            # The circumcenter can be outside the triangle if the triangle is obtuse.
            # However, for the covering radius, we care about the maximum distance to the nearest point.
            # The farthest point in the facet from the vertices is the circumcenter.
            # If the triangle is obtuse, the circumcenter is outside, and the farthest point is one of the vertices?
            # No, the covering radius is the max distance to the NEAREST point.
            # For a facet, the region covered is the spherical triangle.
            # The point in the triangle farthest from the vertices is the circumcenter (if acute) or the midpoint of the longest side (if obtuse).
            # Actually, the circumradius of the facet is the radius of the smallest circle covering the facet.
            # This is the circumradius if the triangle is acute, or half the longest side if obtuse.
            
            # Distance from C to a:
            cos_R = np.clip(np.dot(C, a), -1.0, 1.0)
            R = np.arccos(cos_R)
            
            # Check if triangle is obtuse.
            # Compute side lengths (angular distances)
            d_ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))
            
            sides = [d_ab, d_bc, d_ca]
            sides.sort()
            
            # If the largest side is >= 90 degrees, the circumcenter is outside or on the boundary,
            # and the covering radius of the facet is half the largest side.
            if sides[2] >= np.pi / 2:
                R = sides[2] / 2.0
            else:
                R = R # Circumradius
            
            max_r = max(max_r, R)
        
        return np.degrees(max_r)
    except Exception:
        return np.inf

def _smooth_objective(points_flat):
    """
    Compute the smooth objective: sum of squared angular distances from probe points to nearest config point.
    Also returns the gradient.
    """
    points = points_flat.reshape(N, 3)
    
    # Normalize points to ensure they stay on the sphere during optimization
    # (L-BFGS-B might drift slightly)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Generate probe points on the sphere.
    # We need a set of points dense enough to approximate the max distance well.
    # Using a Fibonacci lattice for probes is efficient.
    # Let's use 2000 probe points.
    n_probes = 2000
    probes = _fibonacci_sphere(n_probes)
    
    # Compute distances from each probe to each config point
    # probes: (P, 3), points: (N, 3)
    # dot product: (P, N)
    dots = probes @ points.T
    
    # Clip to [-1, 1] for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    
    # Angles: arccos(dots) -> (P, N)
    # We want the minimum angle for each probe
    angles = np.arccos(dots)
    min_angles = np.min(angles, axis=1) # (P,)
    
    # Objective: Sum of squared min angles
    # This penalizes large gaps quadratically, focusing on the worst areas.
    # A sum of squares is smoother than a max.
    obj = np.sum(min_angles ** 2)
    
    # Gradient calculation
    # d(obj)/d(points) = sum_probes d(obj)/d(min_angle_p) * d(min_angle_p)/d(points)
    
    # d(obj)/d(min_angle_p) = 2 * min_angle_p
    
    # d(min_angle_p)/d(points_j)
    # min_angle_p = min_j arccos(probe_p . point_j)
    # Let j* be the index of the nearest point for probe p.
    # Then min_angle_p = arccos(probe_p . point_{j*})
    # d(min_angle_p)/d(point_{j*}) = - probe_p / sqrt(1 - (probe_p . point_{j*})^2)
    # Note: sqrt(1 - cos^2) = sin(theta).
    # So d(min_angle_p)/d(point_{j*}) = - probe_p / sin(theta_p)
    
    # We need to accumulate gradients for each point_j.
    # For each probe p, find j*.
    min_indices = np.argmin(angles, axis=1) # (P,)
    
    # We need sin(theta_p) for each probe.
    # cos(theta_p) = dots[p, min_indices[p]]
    cos_theta = dots[np.arange(n_probes), min_indices]
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 1e-15, 1.0))
    
    # Gradient contribution for each probe p to point j*:
    # d(obj)/d(point_{j*}) += 2 * min_angle_p * (- probe_p / sin_theta_p)
    
    grad = np.zeros_like(points)
    
    # Vectorized accumulation
    # We can use np.add.at
    # contrib_p = -2 * min_angles / sin_theta  (scalar for each probe)
    contrib_coeff = -2.0 * min_angles / sin_theta
    
    # For each probe p, add contrib_coeff[p] * probes[p] to grad[min_indices[p]]
    np.add.at(grad, min_indices, contrib_coeff[:, np.newaxis] * probes)
    
    # The gradient is with respect to the flat array.
    # Since we normalized points inside the function, the gradient w.r.t the raw flat array
    # is slightly different. However, L-BFGS-B works on the raw variables.
    # If we force normalization inside the function, the gradient returned should be
    # consistent with the function value.
    # The chain rule for normalization:
    # y = x / ||x||
    # dy/dx = (I - y y^T) / ||x||
    # So grad_raw = (I - y y^T) * grad_y / ||x||
    
    # Let's apply this correction.
    # grad currently is grad_y (gradient w.r.t normalized points).
    # We need grad_x.
    
    # y = points (normalized)
    # x = points_flat reshaped (unnormalized)
    
    # Actually, it's easier to just let the optimizer work on the unit sphere directly
    # by using a parameterization that guarantees unit length, e.g., spherical coordinates.
    # But L-BFGS-B doesn't support constraints well.
    # Alternative: Return the gradient w.r.t the normalized points and hope the optimizer
    # handles the implicit constraint? No, that's wrong.
    
    # Better approach: Use the fact that the objective is rotationally invariant? No.
    # Let's just compute the gradient w.r.t the unnormalized coordinates properly.
    
    # grad_y is the gradient w.r.t the normalized points Y.
    # Y = X / ||X||
    # dL/dX = (I - Y Y^T) dL/dY / ||X||
    
    # X is the input points_flat reshaped.
    X = points_flat.reshape(N, 3)
    norms_X = np.linalg.norm(X, axis=1, keepdims=True)
    Y = X / norms_X
    
    # dL/dY = grad (computed above)
    
    # dL/dX = (I - Y Y^T) grad / norms_X
    # (I - Y Y^T) grad = grad - Y * (Y . grad)
    y_dot_grad = np.sum(Y * grad, axis=1, keepdims=True)
    grad_X = (grad - Y * y_dot_grad) / norms_X
    
    return obj, grad_X.flatten()

def _optimize(points_init, max_time):
    """
    Run L-BFGS-B optimization starting from points_init.
    """
    start_time = time.time()
    
    # Flatten points
    x0 = points_init.flatten()
    
    # Optimization options
    options = {
        'maxiter': 100, # Limit iterations to check time
        'ftol': 1e-8,
        'gtol': 1e-8,
        'maxls': 50
    }
    
    # We need to check time inside the callback
    def callback(xk):
        return time.time() - start_time > max_time - 5
    
    try:
        res = minimize(
            _smooth_objective, 
            x0, 
            method='L-BFGS-B', 
            jac=True,
            options=options,
            callback=callback
        )
    except Exception as e:
        # If optimization fails, return the initial points
        return points_init, _covering_radius_exact(points_init)
    
    best_points = res.x.reshape(N, 3)
    # Normalize again to be safe
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    best_radius = _covering_radius_exact(best_points)
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess
    points_init = _fibonacci_sphere(N)
    
    best_points = points_init.copy()
    best_radius = _covering_radius_exact(best_points)
    
    # Perform a few restarts with different initializations if time allows
    # Restart 1: Fibonacci
    if time.time() < deadline - 30.0:
        p, r = _optimize(points_init, max_time=30.0)
        if r < best_radius:
            best_points = p
            best_radius = r
    
    # Restart 2: Perturbed Fibonacci
    if time.time() < deadline - 30.0:
        noise = np.random.randn(N, 3) * 0.1
        p_init = points_init + noise
        p_init = p_init / np.linalg.norm(p_init, axis=1, keepdims=True)
        p, r = _optimize(p_init, max_time=30.0)
        if r < best_radius:
            best_points = p
            best_radius = r
    
    # Restart 3: Random (if time permits)
    if time.time() < deadline - 30.0:
        p_init = np.random.randn(N, 3)
        p_init = p_init / np.linalg.norm(p_init, axis=1, keepdims=True)
        p, r = _optimize(p_init, max_time=30.0)
        if r < best_radius:
            best_points = p
            best_radius = r
            
    # Final local refinement with a very short L-BFGS run on the best point
    if time.time() < deadline - 10.0:
        p, r = _optimize(best_points, max_time=10.0)
        if r < best_radius:
            best_points = p
            best_radius = r
            
    return best_points
# EVOLVE-BLOCK-END
