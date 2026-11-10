"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START

def _covering_radius(points):
    """
    Compute the covering radius in radians using the convex hull approach.
    The covering radius equals the maximum circumradius of the spherical Delaunay triangles.
    """
    if len(points) < 3:
        return np.pi / 2
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        return np.pi
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2
    
    # Vectorized circumcenter calculation for all faces
    faces = hull.simplices
    p0 = points[faces[:, 0]]
    p1 = points[faces[:, 1]]
    p2 = points[faces[:, 2]]
    
    A = p1 - p0
    B = p2 - p0
    
    A_A = np.einsum('ij,ij->i', A, A)
    A_B = np.einsum('ij,ij->i', A, B)
    B_B = np.einsum('ij,ij->i', B, B)
    
    det = A_A * B_B - A_B * A_B
    
    # Solve for u, v: O = p0 + u*A + v*B
    u = np.zeros_like(A_A)
    v = np.zeros_like(A_A)
    
    valid = np.abs(det) > 1e-12
    if np.any(valid):
        u[valid] = (0.5 * A_A[valid] * B_B[valid] - 0.5 * A_B[valid] * B_B[valid]) / det[valid]
        v[valid] = (0.5 * A_B[valid] * A_A[valid] - 0.5 * A_A[valid] * B_B[valid]) / det[valid]
    
    # For degenerate cases, use centroid
    degenerate = ~valid
    if np.any(degenerate):
        centroid = (p0[degenerate] + p1[degenerate] + p2[degenerate]) / 3.0
        centroid_norms = np.linalg.norm(centroid, axis=1, keepdims=True)
        safe_norms = np.where(centroid_norms < 1e-12, 1.0, centroid_norms)
        centroid_normalized = centroid / safe_norms
        # Place degenerate circumcenters at normalized centroids
        O_degenerate = centroid_normalized
    else:
        O_degenerate = np.empty((0, 3))
    
    O_valid = p0[valid] + u[valid, None] * A[valid] + v[valid, None] * B[valid]
    
    # Combine
    O = np.empty((len(faces), 3))
    O[valid] = O_valid
    O[degenerate] = O_degenerate
    
    # Normalize to get spherical circumcenters
    O_norms = np.linalg.norm(O, axis=1, keepdims=True)
    safe_norms = np.where(O_norms < 1e-12, 1.0, O_norms)
    C = O / safe_norms
    
    # Calculate angular distances from circumcenter to each vertex
    d0 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p0), -1, 1))
    d1 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p1), -1, 1))
    d2 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p2), -1, 1))
    
    r_face = np.maximum(np.maximum(d0, d1), d2)
    
    return float(np.max(r_face))

def _covering_radius_with_grad(points):
    """
    Compute covering radius and provide a subgradient for optimization.
    Returns (radius, gradient) where gradient is of shape (N, 3).
    """
    if len(points) < 3:
        return np.pi / 2, np.zeros((N, 3))
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        return np.pi, np.zeros((N, 3))
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2, np.zeros((N, 3))
    
    faces = hull.simplices
    p0 = points[faces[:, 0]]
    p1 = points[faces[:, 1]]
    p2 = points[faces[:, 2]]
    
    A = p1 - p0
    B = p2 - p0
    
    A_A = np.einsum('ij,ij->i', A, A)
    A_B = np.einsum('ij,ij->i', A, B)
    B_B = np.einsum('ij,ij->i', B, B)
    
    det = A_A * B_B - A_B * A_B
    
    u = np.zeros_like(A_A)
    v = np.zeros_like(A_A)
    
    valid = np.abs(det) > 1e-12
    if np.any(valid):
        u[valid] = (0.5 * A_A[valid] * B_B[valid] - 0.5 * A_B[valid] * B_B[valid]) / det[valid]
        v[valid] = (0.5 * A_B[valid] * A_A[valid] - 0.5 * A_A[valid] * B_B[valid]) / det[valid]
    
    degenerate = ~valid
    if np.any(degenerate):
        centroid = (p0[degenerate] + p1[degenerate] + p2[degenerate]) / 3.0
        centroid_norms = np.linalg.norm(centroid, axis=1, keepdims=True)
        safe_norms = np.where(centroid_norms < 1e-12, 1.0, centroid_norms)
        centroid_normalized = centroid / safe_norms
        O_degenerate = centroid_normalized
    else:
        O_degenerate = np.empty((0, 3))
    
    O_valid = p0[valid] + u[valid, None] * A[valid] + v[valid, None] * B[valid]
    
    O = np.empty((len(faces), 3))
    O[valid] = O_valid
    O[degenerate] = O_degenerate
    
    O_norms = np.linalg.norm(O, axis=1, keepdims=True)
    safe_norms = np.where(O_norms < 1e-12, 1.0, O_norms)
    C = O / safe_norms
    
    d0 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p0), -1, 1))
    d1 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p1), -1, 1))
    d2 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p2), -1, 1))
    
    r_face = np.maximum(np.maximum(d0, d1), d2)
    max_idx = np.argmax(r_face)
    max_r = r_face[max_idx]
    
    # Subgradient: for the face with max radius, push points apart
    grad = np.zeros((N, 3))
    face = faces[max_idx]
    idx0, idx1, idx2 = face[0], face[1], face[2]
    
    # The circumcenter C should move away from the triangle vertices
    # Gradient with respect to each point: move point away from C
    c = C[max_idx]
    
    # For each vertex, the gradient direction is away from the circumcenter
    # But we need to be careful: moving a point changes the circumcenter
    # Approximation: move each point in the direction from C to the point
    for idx in [idx0, idx1, idx2]:
        p = points[idx]
        # Direction from C to p
        diff = p - c
        diff_norm = np.linalg.norm(diff)
        if diff_norm > 1e-12:
            grad[idx] += diff / diff_norm
    
    # Normalize gradient
    grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
    safe_grad_norm = np.where(grad_norm < 1e-12, 1.0, grad_norm)
    grad = grad / safe_grad_norm
    
    return float(max_r), grad

def _optimize_l_bfgs(start_points, time_limit, rng):
    """
    Optimize using L-BFGS-B on the sphere using a parameterization.
    We use the fact that we can represent points via their angles.
    """
    points = start_points.copy()
    
    # Convert to spherical coordinates for optimization
    # Actually, let's use a different approach: optimize in Cartesian with projection
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    start_time = time.time()
    
    # Use Nelder-Mead or L-BFGS with a penalty for off-sphere points
    def objective_and_grad(x):
        if time.time() - start_time > time_limit:
            return best_cr, np.zeros_like(x)
        
        # Reshape to (N, 3)
        pts = x.reshape(N, 3)
        
        # Project to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        safe_norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / safe_norms
        
        # Compute covering radius
        cr = _covering_radius(pts)
        
        # Simple gradient: use subgradient from the worst face
        _, grad = _covering_radius_with_grad(pts)
        
        return cr, grad.flatten()
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < best_cr:
            new_points = result.x.reshape(N, 3)
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            safe_norms = np.where(norms < 1e-12, 1.0, norms)
            new_points = new_points / safe_norms
            new_cr = _covering_radius(new_points)
            
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = new_points.copy()
    except Exception:
        pass
    
    return best_points, best_cr

def _hill_climb_optimize(start_points, time_limit, rng, initial_step=0.05):
    """
    Local optimization via hill-climbing with adaptive step size.
    """
    points = start_points.copy()
    current_cr = _covering_radius(points)
    best_points = points.copy()
    best_cr = current_cr
    
    start_time = time.time()
    
    step_size = initial_step
    min_step = 1e-6
    max_iter = 100000
    no_improve_count = 0
    no_improve_limit = 300
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        idx = rng.integers(N)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        
        improved = False
        for sign in [1, -1]:
            new_point = points[idx] + sign * step_size * direction
            new_norm = np.linalg.norm(new_point)
            if new_norm < 1e-12:
                continue
            new_point /= new_norm
            
            old_point = points[idx]
            points[idx] = new_point
            
            new_cr = _covering_radius(points)
            
            if new_cr < current_cr - 1e-10:
                current_cr = new_cr
                improved = True
                if new_cr < best_cr:
                    best_cr = new_cr
                    best_points = points.copy()
                step_size = min(step_size * 1.02, 0.3)
                no_improve_count = 0
                break
            else:
                points[idx] = old_point
        
        if not improved:
            no_improve_count += 1
            step_size = max(step_size * 0.97, min_step)
            
            if no_improve_count > no_improve_limit:
                step_size = rng.uniform(0.02, 0.15)
                no_improve_count = 0
        
        if step_size < min_step * 1.01:
            break
    
    return best_points, best_cr

def construct():
    total_deadline = 145.0
    start_time = time.time()
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_cr = np.pi
    
    # Strategy 1: Fibonacci spiral + L-BFGS + hill climb
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 10.0
    
    if remaining > 5.0:
        # L-BFGS phase
        lbfgs_time = min(40.0, remaining * 0.4)
        fib_points, fib_cr = _optimize_l_bfgs(fib_points, lbfgs_time, rng)
        
        elapsed = time.time() - start_time
        remaining = total_deadline - elapsed - 10.0
        
        if fib_cr < best_cr:
            best_cr = fib_cr
            best_points = fib_points.copy()
        
        # Hill climb phase
        if remaining > 5.0:
            hc_time = min(30.0, remaining * 0.6)
            fib_points, fib_cr = _hill_climb_optimize(fib_points, hc_time, rng, initial_step=0.03)
            
            if fib_cr < best_cr:
                best_cr = fib_cr
                best_points = fib_points.copy()
    
    # Strategy 2: Multiple random restarts with hill climbing
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 10.0
    
    if remaining > 5.0:
        num_restarts = 6
        time_per_restart = remaining / num_restarts
        
        for i_restart in range(num_restarts):
            if time.time() - start_time > total_deadline - 10.0:
                break
            
            # Diverse random starts
            if i_restart % 3 == 0:
                # Uniform random
                random_points = rng.normal(size=(N, 3))
                random_points /= np.linalg.norm(random_points, axis=1, keepdims=True)
            elif i_restart % 3 == 1:
                # Slightly perturbed Fibonacci
                perturbed = fib_points + rng.normal(scale=0.1, size=(N, 3))
                perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
                random_points = perturbed
            else:
                # Grid-based with jitter
                lat = np.linspace(-np.pi/2, np.pi/2, int(np.sqrt(N)) + 1)
                lon = np.linspace(0, 2*np.pi, int(np.sqrt(N)) + 1)
                L, O = np.meshgrid(lat, lon)
                grid_pts = np.column_stack([
                    np.cos(L) * np.cos(O).ravel(),
                    np.cos(L) * np.sin(O).ravel(),
                    np.sin(L).ravel()
                ])
                # Resample to N points
                if len(grid_pts) >= N:
                    indices = rng.choice(len(grid_pts), N, replace=False)
                    random_points = grid_pts[indices]
                else:
                    random_points = rng.normal(size=(N, 3))
                    random_points /= np.linalg.norm(random_points, axis=1, keepdims=True)
            
            # Hill climb
            alloc_time = min(time_per_restart, total_deadline - (time.time() - start_time) - 10.0)
            if alloc_time < 1.0:
                break
            
            init_step = rng.uniform(0.02, 0.1)
            rand_points, rand_cr = _hill_climb_optimize(random_points, alloc_time, rng, initial_step=init_step)
            
            if rand_cr < best_cr:
                best_cr = rand_cr
                best_points = rand_points.copy()
    
    # Final refinement with small steps
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = total_deadline - elapsed - 5.0
        if remaining > 2.0:
            best_points, best_cr = _hill_climb_optimize(best_points, remaining, rng, initial_step=0.01)
    
    # Final normalization
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
