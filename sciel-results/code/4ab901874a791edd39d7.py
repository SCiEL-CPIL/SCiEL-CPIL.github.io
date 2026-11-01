import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 100
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _energy_and_grad(points):
    """
    Compute logarithmic energy and its gradient.
    E = sum_{i<j} log(1/||x_i - x_j||)
    dE/dx_i = -sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Gradient calculation
    # grad[i] = -sum_j (x_i - x_j) / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    # Energy calculation using squared distances (faster)
    iu = np.triu_indices(N, k=1)
    d_sq_upper = dist_sq[iu]
    d_sq_upper = np.maximum(d_sq_upper, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq_upper))
    
    return energy, grad

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosahedron_subdivide():
    """Generate points via icosahedron subdivision for better initial spread."""
    # Golden ratio
    t = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Icosahedron vertices
    verts = np.array([
        [-1,  t,  0.], [ 1,  t,  0.], [-1, -t,  0.], [ 1, -t,  0.],
        [ 0., -1,  t], [ 0.,  1,  t], [ 0., -1, -t], [ 0.,  1, -t],
        [ t,  0., -1], [ t,  0.,  1], [-t,  0., -1], [-t,  0.,  1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Faces of icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    points = list(verts)
    
    # Subdivide until we have enough points
    while len(points) < N:
        new_points = []
        for face in faces:
            a, b, c = points[face[0]], points[face[1]], points[face[2]]
            # Midpoints
            ab = (a + b) / np.linalg.norm(a + b)
            bc = (b + c) / np.linalg.norm(b + c)
            ca = (c + a) / np.linalg.norm(c + a)
            new_points.extend([ab, bc, ca])
        
        # Add new points and create new faces
        # This is a simplified version - just collect all unique points
        all_pts = list(points) + new_points
        
        # Remove duplicates (points that are very close)
        unique_pts = [all_pts[0]]
        for p in all_pts[1:]:
            is_dup = False
            for u in unique_pts:
                if np.linalg.norm(p - u) < 1e-10:
                    is_dup = True
                    break
            if not is_dup:
                unique_pts.append(p)
        
        points = unique_pts
        if len(points) >= N:
            break
    
    # Take first N points
    result = np.array(points[:N])
    result = result / np.linalg.norm(result, axis=1, keepdims=True)
    return result

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _steepest_descent_sphere(points, max_iter=2000, tol=1e-14, deadline=None):
    """
    Steepest descent on the sphere with adaptive step size and backtracking.
    Uses tangent-space gradient projection.
    """
    p = points.copy()
    e, g = _energy_and_grad(p)
    best_e = e
    best_p = p.copy()
    
    step_size = 0.05
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Project gradient to tangent space
        g_tan = g - (g * p).sum(axis=1, keepdims=True) * p
        
        # Check convergence
        g_norm = np.linalg.norm(g_tan)
        if g_norm < tol:
            break
        
        # Backtracking line search (Armijo condition)
        step = step_size
        accepted = False
        c1 = 1e-4  # Armijo constant
        
        for _ in range(20):
            p_new = p - step * g_tan
            p_new = _project_to_sphere(p_new)
            e_new, _ = _energy_and_grad(p_new)
            
            if e_new < e - c1 * step * np.sum(g_tan**2):
                p = p_new
                e = e_new
                if e < best_e:
                    best_e = e
                    best_p = p.copy()
                _, g = _energy_and_grad(p)
                accepted = True
                break
            step *= 0.5
        
        if not accepted:
            # Try smaller initial step for next iteration
            step_size *= 0.5
            if step_size < 1e-10:
                break
        else:
            # Gradually increase step size if accepted
            step_size = min(step_size * 1.1, 0.1)
    
    return best_p, best_e

def _lbfgs_optimize(points, max_iter=2000, deadline=None):
    """Run L-BFGS-B optimization on sphere coordinates."""
    def _objective_and_grad(x):
        pts = x.reshape(N, 3)
        pts_sphere = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts_sphere)
        
        # Add penalty for leaving the sphere
        lambda_pen = 100.0
        diff_pen = pts - pts_sphere
        penalty = 0.5 * lambda_pen * np.sum(diff_pen**2)
        grad_pen = lambda_pen * diff_pen
        
        total_grad = (g + grad_pen).flatten()
        return e + penalty, total_grad
    
    x0 = points.flatten()
    
    try:
        res = minimize(_objective_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        
        if hasattr(res, 'x') and res.x is not None:
            new_points = _project_to_sphere(res.x.reshape(N, 3))
            e_new, _ = _energy_and_grad(new_points)
            return new_points, e_new
    except Exception:
        pass
    
    return points, _energy_and_grad(points)[0]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = None
    best_energy = float('inf')
    
    # Try multiple initializations
    initializations = [
        ("fibonacci", _fibonacci_spiral()),
        ("random_1", _random_points(seed=42)),
        ("random_2", _random_points(seed=123)),
        ("random_3", _random_points(seed=456)),
        ("icosahedron", _icosahedron_subdivide()),
    ]
    
    for name, init_pts in initializations:
        if time.time() > deadline - 10.0:
            break
        
        # L-BFGS optimization
        opt_pts, opt_e = _lbfgs_optimize(init_pts, max_iter=1500, deadline=deadline)
        
        if opt_e < best_energy:
            best_energy = opt_e
            best_points = opt_pts.copy()
        
        # Steepest descent polish
        if time.time() > deadline - 5.0:
            break
        polished, e_polished = _steepest_descent_sphere(
            opt_pts, max_iter=3000, tol=1e-15, deadline=deadline - 2.0
        )
        if e_polished < best_energy:
            best_energy = e_polished
            best_points = polished.copy()
    
    # Final validation and cleanup
    if best_points is None:
        best_points = _fibonacci_spiral()
    
    best_points = _project_to_sphere(best_points)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                rng = np.random.RandomState(i * 1000 + j)
                best_points[i] += rng.randn(3) * 1e-6
                best_points[i] = _project_to_sphere(best_points[i].reshape(1, 3))[0]
    
    return best_points
