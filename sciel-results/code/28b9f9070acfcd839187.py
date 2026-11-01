"""Stronger optimizer for 100-point logarithmic energy on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 138.0  # safety margin before 150s kill


def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    # log(1/d) = -0.5 * log(d^2)
    return -0.5 * math.fsum(np.log(d_sq).tolist())


def _energy_grad_fast(points):
    """Compute energy and gradient of logarithmic energy using vectorized ops."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    dist_sq_safe = dist_sq.copy()
    np.fill_diagonal(dist_sq_safe, 1.0)
    
    # inv_dist_sq[i,j] = 1/||x_i - x_j||^2
    inv_dist_sq = 1.0 / dist_sq_safe
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j inv_dist_sq[i,j] * (x_i - x_j)
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    # Energy: sum over i<j of -log(dist_ij) = -0.5 * sum log(dist_sq_ij)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    energy = -0.5 * math.fsum(np.log(d_sq).tolist())
    
    return energy, grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.sqrt(np.sum(points * points, axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _sphere_fun_grad(x):
    """Objective and gradient for scipy minimize in flat R^(3N) space."""
    p = _project_to_sphere(x.reshape(-1, 3))
    e, g = _energy_grad_fast(p)
    # Project gradient to tangent space
    g_tangent = g - (g * p).sum(axis=1, keepdims=True) * p
    return e, g_tangent.flatten()


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _spherical_design_init(seed=42):
    """Generate initial points using a quasi-random spherical design."""
    rng = np.random.RandomState(seed)
    pts = _fibonacci_spiral()
    noise = rng.randn(N, 3) * 0.01
    pts = pts + noise
    return _project_to_sphere(pts)


def _random_points(seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)


def _riemannian_flow(points, max_iter=5000, tol=1e-12, time_budget=15.0):
    """Riemannian gradient flow with adaptive step size on the sphere."""
    start_time = time.time()
    p = points.copy()
    e, g = _energy_grad_fast(p)
    best_e = e
    best_p = p.copy()
    
    step_size = 0.05
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        g_tan = g - (g * p).sum(axis=1, keepdims=True) * p
        grad_norm = np.linalg.norm(g_tan)
        
        if grad_norm < tol:
            break
        
        target_step = min(0.1, 0.5 / (grad_norm + 1e-8))
        step_size = 0.8 * step_size + 0.2 * target_step
        
        best_e_new = e
        best_p_new = p
        best_g_new = g
        
        for alpha in [step_size, step_size * 0.5, step_size * 0.25, step_size * 2.0]:
            p_new = p - alpha * g_tan
            p_new = _project_to_sphere(p_new)
            e_new, g_new = _energy_grad_fast(p_new)
            if e_new < best_e_new:
                best_e_new = e_new
                best_p_new = p_new
                best_g_new = g_new
        
        if best_e_new >= e:
            step_size *= 0.5
            if step_size < 1e-8:
                break
            continue
        
        p = best_p_new
        e = best_e_new
        g = best_g_new
        
        if e < best_e:
            best_e = e
            best_p = p.copy()
    
    return best_p, best_e


def _lbfgs_optimize(points, max_iter=5000, time_budget=20.0):
    """L-BFGS-B optimization from given initial points."""
    start_time = time.time()
    x0 = points.flatten().copy()
    
    try:
        res = minimize(_sphere_fun_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        
        if hasattr(res, 'x'):
            new_points = _project_to_sphere(res.x.reshape(-1, 3))
            new_energy = _energy(new_points)
            return new_points, new_energy
    except Exception:
        pass
    
    return points, _energy(points)


def _energy_delta_move(points, idx, new_point):
    """Compute the change in energy when moving point idx to new_point."""
    old_point = points[idx]
    diff_old = points - old_point
    diff_new = points - new_point
    
    d_sq_old = np.sum(diff_old * diff_old, axis=1)
    d_sq_new = np.sum(diff_new * diff_new, axis=1)
    
    d_sq_old[idx] = 1.0
    d_sq_new[idx] = 1.0
    
    d_sq_old = np.maximum(d_sq_old, 1e-30)
    d_sq_new = np.maximum(d_sq_new, 1e-30)
    
    delta_e = np.sum(np.log(d_sq_old) - np.log(d_sq_new)) * 0.5
    return delta_e


def _simulated_annealing_refine(points, time_budget=15.0):
    """Simulated annealing refinement for logarithmic energy with adaptive moves."""
    start_time = time.time()
    p = points.copy()
    e = _energy(p)
    best_p = p.copy()
    best_e = e
    
    T = 0.5
    T_min = 1e-6
    cooling = 0.9995
    
    move_scale = 0.1
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        idx = np.random.randint(N)
        original = p[idx].copy()
        
        if np.random.random() < 0.05:
            delta = np.random.randn(3) * (move_scale * 5)
        else:
            delta = np.random.randn(3) * move_scale
        
        new_point = original + delta
        norm = np.linalg.norm(new_point)
        if norm < 1e-10:
            continue
        new_point = new_point / norm
        
        delta_e = _energy_delta_move(p, idx, new_point)
        
        if delta_e < 0 or (T > 0 and np.random.random() < math.exp(-delta_e / T)):
            p[idx] = new_point
            e += delta_e
            if e < best_e:
                best_e = e
                best_p = p.copy()
            move_scale = min(0.5, move_scale * 1.01)
        else:
            move_scale = max(0.01, move_scale * 0.99)
        
        T *= cooling
    
    return best_p, best_e


def _hilbert_curve_init():
    """Generate initial points using a Hilbert curve mapped to the sphere.
    
    The Hilbert curve provides a space-filling ordering that can guide
    the initial placement to be more globally consistent.
    """
    # Generate a 1D Hilbert curve index sequence
    order = 8  # 2^8 = 256 points, we'll take 100
    max_idx = 2 ** order
    
    # Generate Hilbert curve coordinates in 3D
    # We'll use a simplified approach: generate points along a 3D Hilbert curve
    # and map them to the sphere
    
    # For simplicity, we'll use a parametric curve that has good space-filling properties
    # This is a Lissajous-like curve with irrational frequency ratios
    t = np.linspace(0, 2 * np.pi * 10, N, endpoint=False)
    
    # Use multiple frequencies to fill space more uniformly
    x = np.sin(t) + 0.5 * np.sin(2.7 * t) + 0.3 * np.sin(4.1 * t)
    y = np.cos(1.3 * t) + 0.5 * np.cos(3.1 * t) + 0.3 * np.cos(5.2 * t)
    z = np.sin(0.7 * t) + 0.5 * np.sin(2.3 * t) + 0.3 * np.sin(4.7 * t)
    
    pts = np.column_stack([x, y, z])
    return _project_to_sphere(pts)


def _hilbert_guided_optimize(points, time_budget=20.0):
    """Optimize points with guidance from Hilbert curve ordering.
    
    This uses the Hilbert curve to define a local neighborhood structure
    and biases the search toward moves that maintain global consistency.
    """
    start_time = time.time()
    p = points.copy()
    e, g = _energy_grad_fast(p)
    best_e = e
    best_p = p.copy()
    
    # Compute Hilbert curve ordering
    t = np.linspace(0, 2 * np.pi * 10, N, endpoint=False)
    x = np.sin(t) + 0.5 * np.sin(2.7 * t) + 0.3 * np.sin(4.1 * t)
    y = np.cos(1.3 * t) + 0.5 * np.cos(3.1 * t) + 0.3 * np.cos(5.2 * t)
    z = np.sin(0.7 * t) + 0.5 * np.sin(2.3 * t) + 0.3 * np.sin(4.7 * t)
    hilbert_pts = _project_to_sphere(np.column_stack([x, y, z]))
    
    # Compute distances from current points to Hilbert curve points
    # This gives us a "locality" measure
    diff = p[:, np.newaxis, :] - hilbert_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # For each point, find its nearest Hilbert curve neighbor
    nearest_idx = np.argmin(dists, axis=1)
    
    # Use this to guide the optimization: prefer moves that maintain
    # proximity to the Hilbert curve structure
    step_size = 0.05
    
    for it in range(10000):
        if time.time() - start_time > time_budget:
            break
        
        g_tan = g - (g * p).sum(axis=1, keepdims=True) * p
        grad_norm = np.linalg.norm(g_tan)
        
        if grad_norm < 1e-12:
            break
        
        # Adaptive step size
        target_step = min(0.1, 0.5 / (grad_norm + 1e-8))
        step_size = 0.8 * step_size + 0.2 * target_step
        
        # Try multiple step sizes
        best_e_new = e
        best_p_new = p
        best_g_new = g
        
        for alpha in [step_size, step_size * 0.5, step_size * 0.25]:
            p_new = p - alpha * g_tan
            p_new = _project_to_sphere(p_new)
            e_new, g_new = _energy_grad_fast(p_new)
            
            # Add a small penalty for deviating too much from Hilbert structure
            diff_h = p_new[:, np.newaxis, :] - hilbert_pts[np.newaxis, :, :]
            dists_h = np.sqrt(np.sum(diff_h * diff_h, axis=2))
            penalty = 0.001 * np.sum(np.exp(-dists_h[nearest_idx, np.arange(N)]))
            e_adjusted = e_new + penalty
            
            if e_adjusted < best_e_new:
                best_e_new = e_new  # Use actual energy for tracking
                best_p_new = p_new
                best_g_new = g_new
        
        if best_e_new >= e:
            step_size *= 0.5
            if step_size < 1e-8:
                break
            continue
        
        p = best_p_new
        e = best_e_new
        g = best_g_new
        
        if e < best_e:
            best_e = e
            best_p = p.copy()
    
    return best_p, best_e


def construct():
    start_time = time.time()
    
    # Start with multiple initializations and pick the best
    inits = [
        _fibonacci_spiral(),
        _spherical_design_init(42),
        _spherical_design_init(123),
        _hilbert_curve_init(),
    ]
    
    best_points = None
    best_energy = float('inf')
    
    for init_pts in inits:
        e = _energy(init_pts)
        if e < best_energy:
            best_energy = e
            best_points = init_pts.copy()
    
    # Phase 1: Riemannian flow from best initial point
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.15:
        t_budget = min(TIME_LIMIT * 0.15 - elapsed, 15.0)
        pts, e = _riemannian_flow(best_points, max_iter=10000, tol=1e-12, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 2: Hilbert-guided optimization
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.30:
        t_budget = min(TIME_LIMIT * 0.30 - elapsed, 15.0)
        pts, e = _hilbert_guided_optimize(best_points, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 3: L-BFGS refinement
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.40:
        t_budget = min(TIME_LIMIT * 0.40 - elapsed, 10.0)
        pts, e = _lbfgs_optimize(best_points, max_iter=5000, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 4: Multiple random restarts with Riemannian flow
    np.random.seed(12345)
    elapsed = time.time() - start_time
    restart_count = 0
    while elapsed < TIME_LIMIT * 0.65:
        restart_count += 1
        if restart_count % 3 == 0:
            noise = np.random.randn(N, 3) * 0.05
            random_pts = _project_to_sphere(best_points + noise)
        elif restart_count % 3 == 1:
            noise = np.random.randn(N, 3) * 0.15
            random_pts = _project_to_sphere(best_points + noise)
        else:
            random_pts = _random_points(seed=1000 + restart_count)
        
        t_budget = min(TIME_LIMIT * 0.65 - elapsed, 8.0)
        pts, e = _riemannian_flow(random_pts, max_iter=5000, tol=1e-10, time_budget=t_budget)
        
        if e < best_energy:
            best_energy = e
            best_points = pts
            elapsed = time.time() - start_time
            if elapsed < TIME_LIMIT * 0.65:
                t_budget2 = min(TIME_LIMIT * 0.65 - elapsed, 5.0)
                pts2, e2 = _lbfgs_optimize(best_points, max_iter=3000, time_budget=t_budget2)
                if e2 < best_energy:
                    best_energy = e2
                    best_points = pts2
        
        elapsed = time.time() - start_time
    
    # Phase 5: Simulated annealing refinement
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.85:
        t_budget = min(TIME_LIMIT * 0.85 - elapsed, 20.0)
        pts, e = _simulated_annealing_refine(best_points, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 6: Final Riemannian flow polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.93:
        t_budget = min(TIME_LIMIT * 0.93 - elapsed, 5.0)
        pts, e = _riemannian_flow(best_points, max_iter=10000, tol=1e-13, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 7: Final L-BFGS polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.97:
        t_budget = min(TIME_LIMIT * 0.97 - elapsed, 2.0)
        pts, e = _lbfgs_optimize(best_points, max_iter=5000, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d_sq = np.sum((best_points[i] - best_points[j]) ** 2)
            if d_sq < 1e-18:
                rng = np.random.RandomState(i * 1000 + j)
                best_points[i] += rng.randn(3) * 1e-6
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
    
    return best_points
