import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _log_energy(points):
    """Compute log(1/d) energy using math.fsum for accuracy."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -math.fsum(log_dists.tolist())

def _fast_log_energy(points):
    """Compute log(1/d) energy using np.sum for speed."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -np.sum(log_dists)

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)
    
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # E = -sum_{i<j} log(d_ij) where d_ij = sqrt(dists_sq_ij)
    # E = -0.5 * sum_{i<j} log(dists_sq_ij)
    # dE/dx_i = -0.5 * sum_{j!=i} (1/dists_sq_ij) * 2*(x_i - x_j) = -sum_{j!=i} (x_i - x_j)/dists_sq_ij
    #         = sum_{j!=i} (x_j - x_i)/dists_sq_ij
    # Wait: d(log(d_ij))/dx_i = d(0.5*log(dists_sq_ij))/dx_i = 0.5 * (1/dists_sq_ij) * 2*(x_i-x_j)
    # = (x_i - x_j)/dists_sq_ij
    # So dE/dx_i = -sum_{j!=i} (x_i - x_j)/dists_sq_ij = sum_{j!=i} (x_j - x_i)/dists_sq_ij
    # This is what we have. But let me double-check with a simple 2-point case.
    # E = -log(d_12) = -0.5*log(|x1-x2|^2)
    # dE/dx1 = -0.5 * 2*(x1-x2)/|x1-x2|^2 = -(x1-x2)/|x1-x2|^2 = (x2-x1)/|x1-x2|^2
    # So dE/dx_i = sum_{j!=i} (x_j - x_i)/dists_sq_ij. That's correct.
    
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    # Compute energy with math.fsum for consistency with final score
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    energy = -math.fsum(log_dists.tolist())
    
    return energy, grads_tangent.flatten()

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _lbfgs_polish(points, max_iter=2000, ftol=1e-12, gtol=1e-10, time_limit=10.0):
    """Polish a point configuration using L-BFGS-B."""
    start = time.time()
    x0 = points.flatten()
    e0 = _log_energy(points)
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 10,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50
            }
        )
        if res.success or res.fun < e0:
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < e0 and time.time() - start < time_limit:
                return p_res
    except Exception:
        pass
    return points

def _bb_gradient_descent(points, max_iter=5000, time_limit=15.0, tol=1e-14):
    """Barzilai-Borwein step-size gradient descent on the sphere."""
    start = time.time()
    current = points.copy()
    e_current = _fast_log_energy(current)
    
    # Compute initial gradient
    _, g_current = _energy_and_grad(current.flatten())
    g_current = g_current.reshape(N, 3)
    
    prev_grad = g_current.copy()
    prev_point = current.copy()
    
    # Initial step size
    step = 0.01
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Project gradient to tangent space
        dot_g = np.sum(g_current * current, axis=1, keepdims=True)
        g_tangent = g_current - dot_g * current
        
        # Check convergence
        g_norm = np.linalg.norm(g_tangent)
        if g_norm < tol:
            break
        
        # BB step size
        if it > 0:
            s = current - prev_point
            y = g_current - prev_grad
            s_norm_sq = np.sum(s * s)
            y_norm_sq = np.sum(y * y)
            s_dot_y = np.sum(s * y)
            
            if s_dot_y > 1e-10:
                # Alternate between BB1 and BB2
                if it % 2 == 0:
                    step = s_norm_sq / s_dot_y
                else:
                    step = s_dot_y / y_norm_sq
            # Clamp step size
            step = np.clip(step, 1e-8, 1.0)
        
        # Take step
        new_point = current - step * g_tangent
        new_point = _project_to_sphere(new_point)
        
        e_new = _fast_log_energy(new_point)
        
        if e_new < e_current:
            # Update BB quantities
            prev_grad = g_current.copy()
            prev_point = current.copy()
            current = new_point
            e_current = e_new
            
            # Compute new gradient
            _, g_current = _energy_and_grad(current.flatten())
            g_current = g_current.reshape(N, 3)
        else:
            # Halve step size (backtracking)
            step *= 0.5
            new_point = current - step * g_tangent
            new_point = _project_to_sphere(new_point)
            e_new = _fast_log_energy(new_point)
            if e_new < e_current:
                prev_grad = g_current.copy()
                prev_point = current.copy()
                current = new_point
                e_current = e_new
                _, g_current = _energy_and_grad(current.flatten())
                g_current = g_current.reshape(N, 3)
            else:
                # Skip this iteration
                step *= 0.5
    
    return current

def _simulated_annealing(points, max_iter=500, time_limit=60.0, rng=None):
    """Simulated Annealing to escape local minima."""
    if rng is None:
        rng = np.random.RandomState(12345)
    
    start = time.time()
    best_pts = points.copy()
    best_e = _fast_log_energy(best_pts)
    current_pts = best_pts.copy()
    current_e = best_e
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.001
    cooling = (T_end / T_start) ** (1.0 / max_iter)
    
    # Adaptive step size
    base_step = 0.3
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        T = T_start * (cooling ** it)
        
        # Scale step with temperature
        step_size = base_step * (0.3 + 0.7 * T / T_start)
        
        # Random perturbation: move one or two random points
        n_move = 1 if rng.rand() < 0.7 else 2
        indices = rng.choice(N, size=n_move, replace=False)
        
        new_pts = current_pts.copy()
        for idx in indices:
            noise = rng.randn(3)
            noise = noise / np.linalg.norm(noise)
            new_pts[idx] = current_pts[idx] + step_size * noise
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        e_new = _fast_log_energy(new_pts)
        delta_e = e_new - current_e
        
        # Metropolis acceptance
        if delta_e < 0 or (T > 1e-10 and rng.rand() < math.exp(-min(delta_e / T, 500))):
            current_pts = new_pts
            current_e = e_new
            
            if current_e < best_e:
                best_e = current_e
                best_pts = current_pts.copy()
                
            # Periodic polishing to stay in the basin
            if it % 8 == 0 and time.time() - start < time_limit - 8.0:
                polished = _lbfgs_polish(current_pts, max_iter=80, ftol=1e-9, gtol=1e-7, time_limit=3.0)
                e_polished = _fast_log_energy(polished)
                if e_polished < best_e:
                    best_e = e_polished
                    best_pts = polished.copy()
                current_pts = polished
                current_e = e_polished

    return best_pts, best_e

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initializations
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=42),
    ]
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # First pass: moderate optimization on all initializations
    for idx, init in enumerate(inits):
        if time.time() > deadline - 80.0:
            break
        
        polished = _lbfgs_polish(init, max_iter=1000, ftol=1e-10, gtol=1e-8, time_limit=10.0)
        e_polish = _log_energy(polished)
        
        if e_polish < best_energy:
            best_energy = e_polish
            best_points = polished.copy()
    
    # Second pass: Simulated Annealing
    if time.time() < deadline - 50.0:
        sa_result, sa_energy = _simulated_annealing(best_points, max_iter=400, time_limit=deadline - time.time() - 40.0)
        # Verify with precise energy
        e_sa_precise = _log_energy(sa_result)
        if e_sa_precise < best_energy:
            best_energy = e_sa_precise
            best_points = sa_result.copy()
    
    # Third pass: BB gradient descent for fast additional polish
    if time.time() < deadline - 15.0:
        bb_result = _bb_gradient_descent(best_points, max_iter=3000, time_limit=deadline - time.time() - 10.0)
        e_bb = _log_energy(bb_result)
        if e_bb < best_energy:
            best_energy = e_bb
            best_points = bb_result.copy()
    
    # Fourth pass: final high-precision L-BFGS polish
    if time.time() < deadline - 8.0:
        final_polished = _lbfgs_polish(best_points, max_iter=5000, ftol=1e-14, gtol=1e-12, time_limit=deadline - time.time() - 5.0)
        e_final = _log_energy(final_polished)
        if e_final < best_energy:
            best_energy = e_final
            best_points = final_polished.copy()
    
    # Fifth pass: one more BB pass on the final result
    if time.time() < deadline - 3.0:
        bb_final = _bb_gradient_descent(best_points, max_iter=2000, time_limit=deadline - time.time() - 2.0)
        e_bb_final = _log_energy(bb_final)
        if e_bb_final < best_energy:
            best_points = bb_final.copy()
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    if np.min(dists_sq[iu]) < 1e-18:
        return _fibonacci_sphere(N)
    
    return best_points
