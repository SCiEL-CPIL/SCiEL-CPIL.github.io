"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy using math.fsum for accuracy."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -math.fsum(log_dists.tolist())


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
    
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    energy = -math.fsum(log_dists.tolist())
    
    return energy, grads_tangent.flatten()


def _energy_only(x):
    """Energy only for line search."""
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)
    return _log_energy(points)


def _exp_map_tangent(pts, tangent):
    """
    Exponential map on the sphere: move pts along tangent vectors.
    For a unit vector x and tangent v, the exponential map is
        exp_x(v) = cos(|v|) x + sin(|v|) (v/|v|)
    """
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    safe_norm = np.where(norm > 1e-15, norm, 1.0)
    u = tangent / safe_norm
    c = np.cos(norm)
    s = np.sin(norm)
    out = c * pts + s * u
    return _project_to_sphere(out)


def _riemannian_lbfgs_polish(pts, deadline, maxiter=400, ftol=1e-16, gtol=1e-13):
    """
    Polish a configuration using L-BFGS-B on tangent-space coordinates.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _log_energy(pts)

    def fun(t):
        tv = t.reshape(N, 3)
        dot = np.sum(tv * pts, axis=1, keepdims=True)
        tv = tv - dot * pts
        new_pts = _exp_map_tangent(pts, tv)
        e, g = _energy_and_grad(new_pts.flatten())
        return e, g

    t0 = np.zeros(3 * N)
    try:
        res = minimize(fun, t0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol,
                                'gtol': gtol, 'maxls': 30})
        if time.time() < deadline:
            tv = res.x.reshape(N, 3)
            dot = np.sum(tv * pts, axis=1, keepdims=True)
            tv = tv - dot * pts
            new_pts = _exp_map_tangent(pts, tv)
            new_energy = _log_energy(new_pts)
            if new_energy < best_energy and np.all(np.isfinite(new_pts)):
                best_energy = new_energy
                best_pts = new_pts.copy()
    except Exception:
        pass

    return best_pts, best_energy


def _lbfgs_optimize(points, max_iter=3000, ftol=1e-12, gtol=1e-10, time_limit=20.0):
    """L-BFGS-B optimization with time limit."""
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


def _adam_optimize(points, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                   max_iter=5000, time_limit=15.0):
    """Adam optimizer on the sphere with tangent projection."""
    start = time.time()
    pts = points.copy()
    best_e = _log_energy(pts)
    best_pts = pts.copy()
    
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    
    for t in range(1, max_iter + 1):
        if time.time() - start > time_limit:
            break
        
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]
        
        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ pts
        grads = weighted_sum - pts * sum_weights[:, np.newaxis]
        
        dot_products = np.sum(grads * pts, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * pts
        
        m = beta1 * m + (1 - beta1) * grads_tangent
        v = beta2 * v + (1 - beta2) * grads_tangent ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        update = m_hat / (np.sqrt(v_hat) + eps)
        new_pts = pts - lr * update
        new_pts = _project_to_sphere(new_pts)
        
        e_new = _log_energy(new_pts)
        if e_new < best_e:
            best_e = e_new
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            lr *= 0.95
    
    return best_pts


def _gradient_descent_sphere(points, lr=0.01, max_iter=10000, time_limit=15.0):
    """Gradient descent on the sphere with adaptive step size."""
    start = time.time()
    pts = points.copy()
    best_e = _log_energy(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]
        
        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ pts
        grads = weighted_sum - pts * sum_weights[:, np.newaxis]
        
        dot_products = np.sum(grads * pts, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * pts
        
        grad_norm = np.linalg.norm(grads_tangent)
        if grad_norm < 1e-12:
            break
        
        step = lr / (1.0 + grad_norm * 0.1)
        new_pts = pts - step * grads_tangent
        new_pts = _project_to_sphere(new_pts)
        
        e_new = _log_energy(new_pts)
        if e_new < best_e:
            best_e = e_new
            best_pts = new_pts.copy()
            pts = new_pts
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.9
    
    return best_pts


def _tangent_vector(point, rng):
    """Generate a random unit tangent vector at point on sphere."""
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm


def _basin_hopping_single(init_points, deadline, rng, max_iter=50, T_init=0.1):
    """
    Basin hopping:
    1. Local minimize
    2. Perturb
    3. Local minimize
    4. Accept if better (Metropolis)
    """
    current_points, current_energy = _local_minimize(init_points, deadline, maxiter=200)
    best_points = current_points.copy()
    best_energy = current_energy

    T = T_init
    T_min = 1e-8
    cooling = 0.95

    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        if T < T_min:
            break

        n_perturb = rng.randint(2, 8)
        idxs = rng.choice(N, n_perturb, replace=False)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = T
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=100)

        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()

        T *= cooling

    return best_points, best_energy


def _local_minimize(points, deadline, maxiter=500, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B to find local minimum."""
    x0 = points.flatten()
    
    if time.time() > deadline - 1.0:
        return points, _log_energy(points)

    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': ftol, 'gtol': gtol, 'maxls': 40}
        )
        
        if time.time() > deadline - 1.0:
            return points, _log_energy(points)
            
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)


def _multi_start_optimize(points, time_limit=25.0):
    """Try multiple perturbation strategies and pick the best."""
    start = time.time()
    best_pts = points.copy()
    best_e = _log_energy(best_pts)
    
    for scale in [0.05, 0.1, 0.2]:
        if time.time() - start > time_limit - 10.0:
            break
        
        rng = np.random.RandomState(int(time.time() * 1000) % 100000)
        perturbed = points + rng.randn(N, 3) * scale
        perturbed = _project_to_sphere(perturbed)
        
        gd_result = _gradient_descent_sphere(perturbed, lr=0.01, max_iter=2000,
                                              time_limit=time_limit - (time.time() - start) - 5.0)
        e_gd = _log_energy(gd_result)
        
        if e_gd < best_e:
            best_e = e_gd
            best_pts = gd_result.copy()
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Multiple initializations
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=7),
        _random_uniform_sphere(N, seed=42),
        _random_uniform_sphere(N, seed=99),
    ]
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # First pass: L-BFGS-B on all initializations
    for idx, init in enumerate(inits):
        if time.time() > deadline - 50.0:
            break
        
        result = _lbfgs_optimize(init, max_iter=2000, ftol=1e-12, gtol=1e-10,
                                  time_limit=min(10.0, deadline - time.time() - 40.0))
        e_check = _log_energy(result)
        if e_check < best_energy:
            best_energy = e_check
            best_points = result.copy()
    
    # Second pass: Basin hopping (new from inspiration)
    if time.time() < deadline - 40.0:
        rng = np.random.RandomState(54321)
        bh_result, e_bh = _basin_hopping_single(
            best_points, deadline=deadline - 10.0, rng=rng, max_iter=30, T_init=0.1)
        if e_bh < best_energy:
            best_energy = e_bh
            best_points = bh_result.copy()
    
    # Third pass: Adam fine-tuning
    if time.time() < deadline - 20.0:
        adam_result = _adam_optimize(best_points, lr=0.001, max_iter=5000,
                                      time_limit=deadline - time.time() - 15.0)
        e_adam = _log_energy(adam_result)
        if e_adam < best_energy:
            best_energy = e_adam
            best_points = adam_result.copy()
    
    # Fourth pass: Riemannian L-BFGS polish
    if time.time() < deadline - 15.0:
        riemannian_polished, e_riem = _riemannian_lbfgs_polish(
            best_points, deadline=deadline - 5.0, maxiter=500, ftol=1e-16, gtol=1e-13)
        if e_riem < best_energy:
            best_energy = e_riem
            best_points = riemannian_polished.copy()
    
    # Fifth pass: aggressive L-BFGS-B polishing
    if time.time() < deadline - 10.0:
        polished = _lbfgs_optimize(best_points, max_iter=5000, ftol=1e-14, gtol=1e-12,
                                    time_limit=deadline - time.time() - 5.0)
        e_polish = _log_energy(polished)
        if e_polish < best_energy:
            best_energy = e_polish
            best_points = polished.copy()
    
    # Sixth pass: multi-start if time remains
    if time.time() < deadline - 8.0:
        multi_result = _multi_start_optimize(best_points,
                                              time_limit=deadline - time.time() - 5.0)
        e_multi = _log_energy(multi_result)
        if e_multi < best_energy:
            best_energy = e_multi
            best_points = multi_result.copy()
    
    # Final ultra-tight polish with Riemannian method
    if time.time() < deadline - 4.0:
        final_polished, e_final = _riemannian_lbfgs_polish(
            best_points, deadline=deadline - 2.0, maxiter=1000, ftol=1e-16, gtol=1e-14)
        if e_final < best_energy:
            best_energy = e_final
            best_points = final_polished.copy()
    
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
# EVOLVE-BLOCK-END
