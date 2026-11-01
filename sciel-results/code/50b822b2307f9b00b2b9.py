"""Improved sphere packing via annealed smooth surrogate + Nelder-Mead polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    # Boundary constraint: distance to cube wall
    dist_to_wall = np.min(0.5 - np.abs(centers), axis=1)
    r_wall = np.min(dist_to_wall)
    
    # Pairwise distances
    n = len(centers)
    if n < 2:
        return max(0, r_wall)
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    
    return min(r_wall, r_pair)

def _smooth_objective(centers, alpha):
    """
    Smooth surrogate: maximize a soft-min of all constraints.
    We minimize the negative of the soft-min radius.
    soft_min(x, alpha) = -(1/alpha) * log(sum(exp(-alpha*x)))
    """
    # Boundary distances
    dist_to_wall = 0.5 - np.abs(centers)
    min_wall_per_sphere = np.min(dist_to_wall, axis=1)
    
    # Pairwise half-distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    half_dists = dists / 2.0
    # Set diagonal to a large value so it doesn't dominate
    np.fill_diagonal(half_dists, 1e6)
    
    # Get all constraint values
    wall_vals = min_wall_per_sphere
    pair_vals = half_dists[~np.eye(len(centers), dtype=bool)]
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Soft-min
    # For numerical stability
    a = -alpha
    shifted = a * all_vals
    max_shift = np.max(shifted)
    log_sum_exp = max_shift + np.log(np.sum(np.exp(shifted - max_shift)))
    soft_min = -log_sum_exp / alpha
    
    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min

def _gradient_smooth(centers, alpha):
    """Gradient of the smooth objective (for L-BFGS-B)."""
    eps = 1e-7
    n = len(centers)
    grad = np.zeros_like(centers)
    for i in range(n):
        for d in range(3):
            c_plus = centers.copy()
            c_plus[i, d] += eps
            c_minus = centers.copy()
            c_minus[i, d] -= eps
            f_plus = _smooth_objective(c_plus, alpha)
            f_minus = _smooth_objective(c_minus, alpha)
            grad[i, d] = (f_plus - f_minus) / (2 * eps)
    return grad

def construct():
    start_time = time.time()
    time_limit = 100.0  # Safety margin (150s limit, leave buffer)
    
    best_centers = None
    best_radius = -np.inf
    
    def _evaluate(centers):
        nonlocal best_centers, best_radius
        if not np.all(np.isfinite(centers)):
            return
        if np.any(np.abs(centers) > 0.5):
            return
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    # Initialize with a reasonable starting point
    # Use a 4x4x4 grid (64 points), take first 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init = grid[:N].copy()
    
    _evaluate(init)
    
    # Phase 1: Annealed smooth optimization
    alphas = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    current_centers = init.copy()
    
    for alpha in alphas:
        if time.time() - start_time > time_limit - 30:
            break
        
        # Try multiple starting points for diversity
        for init_seed in range(3):
            if time.time() - start_time > time_limit - 30:
                break
            
            if init_seed == 0:
                c0 = current_centers.copy()
            elif init_seed == 1:
                # FCC-like lattice perturbation
                rng = np.random.RandomState(42 + init_seed)
                c0 = rng.uniform(-0.4, 0.4, size=(N, 3))
            else:
                # Grid perturbation
                rng = np.random.RandomState(100 + init_seed)
                c0 = init + rng.normal(0, 0.05, size=(N, 3))
                c0 = np.clip(c0, -0.45, 0.45)
            
            # L-BFGS-B with smooth objective
            try:
                res = minimize(
                    _smooth_objective,
                    c0,
                    args=(alpha,),
                    method='L-BFGS-B',
                    jac=None,  # Let scipy handle numerical gradient
                    bounds=[(-0.499, 0.499)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.x is not None and np.all(np.isfinite(res.x)):
                    _evaluate(res.x)
                    # Keep the best result as next starting point
                    if best_centers is not None and _compute_radius(res.x) >= _compute_radius(best_centers):
                        current_centers = res.x.copy()
            except Exception:
                pass
    
    # Phase 2: Nelder-Mead polish on the exact (non-smooth) objective
    if time.time() - start_time < time_limit - 10 and best_centers is not None:
        try:
            def neg_radius(c):
                c = c.reshape(N, 3)
                r = _compute_radius(c)
                return -r
            
            res_nm = minimize(
                neg_radius,
                best_centers.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res_nm.x is not None and np.all(np.isfinite(res_nm.x)):
                _evaluate(res_nm.x.reshape(N, 3))
        except Exception:
            pass
    
    # Phase 3: Additional refinement with higher alpha if time permits
    if time.time() - start_time < time_limit - 5 and best_centers is not None:
        for alpha in [256, 512]:
            if time.time() - start_time > time_limit - 5:
                break
            try:
                res = minimize(
                    _smooth_objective,
                    best_centers,
                    args=(alpha,),
                    method='L-BFGS-B',
                    bounds=[(-0.499, 0.499)] * (N * 3),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None and np.all(np.isfinite(res.x)):
                    _evaluate(res.x)
            except Exception:
                pass
        
        # Final Nelder-Mead polish
        if time.time() - start_time < time_limit - 2 and best_centers is not None:
            try:
                def neg_radius(c):
                    c = c.reshape(N, 3)
                    r = _compute_radius(c)
                    return -r
                
                res_nm = minimize(
                    neg_radius,
                    best_centers.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                if res_nm.x is not None and np.all(np.isfinite(res_nm.x)):
                    _evaluate(res_nm.x.reshape(N, 3))
            except Exception:
                pass
    
    # Ensure validity: clip to cube and check
    if best_centers is None:
        # Fallback to grid
        return init
    
    # Final clip to ensure inside cube
    best_centers = np.clip(best_centers, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_centers
