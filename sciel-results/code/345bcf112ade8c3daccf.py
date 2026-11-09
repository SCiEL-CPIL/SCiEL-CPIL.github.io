"""Solve: 50 equal spheres in a unit cube [-1/2, 1/2]^3."""
import numpy as np
import time
import scipy.optimize
from scipy.spatial.distance import pdist

N = 50
T_LIMIT = 145.0  # wall-clock safety margin (evaluator kills at 150s)

def _min_radius(centers):
    """Compute the exact score: min over all pairs and boundaries."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary clearance
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    # For N=50, direct computation is fast
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, min_boundary)

def _soft_min_radius(centers, beta):
    """
    Smooth surrogate for the minimum radius constraint.
    Returns a scalar to be minimized (the soft min value).
    """
    abs_c = np.abs(centers)
    boundary = 0.5 - np.max(abs_c, axis=1)
    d = pdist(centers, metric='euclidean') / 2.0
    all_vals = np.concatenate([boundary, d])
    
    # Soft min: - (1/beta) * log(sum(exp(-beta * vals)))
    # To avoid overflow/underflow, use max trick
    x_neg = -beta * all_vals
    x_max = np.max(x_neg)
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_neg - x_max)))
    f = (1.0 / beta) * log_sum_exp
    return f

def _gradient_soft_min(centers, beta):
    """Gradient of the soft-min objective w.r.t. centers."""
    abs_c = np.abs(centers)
    boundary = 0.5 - np.max(abs_c, axis=1)
    d = pdist(centers, metric='euclidean') / 2.0
    all_vals = np.concatenate([boundary, d])
    
    x_neg = -beta * all_vals
    x_max = np.max(x_neg)
    exp_vals = np.exp(x_neg - x_max)
    sum_exp = np.sum(exp_vals)
    grad_all = -exp_vals / sum_exp
    
    grad_centers = np.zeros_like(centers)
    
    # Boundary gradient
    for i in range(N):
        k = np.argmax(np.abs(centers[i]))
        grad_centers[i, k] += grad_all[i] * (-np.sign(centers[i, k]))
    
    # Pairwise gradient
    pair_idx = 0
    for a in range(N - 1):
        for b in range(a + 1, N):
            w = grad_all[N + pair_idx]
            diff = centers[a] - centers[b]
            dist = np.linalg.norm(diff)
            if dist > 1e-15:
                unit = diff / dist
                grad_centers[a] += w * unit * 0.5
                grad_centers[b] += w * (-unit) * 0.5
            pair_idx += 1
    
    return grad_centers

def construct():
    t0 = time.time()
    
    # Define bounds for DE
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Phase 1: Differential Evolution for global search
    # Use Sobol initialization for good coverage
    def de_objective(x):
        c = x.reshape(-1, 3)
        # Use a moderate beta for global search to keep landscape smooth
        return _soft_min_radius(c, 50.0)

    try:
        # popsize=15 means 15 * (3*N) = 45 * 50 = 2250 variables? No.
        # popsize is a factor. popsize * len(bounds) is the population size?
        # Actually, scipy DE popsize is a factor of the number of variables.
        # len(bounds) = 150. popsize=15 -> 2250 individuals. That's too many.
        # Let's use popsize=10 -> 1500 individuals. Still large.
        # For N=50, let's try a smaller population to fit in time.
        # Let's monitor time.
        
        result_de = scipy.optimize.differential_evolution(
            de_objective, bounds,
            init='sobol',
            popsize=10,
            maxiter=100,
            tol=1e-6,
            update_interval=1,
            polish=False, # We will polish manually
            seed=42,
            workers=1
        )
        
        best_centers = result_de.x.reshape(-1, 3)
        
    except Exception:
        # Fallback to Fibonacci if DE fails or times out
        golden = (1 + np.sqrt(5)) / 2.0
        inv_golden = 1.0 / golden
        best_centers = np.zeros((N, 3))
        for i in range(N):
            best_centers[i, 0] = (i * inv_golden) % 1.0
            best_centers[i, 1] = (i * inv_golden ** 2) % 1.0
            best_centers[i, 2] = (i * inv_golden ** 3) % 1.0
        best_centers = best_centers * 0.96 - 0.48

    # Check time
    if time.time() - t0 > T_LIMIT - 20.0:
        return best_centers

    # Phase 2: Local Refinement with L-BFGS-B
    # Use a higher beta for sharper minimum
    def lbfgs_obj(x):
        c = x.reshape(-1, 3)
        return _soft_min_radius(c, 500.0)
        
    def lbfgs_grad(x):
        c = x.reshape(-1, 3)
        g = _gradient_soft_min(c, 500.0)
        return g.flatten()

    try:
        result_lbfgs = scipy.optimize.minimize(
            lbfgs_obj, best_centers.flatten(), jac=lbfgs_grad, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 20}
        )
        if result_lbfgs.fun < lbfgs_obj(best_centers.flatten()):
            best_centers = result_lbfgs.x.reshape(-1, 3)
    except Exception:
        pass

    # Check time
    if time.time() - t0 > T_LIMIT - 10.0:
        return best_centers

    # Phase 3: Final Polish with Nelder-Mead on Hard Objective
    def nm_obj(x):
        c = x.reshape(-1, 3)
        return -_min_radius(c)

    try:
        result_nm = scipy.optimize.minimize(
            nm_obj, best_centers.flatten(), method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        new_centers = result_nm.x.reshape(-1, 3)
        if _min_radius(new_centers) > _min_radius(best_centers):
            best_centers = new_centers
    except Exception:
        pass

    return best_centers
