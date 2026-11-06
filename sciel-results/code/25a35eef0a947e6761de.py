"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(x):
    """Compute the maximum common radius for given centres."""
    x = x.reshape(-1, 3)
    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)
    min_dist = np.min(d)
    boundary_dist = 0.5 - np.max(np.abs(x), axis=1)
    min_boundary = np.min(boundary_dist)
    return min(min_dist, min_boundary) / 2.0

def _force_based_objective_and_grad(x, beta=50.0):
    """
    Force-based objective: minimize a smooth approximation of -min_distance.
    Uses exponential repulsion for pairwise distances and boundary penalties.
    """
    x = x.reshape(-1, 3)
    n = x.shape[0]
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    
    # Boundary distances
    boundary = 0.5 - np.abs(x)
    boundary_min = np.min(boundary, axis=1)
    
    # Use a smooth min approximation: -log(sum(exp(-beta*d))) / beta
    # For pairs
    iu = np.triu_indices(n, k=1)
    pair_dists = d[iu]
    
    # For boundary
    # Combine all distances
    all_dists = np.concatenate([pair_dists, boundary_min])
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    neg_beta_d = -beta * all_dists
    neg_beta_d -= np.max(neg_beta_d)  # for numerical stability
    log_sum = np.log(np.sum(np.exp(neg_beta_d)))
    obj = log_sum / beta  # This is the negative of the soft min
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Pairwise gradient
    pair_i, pair_j = iu
    exp_vals = np.exp(neg_beta_d[:len(pair_dists)])
    total_exp = np.sum(np.exp(neg_beta_d))
    weights = exp_vals / total_exp
    
    for k in range(len(pair_dists)):
        i, j = pair_i[k], pair_j[k]
        w = weights[k]
        if d[i, j] < 1e-12:
            continue
        diff_ij = x[i] - x[j]
        grad[i] += w * diff_ij / d[i, j]
        grad[j] += -w * diff_ij / d[i, j]
    
    # Boundary gradient
    exp_vals_b = np.exp(neg_beta_d[len(pair_dists):])
    weights_b = exp_vals_b / total_exp
    
    for k in range(n):
        w = weights_b[k]
        # Find which axis is closest to boundary
        b = boundary[k]
        min_axis = np.argmin(b)
        sign_val = np.sign(x[k, min_axis])
        if sign_val == 0:
            sign_val = 1.0
        grad[k, min_axis] += -w * sign_val
    
    return obj, grad

def _anneal_optimize(x0, alphas, deadline, maxiter_per_alpha=200):
    """
    Run annealed optimization with increasing beta (sharper min approximation).
    """
    current_x = x0.copy()
    best_x = x0.copy()
    best_r = _compute_radius(x0)
    
    for beta in alphas:
        if time.time() > deadline - 5.0:
            break
        
        try:
            result = minimize(
                lambda x: _force_based_objective_and_grad(x, beta)[0],
                current_x,
                jac=lambda x: _force_based_objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * N * 3,
                options={'maxiter': maxiter_per_alpha, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e10:
                candidate = result.x.reshape(-1, 3)
                r_candidate = _compute_radius(candidate)
                if r_candidate > best_r:
                    best_r = r_candidate
                    best_x = candidate.copy()
                current_x = candidate.copy()
        except Exception:
            pass
    
    return best_x, best_r

def _local_search(x0, deadline, n_restarts=5):
    """
    Perform local search with random perturbations to escape local minima.
    """
    best_x = x0.copy()
    best_r = _compute_radius(x0)
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the current best solution
        noise_scale = 0.02 * (1.0 - restart / max(1, n_restarts))
        perturbed = best_x + rng.normal(0, noise_scale, size=best_x.shape)
        perturbed = np.clip(perturbed, -0.45, 0.45)
        
        # Try to optimize from this perturbed state with a few beta values
        alphas = [50, 100, 200, 500]
        try:
            result = minimize(
                lambda x: _force_based_objective_and_grad(x, alphas[-1])[0],
                perturbed,
                jac=lambda x: _force_based_objective_and_grad(x, alphas[-1])[1],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * N * 3,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e10:
                candidate = result.x.reshape(-1, 3)
                r_candidate = _compute_radius(candidate)
                if r_candidate > best_r:
                    best_r = r_candidate
                    best_x = candidate.copy()
        except Exception:
            pass
    
    return best_x, best_r

def _initial_grid():
    """Create initial cubic grid packing."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _initial_fcc_like():
    """Create an FCC-inspired initial configuration."""
    # Use a 5x5x4 grid but with some offset to create FCC-like structure
    m1, m2, m3 = 5, 5, 4
    xs = (np.arange(m1) + 0.5) / m1 - 0.5
    ys = (np.arange(m2) + 0.5) / m2 - 0.5
    zs = (np.arange(m3) + 0.5) / m3 - 0.5
    
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                centres.append([x, y, z])
    
    return np.array(centres[:N])

def _initial_random(seed=42):
    """Random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    centres = rng.uniform(-0.45, 0.45, size=(N, 3))
    return centres

def construct():
    """
    Find the optimal packing of 100 equal spheres in a unit cube.
    Uses force-based smooth objective with annealing, multi-start, and local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s hard limit
    
    # Try multiple initializations
    initializations = [
        ('grid', _initial_grid()),
        ('fcc', _initial_fcc_like()),
        ('rand0', _initial_random(0)),
        ('rand1', _initial_random(1)),
        ('rand2', _initial_random(2)),
        ('rand42', _initial_random(42)),
    ]
    
    best_x = None
    best_r = -np.inf
    
    for name, x0 in initializations:
        if time.time() > deadline - 30.0:
            break
        
        # Phase 1: Annealed optimization
        alphas = [10, 20, 50, 100, 200, 500]
        x_opt, r_opt = _anneal_optimize(x0, alphas, deadline, maxiter_per_alpha=150)
        
        # Phase 2: Local search
        if time.time() < deadline - 15.0:
            x_opt, r_opt = _local_search(x_opt, deadline, n_restarts=3)
        
        if r_opt > best_r:
            best_r = r_opt
            best_x = x_opt.copy()
    
    # Final polish on the best solution
    if best_x is not None and time.time() < deadline - 5.0:
        try:
            result = minimize(
                lambda x: _force_based_objective_and_grad(x, 1000)[0],
                best_x,
                jac=lambda x: _force_based_objective_and_grad(x, 1000)[1],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * N * 3,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < 1e10:
                candidate = result.x.reshape(-1, 3)
                r_candidate = _compute_radius(candidate)
                if r_candidate > best_r:
                    best_r = r_candidate
                    best_x = candidate.copy()
        except Exception:
            pass
    
    # Ensure validity: clip to cube
    if best_x is not None:
        best_x = np.clip(best_x, -0.5, 0.5)
        
        # Final validation
        final_r = _compute_radius(best_x)
        if not np.isfinite(final_r) or final_r <= 0:
            return _initial_grid()
        
        return best_x
    else:
        return _initial_grid()

# EVOLVE-BLOCK-END
