"""Optimized sphere packing: fast lower-triangle energy + smooth radius maximization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# Precompute lower-triangle indices for pairwise distances (i < j)
_i_idx, _j_idx = np.triu_indices(N, k=1)
_NPAIRS = len(_i_idx)

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    diff = centers[_i_idx] - centers[_j_idx]
    sq_dists = np.sum(diff**2, axis=1)
    min_pair_dist = np.sqrt(np.min(sq_dists))
    min_wall_dist = np.min(0.5 - np.abs(centers))
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _pairwise_energy_and_grad(centers, r_target):
    """
    Fast pairwise + wall energy and gradient using lower-triangle indexing.
    E = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - w_i)^2
    """
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise: only i < j
    diff = centers[_i_idx] - centers[_j_idx]
    dists = np.sqrt(np.sum(diff**2, axis=1) + 1e-15)
    
    d_min = 2.0 * r_target
    violations = np.maximum(0.0, d_min - dists)
    mask = violations > 0
    
    if np.any(mask):
        # For each violating pair (i,j):
        # E += (2r - d)^2
        # dE/dd = -2(2r - d)
        # dd/dc_i = (c_i - c_j)/d, dd/dc_j = (c_j - c_i)/d
        # So dE/dc_i = -2(2r-d) * (c_i-c_j)/d, dE/dc_j = 2(2r-d) * (c_i-c_j)/d
        coeff = np.where(mask, 2.0 * violations / dists, 0.0)
        # coeff has shape (_NPAIRS,)
        # grad[_i_idx] -= coeff[:, None] * diff
        # grad[_j_idx] += coeff[:, None] * diff
        np.add.at(grad, _i_idx, -coeff[:, np.newaxis] * diff)
        np.add.at(grad, _j_idx, coeff[:, np.newaxis] * diff)
        energy += np.sum(violations**2)
    
    # Walls
    wall_dists = 0.5 - np.abs(centers)
    wall_viol = np.maximum(0.0, r_target - wall_dists)
    wall_mask = wall_viol > 0
    
    if np.any(wall_mask):
        signs = np.sign(centers)
        signs[signs == 0] = 1.0
        grad += 2.0 * wall_viol * signs
        energy += np.sum(wall_viol**2)
    
    return float(energy), grad

def _smooth_radius_obj_and_grad(centers, beta=200.0):
    """
    Smooth surrogate for maximizing the bottleneck radius.
    Uses log-sum-exp to approximate -min(constraints) smoothly.
    Objective: maximize min( min_{i<j} d_ij/2, min_i w_i )
    Surrogate: minimize -r_smooth where
      r_smooth = -(1/beta) * log( sum_k exp(-beta * constraint_k) )
    This is a smooth approximation to min_k constraint_k.
    We want to maximize it, so we minimize negative of it.
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Constraint values: pair distances / 2, and wall distances
    diff = centers[_i_idx] - centers[_j_idx]
    dists = np.sqrt(np.sum(diff**2, axis=1) + 1e-15)
    pair_constraints = dists / 2.0  # _NPAIRS values
    
    wall_dists = 0.5 - np.abs(centers)  # n*3 values
    
    # Combine all constraints
    all_constraints = np.concatenate([pair_constraints, wall_dists.flatten()])
    n_constraints = len(all_constraints)
    
    # Log-sum-exp for numerical stability
    min_c = np.min(all_constraints)
    shifted = all_constraints - min_c
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted))) + (-beta * min_c)
    # r_smooth = -(1/beta) * log_sum_exp
    r_smooth = -(1.0 / beta) * log_sum_exp
    
    # Gradient: d r_smooth / d constraint_k = exp(-beta*(c_k - min_c)) / sum_j exp(-beta*(c_j - min_c))
    # = exp(-beta * shifted_k) / sum(exp(-beta * shifted))
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights)
    
    # Gradient of r_smooth w.r.t. centers
    # For pair constraints: d(pair_constraint_k)/d(c_i) = diff_k / (2 * dist_k)
    # For wall constraints: d(wall)/d(c_ik) = -sign(c_ik)
    
    # Pair part
    pair_weights = weights[:_NPAIRS]
    # d(pair_k)/d(c_i) = (c_i - c_j) / (2 * d_k) = diff_k / (2 * dist_k)
    pair_grad_factor = pair_weights / (2.0 * dists)
    np.add.at(grad, _i_idx, pair_grad_factor[:, np.newaxis] * diff)
    np.add.at(grad, _j_idx, -pair_grad_factor[:, np.newaxis] * diff)
    
    # Wall part
    wall_weights = weights[_NPAIRS:]
    wall_grad = -np.sign(centers) * wall_weights.reshape(n, 3)
    grad += wall_grad
    
    # We want to maximize r_smooth, so minimize -r_smooth
    # grad of objective = -grad of r_smooth
    return float(-r_smooth), -grad

def _sgld_optimize(centers, r_target, time_limit=20.0, seed=42):
    """Run SGLD with cooling schedule."""
    start = time.time()
    rng = np.random.RandomState(seed)
    c = centers.copy()
    
    n_steps = int(time_limit * 400)  # faster steps now
    T_init = 0.015
    T_final = 0.0003
    lr_init = 0.008
    lr_final = 0.0005
    
    best_energy = np.inf
    best_c = c.copy()
    current_energy, _ = _pairwise_energy_and_grad(c, r_target)
    best_energy = current_energy
    
    for step in range(n_steps):
        if time.time() - start > time_limit - 0.5:
            break
        frac = step / max(1, n_steps)
        T = T_init * (T_final / T_init) ** frac
        lr = lr_init * (lr_final / lr_init) ** frac
        
        e, g = _pairwise_energy_and_grad(c, r_target)
        noise_scale = np.sqrt(2.0 * lr * T)
        noise = rng.normal(0, noise_scale, size=c.shape)
        c = c - lr * g + noise
        c = np.clip(c, -0.49999, 0.49999)
        
        if e < best_energy:
            best_energy = e
            best_c = c.copy()
    
    return best_c, best_energy

def _polish_lbfgs(centers, r_target, time_limit=10.0):
    """Polish with L-BFGS-B using fast energy."""
    start = time.time()
    
    def obj_and_grad(x):
        if time.time() - start > time_limit:
            return 1e10, np.zeros_like(x)
        c = x.reshape(N, 3)
        e, g = _pairwise_energy_and_grad(c, r_target)
        return e, g.flatten()
    
    x0 = centers.flatten()
    bounds = [(-0.49999, 0.49999)] * (N * 3)
    
    try:
        result = minimize(
            obj_and_grad, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return centers

def _polish_radius_max(centers, time_limit=15.0, beta=200.0):
    """Polish by directly maximizing smooth radius surrogate."""
    start = time.time()
    
    def obj_and_grad(x):
        if time.time() - start > time_limit:
            return 1e10, np.zeros_like(x)
        c = x.reshape(N, 3)
        c = np.clip(c, -0.49999, 0.49999)
        e, g = _smooth_radius_obj_and_grad(c, beta=beta)
        return e, g.flatten()
    
    x0 = centers.flatten()
    bounds = [(-0.49999, 0.49999)] * (N * 3)
    
    try:
        result = minimize(
            obj_and_grad, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return centers

def _grid_init(n):
    """Grid initialization."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n, seed=None):
    """Random initialization inside cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def construct():
    """Construct optimal packing of N spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_centers = _grid_init(N)
    best_radius = _compute_radius(best_centers)
    
    # Progressive target radii - tighter sequence
    targets = [0.10, 0.105, 0.108, 0.110, 0.111, 0.1113, 0.11138, 0.11140, 0.11142, 0.11145]
    
    for t_idx, target_r in enumerate(targets):
        if time.time() > deadline - 30.0:
            break
        if target_r <= best_radius:
            continue
        
        # Multiple restarts
        n_restarts = 2 if t_idx < 4 else 1
        for restart in range(n_restarts):
            if time.time() > deadline - 25.0:
                break
            
            if restart == 0:
                rng_init = np.random.RandomState(100 + t_idx)
                init_c = best_centers + rng_init.normal(0, 0.002, size=(N, 3))
                init_c = np.clip(init_c, -0.4999, 0.4999)
            else:
                init_c = _random_init(N, seed=200 + t_idx)
            
            # SGLD phase
            sgld_time = min(12.0, deadline - time.time() - 20.0)
            if sgld_time < 1.5:
                break
            sgld_c, sgld_e = _sgld_optimize(init_c, target_r, time_limit=sgld_time, seed=500 + t_idx * 7 + restart)
            
            # Polish with constraint satisfaction
            polish_time = min(6.0, deadline - time.time() - 15.0)
            if polish_time > 1.0:
                polished_c = _polish_lbfgs(sgld_c, target_r, time_limit=polish_time)
            else:
                polished_c = sgld_c
            
            r_new = _compute_radius(polished_c)
            if r_new > best_radius:
                best_radius = r_new
                best_centers = polished_c.copy()
    
    # Final high-precision polish: directly maximize smooth radius
    if time.time() < deadline - 5.0:
        try:
            polish_time = min(12.0, deadline - time.time() - 2.0)
            if polish_time > 2.0:
                # Try a few beta values for the smooth surrogate
                for beta_val in [150.0, 300.0]:
                    if time.time() > deadline - 3.0:
                        break
                    remaining = deadline - time.time() - 1.5
                    if remaining < 1.0:
                        break
                    polished_c = _polish_radius_max(best_centers, time_limit=min(remaining, 8.0), beta=beta_val)
                    r_new = _compute_radius(polished_c)
                    if r_new > best_radius:
                        best_radius = r_new
                        best_centers = polished_c.copy()
        except Exception:
            pass
    
    # Ultra-final tight polish
    if time.time() < deadline - 2.0:
        try:
            final_target = best_radius * 1.0001
            remaining = deadline - time.time() - 1.0
            if remaining > 1.0:
                polished_c = _polish_lbfgs(best_centers, final_target, time_limit=remaining)
                r_new = _compute_radius(polished_c)
                if r_new > best_radius:
                    best_radius = r_new
                    best_centers = polished_c.copy()
        except Exception:
            pass
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    return best_centers
