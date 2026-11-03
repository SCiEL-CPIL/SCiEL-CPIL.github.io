"""Combined program: Iterative Radius Increase with Energy Minimization."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 100
HALF = 0.5
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _compute_radius(C):
    """Compute the maximum possible radius for a set of centers C."""
    if len(C) < 2:
        return HALF - np.max(np.abs(C))
    dist_to_bound = HALF - np.abs(C).max(axis=1)
    r_bound = np.min(dist_to_bound)
    tree = cKDTree(C)
    distances, _ = tree.query(C, k=2)
    min_dist = np.min(distances[:, 1])
    r_pair = min_dist / 2.0
    return min(r_bound, r_pair)

def _softplus(x):
    """Numerically stable softplus."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def _energy_and_gradient(r, flat_pos, beta=50.0):
    """
    Compute energy and gradient for fixed radius r.
    Energy is sum of squared softplus penalties for overlaps and boundary violations.
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Boundary distances
    d_walls = HALF - np.abs(pos)
    wall_dists = d_walls.flatten()
    
    # Penalties
    # Overlap: 2r - d > 0 means overlap. We want to minimize softplus(beta * (2r - d))
    # Boundary: d_wall < 0 means violation. We want to minimize softplus(beta * (-d_wall))
    
    overlap_arg = beta * (2.0 * r - pair_dists)
    wall_arg = beta * (-wall_dists)
    
    sp_overlap = _softplus(overlap_arg)
    sp_wall = _softplus(wall_arg)
    
    energy = np.sum(sp_overlap**2) + np.sum(sp_wall**2)
    
    # Gradients
    # d(softplus(x)^2)/dx = 2 * softplus(x) * sigmoid(x)
    # sigmoid(x) = 1 / (1 + exp(-x))
    
    grad_pos = np.zeros_like(pos)
    
    # Pairwise gradient
    sig_overlap = 1.0 / (1.0 + np.exp(-overlap_arg))
    grad_pair_val = 2.0 * sp_overlap * sig_overlap * (-beta) # d(arg)/d(d) = -beta, d(energy)/d(d) = ...
    # Wait, chain rule:
    # E = sum (sp(arg))^2
    # dE/darg = 2 * sp(arg) * sigmoid(arg)
    # darg/dd = -beta
    # dE/dd = -2 * beta * sp(arg) * sigmoid(arg)
    # dE/dpos_i = sum_{j} dE/dd_ij * d(d_ij)/dpos_i
    # d(d_ij)/dpos_i = (pos_i - pos_j) / d_ij
    
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d = pair_dists[idx]
        if d < 1e-12:
            continue
        coeff = grad_pair_val[idx] / d
        vec = pos[i] - pos[j]
        grad_pos[i] += coeff * vec
        grad_pos[j] -= coeff * vec
        
    # Boundary gradient
    sig_wall = 1.0 / (1.0 + np.exp(-wall_arg))
    grad_wall_val = 2.0 * sp_wall * sig_wall * (-beta) # darg/dd_wall = -beta
    # dE/dd_wall = -2 * beta * sp(arg) * sigmoid(arg)
    # d(d_wall)/dpos_ik = -sign(pos_ik)
    # dE/dpos_ik = dE/dd_wall * d(d_wall)/dpos_ik = (-2 * beta * sp * sig) * (-sign) = 2 * beta * sp * sig * sign
    
    for idx in range(N * 3):
        i = idx // 3
        k = idx % 3
        s = np.sign(pos[i, k])
        if s == 0: s = 1.0
        # grad_wall_val is dE/dd_wall.
        # dE/dpos = grad_wall_val * (-s)
        # Wait, d_wall = HALF - |pos|. d(d_wall)/dpos = -sign(pos).
        # So dE/dpos = dE/dd_wall * (-sign(pos)).
        # dE/dd_wall = 2 * sp * sig * (-beta).
        # So dE/dpos = 2 * beta * sp * sig * sign(pos).
        grad_pos[i, k] += 2.0 * beta * sp_wall[idx] * sig_wall[idx] * s
        
    return energy, grad_pos.flatten()

def _minimize_energy(r, pos, max_iter=200, ftol=1e-12, gtol=1e-10):
    """Minimize energy for fixed radius r."""
    bounds = [(-HALF, HALF)] * (N * 3)
    try:
        res = minimize(
            lambda x: _energy_and_gradient(r, x)[0],
            pos.flatten(),
            jac=lambda x: _energy_and_gradient(r, x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if np.isfinite(res.fun):
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def _generate_initial_points(seed):
    """Generate diverse initial points."""
    rng = np.random.default_rng(seed)
    # Mix of BCC, FCC, and Random
    method = seed % 3
    if method == 0:
        # BCC-like
        size = 4
        raw = []
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    raw.append([i, j, k])
                    raw.append([i + 0.5, j + 0.5, k + 0.5])
        raw = np.array(raw)
        center = np.array([size/2 - 0.25, size/2 - 0.25, size/2 - 0.25])
        raw = raw - center
        max_coord = np.max(np.abs(raw))
        scale = 0.45 / max_coord
        raw *= scale
        idx = rng.permutation(len(raw))
        return raw[idx[:N]].copy()
    elif method == 1:
        # FCC-like
        size = 4
        raw = []
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    raw.append([i, j, k])
                    raw.append([i + 0.5, j + 0.5, k])
                    raw.append([i + 0.5, j, k + 0.5])
                    raw.append([i, j + 0.5, k + 0.5])
        raw = np.array(raw)
        center = np.array([size/2 - 0.25, size/2 - 0.25, size/2 - 0.25])
        raw = raw - center
        max_coord = np.max(np.abs(raw))
        scale = 0.45 / max_coord
        raw *= scale
        idx = rng.permutation(len(raw))
        return raw[idx[:N]].copy()
    else:
        # Random
        return rng.uniform(-0.45, 0.45, (N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - SAFETY_MARGIN)
    
    best_radius = 0.0
    best_centers = None
    
    # Start with a reasonable radius estimate
    # BCC packing density in cube is roughly 0.68.
    # Volume of 100 spheres = 100 * 4/3 pi r^3.
    # 100 * 4/3 pi r^3 <= 0.68 * 1
    # r^3 <= 0.68 / (400/3 pi) = 0.68 / 418.8 = 0.00162
    # r <= 0.117
    # Let's start at 0.10 to be safe.
    r_current = 0.10
    
    # Try multiple seeds
    seeds = [42, 123, 456, 789, 101112, 202021, 303031, 404041]
    
    for seed in seeds:
        if _check_time():
            break
            
        centers = _generate_initial_points(seed)
        centers = np.clip(centers, -HALF + 1e-9, HALF - 1e-9)
        
        # Iterative Radius Increase
        r_try = r_current
        improved = False
        max_iter_r = 20 # Number of radius increase steps
        
        for _ in range(max_iter_r):
            if _check_time():
                break
                
            # Minimize energy for current r_try
            optimized_centers = _minimize_energy(r_try, centers, max_iter=100)
            
            # Check if feasible
            actual_radius = _compute_radius(optimized_centers)
            
            if actual_radius >= r_try - 1e-6:
                # Feasible! Increase radius
                r_try += 0.002
                centers = optimized_centers
                improved = True
            else:
                # Not feasible, stop increasing
                break
        
        # Final polish with the highest feasible radius found for this seed
        # Use the actual radius achieved
        final_radius = _compute_radius(centers)
        
        if final_radius > best_radius:
            best_radius = final_radius
            best_centers = centers.copy()
            
        # Try to improve best_radius further with a global SA + Polish loop if time permits
        if time.time() < deadline - 20.0 and best_radius > 0:
             # Use the best_centers as starting point for SA
             T = 0.01
             rng = np.random.default_rng(seed + 1000)
             sa_iters = 0
             while T > 0.001 and time.time() < deadline - 10.0 and sa_iters < 500:
                 i = rng.integers(N)
                 move = rng.normal(0, T, 3)
                 new_centers = best_centers.copy()
                 new_centers[i] += move
                 if np.any(np.abs(new_centers[i]) > HALF - 1e-9):
                     T *= 0.95
                     sa_iters += 1
                     continue
                 new_r = _compute_radius(new_centers)
                 if new_r > best_radius or rng.random() < np.exp((new_r - best_radius) / T):
                     best_centers = new_centers
                     best_radius = new_r
                 T *= 0.95
                 sa_iters += 1
             
             # Final polish
             if not _check_time():
                 # Use soft-min polish on the best found
                 # Re-using the soft-min function from the previous version as it's very good for final steps
                 # But we need to define it or use energy minimization at high r.
                 # Let's do one more energy minimization at r = best_radius * 1.001
                 r_polish = best_radius * 1.001
                 polished = _minimize_energy(r_polish, best_centers, max_iter=200)
                 r_polished = _compute_radius(polished)
                 if r_polished > best_radius:
                     best_radius = r_polished
                     best_centers = polished
                     
    # Final high-precision polish using the soft-min approach if time remains
    if not _check_time() and best_centers is not None:
        # Define soft-min objective inline for final polish
        def _soft_min_obj(flat_pos, beta=100.0):
            pos = flat_pos.reshape(N, 3)
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            dists = np.sqrt(dists_sq)
            iu = np.triu_indices(N, k=1)
            pair_dists = dists[iu]
            d_walls = HALF - np.abs(pos)
            wall_dists = d_walls.flatten()
            all_eff_dists = np.concatenate([pair_dists / 2.0, wall_dists])
            scaled = -beta * all_eff_dists
            max_scaled = np.max(scaled)
            log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
            soft_min_val = -log_sum_exp / beta
            weights = np.exp(scaled - max_scaled)
            weights /= np.sum(weights)
            grad_pos = np.zeros_like(pos)
            for idx in range(len(iu[0])):
                i, j = iu[0][idx], iu[1][idx]
                w = weights[idx]
                d = pair_dists[idx]
                if d > 1e-14:
                    vec = pos[i] - pos[j]
                    coeff = w / (2.0 * d)
                    grad_pos[i] += coeff * vec
                    grad_pos[j] -= coeff * vec
            for idx in range(N * 3):
                i = idx // 3
                k = idx % 3
                w = weights[len(pair_dists) + idx]
                s = np.sign(pos[i, k])
                if s == 0: s = 1.0
                grad_pos[i, k] -= w * s
            return -soft_min_val, -grad_pos.flatten()
            
        bounds = [(-HALF, HALF)] * (N * 3)
        try:
            res = minimize(
                lambda x: _soft_min_obj(x)[0],
                best_centers.flatten(),
                jac=lambda x: _soft_min_obj(x)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if np.isfinite(res.fun):
                new_pos = res.x.reshape(N, 3)
                if np.all(np.isfinite(new_pos)):
                    r_final = _compute_radius(new_pos)
                    if r_final > best_radius:
                        best_radius = r_final
                        best_centers = new_pos
        except Exception:
            pass

    best_centers = np.clip(best_centers, -HALF + 1e-10, HALF - 1e-10)
    return best_centers
