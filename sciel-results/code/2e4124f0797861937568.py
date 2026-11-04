import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_objective_and_grad(x, beta=50.0):
    """
    Smooth surrogate: maximize the soft-min of all constraint distances.
    Objective = -soft_min(2*r_pairwise, wall_dists)
    where soft_min uses log-sum-exp for smoothness.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    half_pair_dists = pair_dists / 2.0
    
    # Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # Combine all constraints
    all_dists = np.concatenate([half_pair_dists, wall_dists])
    
    # Soft-min via log-sum-exp (numerically stable)
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min_val = min_d - log_sum
    
    # Gradient of soft-min
    weights = np.exp(-beta * (all_dists - min_d))
    weights /= np.sum(weights)
    
    grad = np.zeros_like(centers)
    
    # Pairwise gradients
    for k in range(len(pair_dists)):
        w = weights[k]
        if w < 1e-10:
            continue
        i, j = i_idx[k], j_idx[k]
        vec = centers[i] - centers[j]
        d = pair_dists[k]
        if d < 1e-15:
            continue
        # d(half_pair_dist)/d(centers[i]) = vec / (2*d)
        # d(soft_min)/d(centers[i]) = w * vec / (2*d)
        grad[i] += w * vec / (2.0 * d)
        grad[j] -= w * vec / (2.0 * d)
    
    # Wall gradients
    for idx in range(n):
        w = weights[n*(n-1)//2 + idx]
        if w < 1e-10:
            continue
        max_k = np.argmax(np.abs(centers[idx]))
        sign = np.sign(centers[idx, max_k])
        # d(wall_dist)/d(centers[idx, max_k]) = -sign
        # d(soft_min)/d(centers[idx, max_k]) = w * (-sign)
        grad[idx, max_k] -= w * sign
    
    # We want to maximize soft_min, so minimize -soft_min
    objective = -soft_min_val
    return objective, grad.flatten()

def _hard_energy_and_grad(x, R):
    """
    Hard squared-violation energy for feasibility checking.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    mask_pairs = d_ij < 2.0 * R
    
    if np.any(mask_pairs):
        v_ij = 2.0 * R - d_ij[mask_pairs]
        d_valid = d_ij[mask_pairs]
        i_valid = i_idx[mask_pairs]
        j_valid = j_idx[mask_pairs]
        
        energy += np.sum(v_ij**2)
        
        factors = 2.0 * v_ij / d_valid
        vecs = centers[i_valid] - centers[j_valid]
        forces = factors[:, np.newaxis] * vecs
        
        np.add.at(grad, i_valid, -forces)
        np.add.at(grad, j_valid, forces)

    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(violations_wall**2)
        
        for idx in i_valid_wall:
            v = violations_wall[np.where(i_valid_wall == idx)[0][0]]
            max_k = np.argmax(abs_centers[idx])
            
            grad[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])
            
    return energy, grad.flatten()

def _adam_soft_min_optimize(init_centers, max_iter=1000, beta_start=10.0, beta_end=100.0, lr_start=0.001, lr_end=0.0001):
    """
    Optimize soft-min using Adam with annealed temperature (beta).
    """
    x = init_centers.flatten().copy()
    n_params = len(x)
    
    # Adam state
    m = np.zeros(n_params)
    v = np.zeros(n_params)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    best_x = x.copy()
    best_r = _compute_radius(x.reshape(N, 3))
    
    for it in range(max_iter):
        # Anneal beta and learning rate
        t_frac = it / max(1, max_iter - 1)
        beta = beta_start * (beta_end / beta_start) ** t_frac
        lr = lr_start * (lr_end / lr_start) ** t_frac
        
        # Compute objective and gradient
        obj, grad = _soft_min_objective_and_grad(x, beta=beta)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Project to bounds
        x = np.clip(x, -0.5, 0.5)
        
        # Check improvement
        centers = x.reshape(N, 3)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_x = x.copy()
    
    return best_x.reshape(N, 3), best_r

def _optimize_soft_min(init_centers, max_iter=500, ftol=1e-14, gtol=1e-12, beta=50.0):
    """
    Optimize using soft-min surrogate.
    """
    def obj_grad(x):
        val, g = _soft_min_objective_and_grad(x, beta=beta)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            obj_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        return new_centers, res.success
    except Exception:
        return init_centers, False

def _optimize_hard(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Optimize using hard energy for feasibility.
    """
    def obj_grad(x):
        val, g = _hard_energy_and_grad(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            obj_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _hard_energy_and_grad(new_centers, target_R)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def _generate_diverse_starts(rng, n_starts=5):
    """
    Generate diverse initial configurations.
    """
    starts = []
    
    # Grid-based
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    starts.append(grid[:N].copy() + rng.normal(0, 0.01, size=(N, 3)))
    
    # Random uniform
    starts.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # BCC-like lattice
    lattice_pts = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                x = -0.5 + (ix + 0.5) * 0.5
                y = -0.5 + (iy + 0.5) * 0.5
                z = -0.5 + (iz + 0.5) * 0.5
                lattice_pts.append([x, y, z])
                lattice_pts.append([x + 0.125, y + 0.125, z + 0.125])
    lattice = np.array(lattice_pts[:N])
    starts.append(lattice + rng.normal(0, 0.005, size=(N, 3)))
    
    # More random
    for _ in range(n_starts - 3):
        starts.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    return [np.clip(s, -0.49, 0.49) for s in starts]

def _final_squeeze(pos, current_r, time_budget):
    """
    Final high-precision squeeze loop to extract maximum radius.
    """
    best_pos = pos.copy()
    best_r = current_r
    deadline = time.time() + time_budget
    
    # Phase 1: Fine-grained target radius increase with Adam
    step = 1e-5
    consecutive_fail = 0
    
    while time.time() < deadline - 10.0:
        target_r = best_r + step
        
        if consecutive_fail >= 5:
            step *= 0.5
            consecutive_fail = 0
            if step < 1e-8:
                break
        
        # Try Adam soft-min then hard projection
        adam_pos, adam_r = _adam_soft_min_optimize(
            best_pos, max_iter=300, beta_start=20.0, beta_end=80.0,
            lr_start=0.0005, lr_end=0.00005
        )
        
        # Hard projection at target
        hard_pos, energy, _ = _optimize_hard(adam_pos, target_r, max_iter=500, ftol=1e-15, gtol=1e-13)
        hard_r = _compute_radius(hard_pos)
        
        # Also try from current best
        hard_pos2, energy2, _ = _optimize_hard(best_pos, target_r, max_iter=500, ftol=1e-15, gtol=1e-13)
        hard_r2 = _compute_radius(hard_pos2)
        
        # Pick best
        candidates = [(adam_r, adam_pos), (hard_r, hard_pos), (hard_r2, hard_pos2)]
        candidates.sort(key=lambda x: -x[0])
        best_cand_r, best_cand_pos = candidates[0]
        
        if best_cand_r > best_r:
            best_r = best_cand_r
            best_pos = best_cand_pos
            consecutive_fail = 0
        else:
            consecutive_fail += 1
    
    # Phase 2: Very tight final polish
    if time.time() < deadline - 2.0:
        # Multiple passes of hard optimization at current best radius
        for _ in range(3):
            if time.time() > deadline - 2.0:
                break
            polished, _, _ = _optimize_hard(best_pos, best_r, max_iter=1000, ftol=1e-16, gtol=1e-14)
            polished_r = _compute_radius(polished)
            if polished_r > best_r:
                best_r = polished_r
                best_pos = polished
    
    # Phase 3: One more Adam pass with very high beta for final polish
    if time.time() < deadline - 1.0:
        adam_pos, adam_r = _adam_soft_min_optimize(
            best_pos, max_iter=500, beta_start=50.0, beta_end=200.0,
            lr_start=0.0002, lr_end=0.00001
        )
        if adam_r > best_r:
            best_r = adam_r
            best_pos = adam_pos
    
    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # 1. Multi-start initialization with Adam
    starts = _generate_diverse_starts(rng, n_starts=5)
    
    best_pos = None
    best_r = 0.0
    
    for s in starts:
        if time.time() > deadline - 80.0:
            break
        
        # Use Adam with annealed beta
        pos, r = _adam_soft_min_optimize(
            s, max_iter=800, beta_start=8.0, beta_end=60.0,
            lr_start=0.001, lr_end=0.0001
        )
        
        if r > best_r:
            best_r = r
            best_pos = pos
    
    if best_pos is None or best_r <= 0:
        best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        best_r = _compute_radius(best_pos)
    
    # 2. Iterative radius increase
    current_r = best_r
    step_size = 0.0003
    
    consecutive_failures = 0
    max_failures = 5
    
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 40.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-8:
                break
        
        # Generate candidates: current best + perturbations
        candidates = [best_pos.copy()]
        for sigma in [0.002, 0.008, 0.02]:
            candidates.append(best_pos + rng_search.normal(0, sigma, size=(N, 3)))
        
        success = False
        best_candidate = None
        best_candidate_r = 0.0
        
        for start_pos in candidates:
            if time.time() > deadline - 45.0:
                break
            start_pos = np.clip(start_pos, -0.5, 0.5)
            
            # Adam soft-min optimization
            soft_pos, soft_r = _adam_soft_min_optimize(
                start_pos, max_iter=400, beta_start=15.0, beta_end=70.0,
                lr_start=0.0008, lr_end=0.00008
            )
            
            # Hard optimization at target radius
            hard_pos, energy, _ = _optimize_hard(soft_pos, target_r, max_iter=400)
            hard_r = _compute_radius(hard_pos)
            
            # Also try hard from original
            hard_pos2, energy2, _ = _optimize_hard(start_pos, target_r, max_iter=400)
            hard_r2 = _compute_radius(hard_pos2)
            
            # Pick the best result
            candidate_results = [(soft_r, soft_pos), (hard_r, hard_pos), (hard_r2, hard_pos2)]
            candidate_results.sort(key=lambda x: -x[0])
            cand_r, cand_pos = candidate_results[0]
            
            if cand_r > best_candidate_r:
                best_candidate_r = cand_r
                best_candidate = cand_pos
            
            # Check for success at target radius
            if energy < 1e-10 and hard_r >= target_r - 1e-7:
                best_pos = hard_pos
                best_r = hard_r
                current_r = hard_r
                success = True
                consecutive_failures = 0
                break
        
        if not success:
            # Accept improvement even if target not fully met
            if best_candidate is not None and best_candidate_r > current_r:
                best_pos = best_candidate
                best_r = best_candidate_r
                current_r = best_candidate_r
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                
                # Polish current best with Adam
                if time.time() < deadline - 50.0 and consecutive_failures % 3 == 0:
                    polished, polished_r = _adam_soft_min_optimize(
                        best_pos, max_iter=400, beta_start=30.0, beta_end=100.0,
                        lr_start=0.0004, lr_end=0.00004
                    )
                    if polished_r > best_r:
                        best_pos = polished
                        best_r = polished_r
                        current_r = best_r

    # 3. Final high-precision squeeze
    if time.time() < deadline - 5.0:
        time_remaining = deadline - time.time() - 5.0
        best_pos, best_r = _final_squeeze(best_pos, best_r, time_remaining)

    return best_pos
