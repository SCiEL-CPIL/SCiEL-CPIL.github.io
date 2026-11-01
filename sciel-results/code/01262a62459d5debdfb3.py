"""Combined sphere packing program: soft-min L-BFGS with annealing + force-directed refinement."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100

# EVOLVE-BLOCK-START

def _compute_r(centres):
    """Compute the maximum valid radius for given centres in [-0.5, 0.5]^3."""
    if not np.all(np.isfinite(centres)):
        return 0.0
    if centres.shape != (N, 3):
        return 0.0
    wall_dists = np.min(0.5 - np.abs(centres), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    tree = cKDTree(centres)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        return min_wall
    diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_wall)


def _soft_min_obj_and_grad(x, beta, iu):
    """
    Soft-min objective: G(x) = (1/beta) * ln(sum exp(-beta * c_k))
    As beta -> inf, G(x) -> -min_k c_k(x), so minimizing G maximizes min c_k.
    """
    c = x.reshape(-1, 3)
    
    # Wall constraints
    wall_vals = 0.5 - np.abs(c)  # (N, 3)
    wall_flat = wall_vals.flatten()  # (3N,)
    
    # Pair constraints
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (N, N, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=-1))  # (N, N)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_flat = pair_vals[iu]  # (num_pairs,)
    
    all_vals = np.concatenate([wall_flat, pair_flat])
    
    # Stable log-sum-exp
    neg_beta_c = -beta * all_vals
    max_nbc = np.max(neg_beta_c)
    exp_vals = np.exp(neg_beta_c - max_nbc)
    S = np.sum(exp_vals)
    if S < 1e-300:
        S = 1e-300
    weights = exp_vals / S
    
    obj = (max_nbc + np.log(S)) / beta
    
    # Gradient
    grad = np.zeros_like(c)
    
    # Wall: dG/dc[i,k] = w_{i,k} * sign(c[i,k])
    wall_weights = weights[:3 * N].reshape(-1, 3)
    signs = np.sign(c)
    signs[c == 0] = 1.0
    grad += wall_weights * signs
    
    # Pairs: dG/dc_i = -sum_j w_ij * (c_i - c_j)/dist_ij
    pair_weights = weights[3 * N:]
    mask = dists > 1e-12
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    unit_diff = diff * inv_dists[:, :, np.newaxis]
    
    w_mat = np.zeros((N, N))
    w_mat[iu] = pair_weights
    w_mat[iu[1], iu[0]] = pair_weights
    
    grad += -np.sum(w_mat[:, :, np.newaxis] * unit_diff, axis=1)
    
    return obj, grad.ravel()


def _optimize_soft_min(x0, deadline, beta_start=10.0, beta_end=5000.0, n_steps=10, maxiter_per_beta=150):
    """Annealed soft-min L-BFGS optimization."""
    best_x = x0.copy()
    best_r = _compute_radius(x0.reshape(-1, 3))
    iu = np.triu_indices(N, k=1)
    bounds = [(-0.4999, 0.4999)] * (N * 3)
    
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_x,
                args=(beta, iu),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter_per_beta, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            new_x = np.clip(res.x, -0.4999, 0.4999)
            new_r = _compute_radius(new_x.reshape(-1, 3))
            if new_r > best_r:
                best_r = new_r
                best_x = new_x.copy()
        except Exception:
            pass
    
    return best_x, best_r


def _force_refine(centres, max_iters=200, dt_init=0.005, deadline=None):
    """Force-directed repulsion with wall constraints."""
    centres = centres.copy().astype(np.float64)
    best_centres = centres.copy()
    best_r = _compute_r(best_centres)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        tree = cKDTree(centres)
        pairs = tree.query_pairs(0.6, output_type='ndarray')
        
        forces = np.zeros_like(centres)
        
        if len(pairs) > 0:
            for idx in range(len(pairs)):
                i, j = pairs[idx]
                d_vec = centres[i] - centres[j]
                d = np.sqrt(np.sum(d_vec**2))
                if d < 1e-12:
                    continue
                f = 1.0 / (d**2 + 1e-6)
                forces[i] += f * d_vec / d
                forces[j] -= f * d_vec / d
        
        # Wall forces
        for axis in range(3):
            over = centres[:, axis] - 0.5
            mask_pos = over > -0.03
            if np.any(mask_pos):
                forces[mask_pos, axis] -= 30.0 * np.clip(over[mask_pos], 0, None) + 2.0 * np.clip(0.03 + over[mask_pos], 0, None)
            under = centres[:, axis] + 0.5
            mask_neg = under < 0.03
            if np.any(mask_neg):
                forces[mask_neg, axis] += 30.0 * np.clip(-under[mask_neg], 0, None) + 2.0 * np.clip(0.03 - under[mask_neg], 0, None)
        
        dt = dt_init * (0.997 ** it)
        centres += dt * forces
        centres = np.clip(centres, -0.4999, 0.4999)
        
        r = _compute_r(centres)
        if r > best_r:
            best_r = r
            best_centres = centres.copy()
    
    return best_centres, best_r


def _init_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _init_fcc():
    """FCC lattice initialization."""
    a = 1.0 / 4.0
    points = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                x = i * a
                y = j * a
                z = k * a
                points.append([x - 0.5, y - 0.5, z - 0.5])
                points.append([x - 0.5, y + 0.5 * a - 0.5, z + 0.5 * a - 0.5])
                points.append([x + 0.5 * a - 0.5, y - 0.5, z + 0.5 * a - 0.5])
                points.append([x + 0.5 * a - 0.5, y + 0.5 * a - 0.5, z - 0.5])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    if len(points) < N:
        grid = _init_grid()
        points = np.vstack([points, grid[len(points):]])
    
    return points[:N].copy()


def _init_bcc():
    """BCC lattice initialization."""
    a = 1.0 / 4.0
    points = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                x = i * a
                y = j * a
                z = k * a
                points.append([x - 0.5, y - 0.5, z - 0.5])
                points.append([x + 0.5 * a - 0.5, y + 0.5 * a - 0.5, z + 0.5 * a - 0.5])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    if len(points) < N:
        grid = _init_grid()
        points = np.vstack([points, grid[len(points):]])
    
    return points[:N].copy()


def _init_random():
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, (N, 3))


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    initializations = [
        _init_grid,
        _init_fcc,
        _init_bcc,
        _init_random,
    ]
    
    best_centres = None
    best_r = 0.0
    
    for init_fn in initializations:
        if time.time() > deadline - 25.0:
            break
        
        init = init_fn()
        init = np.clip(init, -0.49, 0.49)
        
        # Force-directed refinement
        c_refined, r_refined = _force_refine(init, max_iters=120, dt_init=0.005, deadline=deadline - 20.0)
        
        if r_refined > best_r:
            best_r = r_refined
            best_centres = c_refined.copy()
        
        if time.time() > deadline - 20.0:
            break
        
        # Soft-min annealed L-BFGS
        c_opt, r_opt = _optimize_soft_min(c_refined, deadline=deadline - 15.0,
                                          beta_start=10.0, beta_end=3000.0, n_steps=8, maxiter_per_beta=100)
        
        if r_opt > best_r:
            best_r = r_opt
            best_centres = c_opt.copy()
    
    # Perturbation restarts from best solution
    if best_centres is not None and time.time() < deadline - 30.0:
        for pert_scale in [0.02, 0.04, 0.06, 0.08]:
            if time.time() > deadline - 20.0:
                break
            
            perturbed = best_centres + np.random.randn(N, 3) * pert_scale
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            c_refined, r_refined = _force_refine(perturbed, max_iters=80, dt_init=0.004, deadline=deadline - 15.0)
            
            if r_refined > best_r:
                best_r = r_refined
                best_centres = c_refined.copy()
            
            if time.time() > deadline - 15.0:
                break
            
            c_opt, r_opt = _optimize_soft_min(c_refined, deadline=deadline - 10.0,
                                              beta_start=20.0, beta_end=5000.0, n_steps=6, maxiter_per_beta=80)
            
            if r_opt > best_r:
                best_r = r_opt
                best_centres = c_opt.copy()
    
    # Final polish with high-beta soft-min
    if best_centres is not None and time.time() < deadline - 5.0:
        c_final, r_final = _optimize_soft_min(best_centres, deadline=deadline - 2.0,
                                              beta_start=500.0, beta_end=10000.0, n_steps=5, maxiter_per_beta=200)
        if r_final > best_r:
            best_r = r_final
            best_centres = c_final.copy()
    
    # Final validation
    if best_centres is not None:
        final_r = _compute_r(best_centres)
        if final_r > best_r:
            best_r = final_r
        return best_centres
    else:
        return _init_grid()

# EVOLVE-BLOCK-END
