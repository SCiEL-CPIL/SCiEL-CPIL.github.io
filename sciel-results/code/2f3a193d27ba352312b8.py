"""Combined solver: smooth surrogate optimization + repulsion annealing + exact polish for 50-sphere packing.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5, 0.5]^3."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 150s kill

def _compute_radius(pts):
    """Compute the maximum valid radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 3):
        return -1e6
    if not np.all(np.isfinite(pts)):
        return -1e6
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _neg_radius(x):
    """Objective: negative radius (to be minimized)."""
    pts = x.reshape(N, 3)
    return -_compute_radius(pts)

def _soft_min_radius_and_grad(x, beta=50.0):
    """
    Smooth surrogate for negative radius and its gradient.
    Uses log-sum-exp soft-min.
    """
    pts = x.reshape(N, 3)
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e6, np.zeros_like(x)
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-12)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(dists_sq, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_flat = dists[iu] / 2.0
    
    # Wall distances
    wall_dists = (0.5 - np.abs(pts)).flatten()
    
    all_dists = np.concatenate([pair_flat, wall_dists])
    
    # Stable soft-min: min(d) ≈ d_min - (1/beta)*log(sum(exp(-beta*(d-d_min))))
    d_min = np.min(all_dists)
    shifted = -beta * (all_dists - d_min)
    # Clip to prevent overflow
    shifted = np.clip(shifted, -700, 0)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum = np.log(sum_exp)
    soft_min_val = d_min - log_sum / beta
    
    # Gradient calculation
    weights = exp_shifted / sum_exp
    
    # Now chain rule to x
    grad_x = np.zeros_like(pts)
    
    # Pairwise part: indices 0 to len(pair_flat)-1
    n_pairs = len(pair_flat)
    pair_grads = -weights[:n_pairs]
    
    # For each pair (i, j), d_ij = 0.5 * ||p_i - p_j||
    # d(d_ij)/d(p_i) = 0.5 * (p_i - p_j) / ||p_i - p_j||
    # d(d_ij)/d(p_j) = 0.5 * (p_j - p_i) / ||p_i - p_j||
    
    # Vectorized:
    i_idx, j_idx = iu
    valid_dists = dists[i_idx, j_idx]
    valid_dists = np.maximum(valid_dists, 1e-8)
    
    dir_vecs = diff[i_idx, j_idx] / valid_dists[:, None]
    weights_pair = pair_grads * 0.5
    
    grad_i = weights_pair[:, None] * dir_vecs
    grad_j = -grad_i
    
    np.add.at(grad_x, i_idx, grad_i)
    np.add.at(grad_x, j_idx, grad_j)
    
    # Wall part: indices n_pairs to end
    wall_grads = -weights[n_pairs:]
    # wall_dists = (0.5 - |p_ik|).flatten()
    # d(0.5 - |x|)/dx = -sign(x)
    # So grad_x_ik += wall_grads[k] * (-sign(p_ik))
    
    wall_grads_reshaped = wall_grads.reshape(N, 3)
    signs = np.sign(pts)
    grad_x += wall_grads_reshaped * (-signs)
    
    return -soft_min_val, grad_x.flatten()

def _repulsion_anneal(pts, n_steps=50, lr_start=0.01, lr_end=0.001):
    """
    Repulsion annealing: iteratively push spheres apart using gradient of repulsive potential.
    Vectorized for speed.
    """
    pts = pts.copy().astype(np.float64)
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 30.0:
            break
        lr = lr_start * (lr_end / lr_start) ** (step / max(1, n_steps - 1))
        
        # Compute forces (vectorized)
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion: compute all pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
        dists = np.sqrt(np.sum(diff * diff, axis=-1))  # (N, N)
        np.fill_diagonal(dists, 1e6)  # Avoid division by zero on diagonal
        
        # Force magnitude: 1/dist^2, applied along direction
        # Force on i from j: (1/dist^2) * (pts[i]-pts[j])/dist = diff[i,j] / dist^3
        inv_dist3 = 1.0 / (dists ** 3 + 1e-8)
        # Zero out diagonal
        np.fill_diagonal(inv_dist3, 0)
        
        # forces[i] = sum_j inv_dist3[i,j] * diff[i,j]
        forces += np.sum(inv_dist3[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion: push away from walls
        # For each axis, compute distance to nearest wall
        for k in range(3):
            pos = pts[:, k]
            # Distance to positive wall: 0.5 - pos (for pos > 0)
            # Distance to negative wall: 0.5 + pos (for pos < 0)
            wall_dist_pos = 0.5 - pos
            wall_dist_neg = 0.5 + pos
            
            # Force from positive wall (push left if close to right wall)
            force_pos = np.where(wall_dist_pos < 0.15, -1.0 / (wall_dist_pos + 0.01), 0.0)
            # Force from negative wall (push right if close to left wall)
            force_neg = np.where(wall_dist_neg < 0.15, 1.0 / (wall_dist_neg + 0.01), 0.0)
            
            forces[:, k] += force_pos + force_neg
        
        # Update positions
        pts += lr * forces
        pts = np.clip(pts, -0.495, 0.495)
    
    return pts

def _metropolis_mc(pts, n_steps=2000, temp_start=0.05, temp_end=0.005):
    """
    Metropolis Monte Carlo to escape local minima.
    Tries random perturbations and accepts if radius improves or with probability exp(-delta/temp).
    """
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    rng = np.random.default_rng(42)
    
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 20.0:
            break
        frac = step / max(n_steps - 1, 1)
        T = temp_start * (temp_end / temp_start) ** frac
        
        trial = pts.copy()
        trial += rng.normal(scale=T, size=(N, 3))
        trial = np.clip(trial, -0.499, 0.499)
        
        r_trial = _compute_radius(trial)
        r_cur = _compute_radius(pts)
        
        if r_trial >= r_cur:
            pts = trial
            if r_trial > best_r:
                best_r = r_trial
                best_pts = trial.copy()
        else:
            delta_r = r_cur - r_trial
            if delta_r < T * 10:
                accept_prob = np.exp(-delta_r / max(T, 1e-10))
                if rng.random() < accept_prob:
                    pts = trial
    
    return best_pts, best_r

def _local_search(pts, n_steps=3000):
    """
    Local search: repeatedly find the most constrained pair or wall and adjust.
    """
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    rng = np.random.default_rng(123)
    
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 10.0:
            break
        
        # Find the minimum constraint
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_d = np.min(dists) / 2.0
        min_pair_idx = np.argmin(dists)
        i, j = np.unravel_index(min_pair_idx, (N, N))
        
        wall_dists = 0.5 - np.abs(pts)
        min_wall = np.min(wall_dists)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        wi, wk = min_wall_idx
        
        # Decide which constraint to relax
        if min_pair_d < min_wall:
            # Relax pair constraint
            direction = pts[i] - pts[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            else:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
            
            improved = False
            for scale in (0.005, 0.002, 0.001, 0.0005, 0.0002):
                trial = pts.copy()
                trial[i] += scale * direction
                trial[j] -= scale * direction
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    pts = trial
                    best_r = r_trial
                    best_pts = trial.copy()
                    improved = True
                    break
            if improved:
                continue
        else:
            # Relax wall constraint
            sgn = np.sign(pts[wi, wk])
            if sgn == 0:
                sgn = 1.0
            direction = np.zeros(3)
            direction[wk] = sgn
            
            improved = False
            for scale in (0.005, 0.002, 0.001, 0.0005, 0.0002):
                trial = pts.copy()
                trial[wi] += scale * direction
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    pts = trial
                    best_r = r_trial
                    best_pts = trial.copy()
                    improved = True
                    break
            if improved:
                continue
        
        # Random perturbation to escape local optima
        if step % 100 == 0 and step > 0:
            for scale in (0.01, 0.005, 0.002, 0.001):
                trial = pts.copy()
                trial += rng.normal(scale=scale, size=(N, 3))
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    pts = trial
                    best_r = r_trial
                    best_pts = trial.copy()
                    break
    
    return best_pts, best_r

def _seed_fcc():
    """FCC lattice supercell, then farthest-point sample to N points."""
    pts = []
    a = 1.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, k*a])
                pts.append([(i+0.5)*a, j*a, (k+0.5)*a])
                pts.append([i*a, (j+0.5)*a, (k+0.5)*a])
    pts = np.array(pts)
    pts = pts - pts.mean(axis=0)
    me = np.max(np.abs(pts))
    if me > 0:
        pts = pts / me * 0.45

    sel = [int(np.argmin(np.sum(pts**2, axis=1)))]
    rem = list(range(len(pts)))
    rem.remove(sel[0])
    while len(sel) < N and rem:
        dm = np.full(len(rem), np.inf)
        for ri, idx in enumerate(rem):
            dmin = np.min(np.linalg.norm(pts[idx] - pts[sel], axis=1))
            dm[ri] = dmin
        bi = int(np.argmax(dm))
        sel.append(rem[bi])
        rem.pop(bi)
    return pts[sel]

def _seed_grid():
    m = int(np.ceil(N ** (1.0/3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return g[:N]

def _seed_random(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def _seed_shell_core():
    """Points on a spherical shell plus a central cluster."""
    rng = np.random.default_rng(7)
    shell_n = 35
    core_n = N - shell_n
    idx = np.arange(shell_n)
    phi = np.arccos(1 - 2 * (idx + 0.5) / shell_n)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    R = 0.32
    shell = np.column_stack([
        R * np.sin(phi) * np.cos(theta),
        R * np.sin(phi) * np.sin(theta),
        R * np.cos(phi),
    ])
    shell = np.clip(shell, -0.45, 0.45)
    ticks = (np.arange(3) + 0.5) / 3 - 0.5
    core = np.array([(x * 0.2, y * 0.2, z * 0.2)
                     for z in ticks for y in ticks for x in ticks])[:core_n]
    return np.vstack([shell, core])

def _seed_bcc():
    """BCC lattice points, scaled to fit in cube."""
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts, dtype=float)
    pts = (pts - 1.0)
    pts = pts / np.max(np.abs(pts)) * 0.45
    sel = [int(np.argmin(np.sum(pts**2, axis=1)))]
    rem = list(range(len(pts)))
    rem.remove(sel[0])
    while len(sel) < N and rem:
        dm = np.full(len(rem), np.inf)
        for ri, idx in enumerate(rem):
            dmin = np.min(np.linalg.norm(pts[idx] - pts[sel], axis=1))
            dm[ri] = dmin
        bi = int(np.argmax(dm))
        sel.append(rem[bi])
        rem.pop(bi)
    return pts[sel]

def construct():
    """Maximize the common radius of 50 equal spheres in [-0.5,0.5]^3."""
    start = time.time()
    deadline = start + (150.0 - 15.0)

    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_shell_core(),
        _seed_bcc(),
        _seed_random(1),
        _seed_random(2),
    ]

    best_c = None
    best_r = -1.0

    for si, s in enumerate(seeds):
        if time.time() > deadline:
            break
        s = np.clip(s, -0.49, 0.49)

        # Stage A: repulsion annealing
        c_an = _repulsion_anneal(s, n_steps=100)
        r_an = _compute_radius(c_an)

        # Stage B: smooth softmin polish
        c_cur = c_an
        r_cur = r_an
        x0 = c_cur.flatten()
        bounds = [(-0.49, 0.49)] * (N * 3)
        
        if time.time() < deadline:
            for beta in (20.0, 50.0, 100.0, 200.0, 500.0):
                if time.time() > deadline:
                    break
                try:
                    res = minimize(
                        _soft_min_radius_and_grad,
                        x0,
                        jac=True,
                        method='L-BFGS-B',
                        bounds=bounds,
                        args=(beta,),
                        options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12,
                                 'maxls': 50},
                    )
                    c_new = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
                    r_new = _compute_radius(c_new)
                    if r_new > r_cur:
                        r_cur = r_new
                        c_cur = c_new
                    x0 = c_cur.flatten()
                except Exception:
                    pass

        # Stage C: local search
        if time.time() < deadline:
            c_ls, r_ls = _local_search(c_cur, n_steps=2000)
            if r_ls > r_cur:
                r_cur = r_ls
                c_cur = c_ls

        # Stage D: Metropolis MC to escape local minima
        if time.time() < deadline:
            c_mc, r_mc = _metropolis_mc(c_cur, n_steps=1500)
            if r_mc > r_cur:
                r_cur = r_mc
                c_cur = c_mc

        if r_cur > best_r:
            best_r = r_cur
            best_c = c_cur.copy()

    # Final extra polish on the global best if time remains
    if best_c is not None and time.time() < deadline:
        x0 = best_c.flatten()
        bounds = [(-0.49, 0.49)] * (N * 3)
        
        if time.time() < deadline:
            for beta in (100.0, 200.0, 500.0, 1000.0):
                if time.time() > deadline:
                    break
                try:
                    res = minimize(
                        _soft_min_radius_and_grad,
                        x0,
                        jac=True,
                        method='L-BFGS-B',
                        bounds=bounds,
                        args=(beta,),
                        options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12,
                                 'maxls': 50},
                    )
                    c_new = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
                    r_new = _compute_radius(c_new)
                    if r_new > best_r:
                        best_r = r_new
                        best_c = c_new
                    x0 = best_c.flatten()
                except Exception:
                    pass

        if time.time() < deadline:
            c_ls, r_ls = _local_search(best_c, n_steps=1500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls

        if time.time() < deadline:
            c_mc, r_mc = _metropolis_mc(best_c, n_steps=1000)
            if r_mc > best_r:
                best_r = r_mc
                best_c = c_mc

    if best_c is None:
        best_c = _seed_grid()

    best_c = np.clip(best_c, -0.499999, 0.499999)
    return best_c

# EVOLVE-BLOCK-END
