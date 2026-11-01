import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0

def compute_radius(pts):
    """
    Compute the maximum radius for the given centers.
    Returns -1e6 if invalid (outside cube).
    """
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall_dist)

def neg_radius(pts):
    r = compute_radius(pts)
    return -r

def soft_min_radius(pts, beta=100.0):
    """
    Smooth surrogate for the radius using log-sum-exp soft-min.
    This is differentiable and amenable to gradient-based optimization.
    """
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise distances / 2
    pair_dists = dists / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    # Flatten wall distances: shape (N, 3)
    wall_flat = wall_dists.flatten()
    
    # Combine all distances
    pair_flat = pair_dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([pair_flat, wall_flat])
    
    # Soft-min via log-sum-exp: -log(sum(exp(-beta * d))) / beta
    # For numerical stability, shift by min
    d_min = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - d_min))
    soft_min_val = d_min - np.log(np.sum(exp_terms)) / beta
    
    return soft_min_val

def neg_soft_radius(pts, beta=100.0):
    r = soft_min_radius(pts, beta)
    return -r

def neg_soft_radius_grad(pts, beta=100.0):
    """
    Compute gradient of the soft-min radius objective.
    """
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return np.ones_like(pts)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    pair_dists = dists / 2.0
    wall_dists = 0.5 - np.abs(pts)
    wall_flat = wall_dists.flatten()
    
    pair_flat = pair_dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([pair_flat, wall_flat])
    
    d_min = np.min(all_dists)
    shifted = all_dists - d_min
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    grad = np.zeros_like(pts)
    
    # Pairwise contributions
    # all_dists[:N*(N-1)/2] are pairwise
    n_pairs = N * (N - 1) // 2
    pair_weights = weights[:n_pairs]
    
    # Reconstruct which pairs these correspond to
    i_idx, j_idx = np.triu_indices(N, k=1)
    for idx in range(n_pairs):
        i, j = i_idx[idx], j_idx[idx]
        w = pair_weights[idx]
        d_ij = dists[i, j]
        if d_ij > 1e-15:
            # d_pair = |p_i - p_j| / 2
            # grad w.r.t. p_i: w * (-beta) * exp(-beta*(d-pair_min)) * (-1) * (p_i - p_j)/|p_i-p_j| / 2
            # = w * beta * exp_term * (p_i - p_j) / (2 * |p_i-p_j|) * (-1) for soft-min
            # Actually: soft_min = d_min - log(sum(exp(-beta*(d-d_min))))/beta
            # d(soft_min)/d(p_i) = -1/beta * d(log(sum))/dp_i
            # = -1/beta * sum_j w_j * d(-beta*(d_ij - d_min))/dp_i
            # = sum_j w_j * d(d_ij)/dp_i  (the -beta and -1/beta cancel, but sign...)
            # Let me re-derive:
            # f = d_min - (1/beta) * log(sum_k exp(-beta*(d_k - d_min)))
            # df/dp_i = - (1/beta) * [sum_k exp(-beta*(d_k-d_min)) * (-beta) * dd_k/dp_i] / sum_k exp(...)
            # = sum_k w_k * dd_k/dp_i
            # where w_k = exp(-beta*(d_k-d_min)) / sum_exp
            # For pairwise: d_k = |p_i - p_j|/2
            # dd_k/dp_i = (p_i - p_j) / (2 * |p_i - p_j|)
            # dd_k/dp_j = -(p_i - p_j) / (2 * |p_i - p_j|)
            vec = (pts[i] - pts[j]) / (2.0 * d_ij)
            grad[i] += w * vec
            grad[j] -= w * vec
    
    # Wall contributions
    wall_weights = weights[n_pairs:]
    # wall_flat is flattened from (N, 3)
    wall_w = wall_weights.reshape(N, 3)
    # d(wall_dist)/dp_ik = -sign(p_ik)
    for i in range(N):
        for k in range(3):
            w = wall_w[i, k]
            if abs(pts[i, k]) > 1e-15:
                grad[i, k] += w * (-np.sign(pts[i, k]))
    
    return grad

def get_552_init():
    """
    Generate a 5x5x2 grid initialization (exactly 50 points).
    """
    nx, ny, nz = 5, 5, 2
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    
    pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    # Add small jitter to break symmetry
    pts += np.random.normal(0, 0.002, size=(N, 3))
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def get_fcc_init():
    """
    Generate an FCC lattice initialization.
    """
    fcc_pts = []
    cell_size = 1.0 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x0, y0, z0 = i * cell_size, j * cell_size, k * cell_size
                fcc_pts.append([x0, y0, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0 + 0.5 * cell_size, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0, z0 + 0.5 * cell_size])
                fcc_pts.append([x0, y0 + 0.5 * cell_size, z0 + 0.5 * cell_size])
    
    fcc_pts = np.array(fcc_pts)
    fcc_pts -= 0.5
    
    mask = np.all(np.abs(fcc_pts) <= 0.5, axis=1)
    fcc_pts = fcc_pts[mask]
    
    if len(fcc_pts) > N:
        selected = []
        remaining = fcc_pts.copy()
        for _ in range(N):
            if not remaining.size:
                break
            pt = remaining[0]
            selected.append(pt)
            d = np.linalg.norm(remaining - pt, axis=1)
            remaining = remaining[d > 0.15]
        fcc_pts = np.array(selected)
        
    if len(fcc_pts) < N:
        n_pad = N - len(fcc_pts)
        pad = np.random.uniform(-0.45, 0.45, size=(n_pad, 3))
        fcc_pts = np.vstack([fcc_pts, pad])
        
    return fcc_pts

def local_refine_smooth(pts, beta_start=50.0, beta_end=200.0, max_iter=200):
    """
    Two-phase optimization: first with smooth surrogate, then with exact objective.
    """
    x = pts.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    # Phase 1: Smooth surrogate with increasing beta
    beta = beta_start
    n_betas = 5
    for b_idx in range(n_betas):
        beta = beta_start * (beta_end / beta_start) ** (b_idx / max(n_betas - 1, 1))
        try:
            res = minimize(
                neg_soft_radius, x,
                args=(beta,),
                jac=lambda x, b=beta: neg_soft_radius_grad(x, b),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-7, 'maxls': 50}
            )
            x = res.x
        except Exception:
            pass
    
    # Phase 2: Exact objective with Nelder-Mead for final polish
    x_2d = x.reshape(N, 3)
    try:
        res = minimize(
            neg_radius, x_2d,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        x_2d = res.x
    except Exception:
        pass
    
    return x_2d

def construct():
    best_pts = None
    best_r = -np.inf
    
    # 1. 5x5x2 Grid Init (exactly 50 points, good structure)
    grid_init = get_552_init()
    
    # 2. FCC Init
    fcc_init = get_fcc_init()
    
    # 3. Random Init
    rand_init = np.random.uniform(-0.45, 0.45, size=(N, 3))
    
    candidates = [grid_init, fcc_init, rand_init]
    
    # Add more random restarts if time allows
    while time.time() - START_TIME < TIME_LIMIT - 30.0:
        candidates.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))
    
    for i, init_pts in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - 15.0:
            break
        
        init_pts = np.clip(init_pts, -0.49, 0.49)
        
        try:
            refined = local_refine_smooth(init_pts)
            refined = np.clip(refined, -0.499, 0.499)
            r = compute_radius(refined)
            if r > best_r:
                best_r = r
                best_pts = refined.copy()
        except Exception:
            pass
    
    # If no valid points found, return grid
    if best_pts is None:
        nx, ny, nz = 5, 5, 2
        xs = (np.arange(nx) + 0.5) / nx - 0.5
        ys = (np.arange(ny) + 0.5) / ny - 0.5
        zs = (np.arange(nz) + 0.5) / nz - 0.5
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        return grid[:N]
    
    # Final high-precision polish
    if time.time() - START_TIME < TIME_LIMIT - 5.0:
        try:
            res = minimize(
                neg_radius, best_pts,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            r = compute_radius(res.x)
            if r > best_r:
                best_r = r
                best_pts = res.x.copy()
        except Exception:
            pass
    
    return best_pts
