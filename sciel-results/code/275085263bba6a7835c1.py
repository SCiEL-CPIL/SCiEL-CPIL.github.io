"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
import time

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    r = min(min_wall, min_pair / 2.0)
    return r

def _generate_hex_grid():
    """Generate a hexagonal lattice packing as initial configuration."""
    best_centers = None
    best_r = 0.0
    for n_rows in range(10, 25):
        n_cols = int(np.ceil(N / n_rows))
        r1 = 1.0 / (2.0 * n_cols)
        r2 = 1.0 / ((n_rows - 1) * np.sqrt(3) + 2)
        r = min(r1, r2)
        s = 2 * r
        centers = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                x = -0.5 + r + j * s + (i % 2) * s / 2
                y = -0.5 + r + i * s * np.sqrt(3) / 2
                centers.append([x, y])
            if len(centers) >= N:
                break
        centers = np.array(centers[:N])
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            xs = centers[:, 0]; ys = centers[:, 1]
            cx = (xs.max() + xs.min()) / 2
            cy = (ys.max() + ys.min()) / 2
            centers[:, 0] -= cx
            centers[:, 1] -= cy
            if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
                continue
        r_actual = _compute_radius(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers.copy()
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    return best_centers

def _soft_min(x, beta=50.0):
    """Soft minimum via log-sum-exp for numerical stability."""
    x_min = np.min(x)
    return x_min - np.log(np.sum(np.exp(-(x - x_min) * beta))) / beta

def _gradient_step(centers, t, beta=50.0):
    """
    One gradient step maximizing t (the common radius) with respect to centers,
    using a smooth surrogate (soft-min) of the active constraints.
    Returns (new_centers, current_r).
    """
    # Pairwise distances (19900 for N=200)
    d = pdist(centers)
    # Wall distances per point
    wall = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)

    # We want to maximize t = min(min_pair/2, min_wall).
    # Surrogate: maximize  soft_min(d)/2  and  soft_min(wall) jointly.
    # Equivalently minimize  -soft_min(d)/2 - soft_min(wall)  (plus a small
    # balance term so we push the weaker constraint harder).
    #
    # Gradients:
    #   d(soft_min(d))/d c_i  =  sum_j w_j * (c_i - c_j)/d_ij   where w_j from soft-min weights
    #   d(soft_min(wall))/d c_i  from wall terms.
    #
    # Soft-min weight for a value x_k: w_k = exp(-beta*(x_k - x_min)) / sum exp(-beta*(x_k - x_min))
    #
    # We build the full pairwise distance matrix to get per-pair weights.
    D = squareform(d)
    np.fill_diagonal(D, np.inf)

    # Pair weights (soft-min over all pairs)
    dmin = np.min(D)
    wd = np.exp(-beta * (D - dmin))
    wd /= np.sum(wd)

    # Wall weights (soft-min over 2N wall terms)
    # wall has N entries (min of 4 per point already taken). Use per-point wall as the value.
    wmin = np.min(wall)
    ww = np.exp(-beta * (wall - wmin))
    ww /= np.sum(ww)

    # Pair gradient: for each i, sum over j of w_{ij} * (c_i - c_j)/d_ij
    # We need 1/d_ij (safe).
    inv_d = np.zeros_like(D)
    mask = D < np.inf
    inv_d[mask] = 1.0 / D[mask]

    # grad_pair[i] = sum_j w_{ij} * (c_i - c_j) * inv_d[i,j]
    # = c_i * sum_j w_{ij} inv_d[i,j]  -  sum_j w_{ij} inv_d[i,j] * c_j
    A = wd * inv_d  # (N, N)
    row_sum = A.sum(axis=1)
    # sum_j A[i,j] c_j
    Ac = A @ centers
    grad_pair = centers * row_sum[:, None] - Ac

    # Wall gradient: wall_i = min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5).
    # d(soft_min(wall))/d c_i = ww_i * d wall_i / d c_i
    # d wall_i / d c_i: for x: +1 if x<0 else -1 (since wall from x is 0.5-|x|, d/dx = -sign(x))
    # Actually wall_i = min over the four; but we approximate using the dominant wall.
    # Simpler: use subgradient of min(0.5-|x|, 0.5-|y|) = the active one.
    # d(0.5 - |x|)/dx = -sign(x)  (for x != 0)
    # We use the per-point wall = min(0.5-|x|, 0.5-|y|).
    # Gradient of that min w.r.t x: if 0.5-|x| <= 0.5-|y| then -sign(x) else 0.
    gx = np.where((0.5 - np.abs(centers[:, 0])) <= (0.5 - np.abs(centers[:, 1])),
                  -np.sign(centers[:, 0]), 0.0)
    gy = np.where((0.5 - np.abs(centers[:, 1])) <= (0.5 - np.abs(centers[:, 0])),
                  -np.sign(centers[:, 1]), 0.0)
    # handle sign(0) -> treat as 0 (no push)
    gx[np.abs(centers[:, 0]) < 1e-12] = 0.0
    gy[np.abs(centers[:, 1]) < 1e-12] = 0.0
    grad_wall = ww[:, None] * np.stack([gx, gy], axis=1)

    # Combined ascent direction: we want to increase both soft_min(d)/2 and soft_min(wall).
    # Objective to maximize: F = soft_min(d)/2 + soft_min(wall)
    # dF/dc = 0.5 * grad_pair + grad_wall
    grad = 0.5 * grad_pair + grad_wall

    # Normalize gradient to unit-ish scale to make lr meaningful.
    gnorm = np.linalg.norm(grad)
    if gnorm > 1e-12:
        grad = grad / gnorm

    return grad

def _gradient_refine(centers, max_iter=3000, lr=0.004, beta=60.0):
    """
    Gradient-based refinement of the smooth surrogate.
    Alternates: move centers along ascent direction, clip to square, accept if r improves.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    t = best_r
    current_lr = lr
    no_improve = 0

    for it in range(max_iter):
        if _time_left() < 8.0:
            break

        grad = _gradient_step(centers, t, beta)
        trial = centers + current_lr * grad
        # clip to square (slightly inside to keep feasibility margin)
        trial[:, 0] = np.clip(trial[:, 0], -0.5, 0.5)
        trial[:, 1] = np.clip(trial[:, 1], -0.5, 0.5)

        r_trial = _compute_radius(trial)
        if r_trial > best_r + 1e-13:
            best_r = r_trial
            best_centers = trial.copy()
            centers = trial.copy()
            t = best_r
            no_improve = 0
            current_lr = max(current_lr * 1.02, 0.02)  # small growth on success
        else:
            no_improve += 1
            current_lr *= 0.97
            if no_improve > 200 and current_lr < 1e-5:
                break

    return best_centers, best_r

def _local_search(centers, max_iter=6000, step_scale=0.01):
    """Local search using gradient-free perturbation of most-constrained points."""
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    tree = cKDTree(centers)
    for it in range(max_iter):
        if _time_left() < 6.0:
            break
        dists, idx = tree.query(centers, k=3)
        nn_dists = dists[:, 1]
        wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
        n_perturb = 2 + (it % 2)
        candidates = np.concatenate([
            np.argsort(nn_dists)[:n_perturb],
            np.argsort(wall_dists)[:n_perturb]
        ])
        candidates = np.unique(candidates[:n_perturb])
        improvements = 0
        for _ in range(8):
            trial = centers.copy()
            for idx in candidates:
                angle = np.random.uniform(0, 2 * np.pi)
                step = step_scale * (0.5 + np.random.rand())
                trial[idx, 0] += step * np.cos(angle)
                trial[idx, 1] += step * np.sin(angle)
            trial[:, 0] = np.clip(trial[:, 0], -0.5, 0.5)
            trial[:, 1] = np.clip(trial[:, 1], -0.5, 0.5)
            r_trial = _compute_radius(trial)
            if r_trial > best_r + 1e-12:
                best_r = r_trial
                best_centers = trial.copy()
                centers = trial.copy()
                tree = cKDTree(centers)
                step_scale *= 0.95
                improvements += 1
        if improvements == 0:
            step_scale *= 0.99
            if step_scale < 1e-6:
                break
    return best_centers, best_r

def _polish(centers, max_iter=4000, lr=0.002, beta=120.0):
    """
    High-precision polish: smaller lr, larger beta (sharper soft-min),
    to squeeze out the last bits of radius near the optimum.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    t = best_r
    current_lr = lr
    no_improve = 0
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        grad = _gradient_step(centers, t, beta)
        trial = centers + current_lr * grad
        trial[:, 0] = np.clip(trial[:, 0], -0.5, 0.5)
        trial[:, 1] = np.clip(trial[:, 1], -0.5, 0.5)
        r_trial = _compute_radius(trial)
        if r_trial > best_r + 1e-14:
            best_r = r_trial
            best_centers = trial.copy()
            centers = trial.copy()
            t = best_r
            no_improve = 0
            current_lr = max(current_lr * 1.01, 0.005)
        else:
            no_improve += 1
            current_lr *= 0.98
            if no_improve > 300 and current_lr < 1e-6:
                break
    return best_centers, best_r

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    centers = _generate_hex_grid()
    r = _compute_radius(centers)

    # Phase 1: gradient-based refinement on smooth surrogate (bulk of improvement)
    if _time_left() > 25.0:
        centers, r = _gradient_refine(centers, max_iter=3000, lr=0.004, beta=60.0)

    # Phase 2: gradient-free local search to escape any flat spots
    if _time_left() > 15.0:
        centers, r = _local_search(centers, max_iter=4000, step_scale=0.008)

    # Phase 3: high-precision polish (sharper soft-min, smaller steps)
    if _time_left() > 5.0:
        centers, r = _polish(centers, max_iter=4000, lr=0.002, beta=120.0)

    # Final verification and clipping
    centers = np.clip(centers, -0.5, 0.5)
    return centers
# EVOLVE-BLOCK-END
