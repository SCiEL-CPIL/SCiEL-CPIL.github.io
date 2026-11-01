"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _score(pos):
    """Largest common radius for a placement (validity assumed)."""
    if pos.shape != (N, 2) or not np.all(np.isfinite(pos)):
        return 0.0
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    min_wall = (0.5 - np.abs(pos)).min()
    return min(0.5 * min_pair, min_wall)


def _dist_matrix(pos):
    """Compute full pairwise distance matrix."""
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    return dist


def _score_from_dist(dist, pos):
    """Score from precomputed distance matrix."""
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    min_wall = (0.5 - np.abs(pos)).min()
    return min(0.5 * min_pair, min_wall)


def _violation(pos, r):
    """Sum of squared violations: pairwise overlap + wall penetration."""
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, 2 * r)
    pair_viol = np.maximum(2 * r - dist, 0.0)
    wall_viol = np.maximum(np.abs(pos) - (0.5 - r), 0.0)
    return np.sum(pair_viol ** 2) + np.sum(wall_viol ** 2)


def _soft_min_score(pos, beta=50.0):
    """Smooth surrogate: soft-min of all pairwise half-distances and wall distances.
    Uses log-sum-exp trick for numerical stability."""
    # Pairwise distances
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    half_dist = 0.5 * dist
    # Wall distances
    wall = 0.5 - np.abs(pos)
    # Combine all into one array
    all_vals = np.concatenate([half_dist.flatten(), wall.flatten()])
    all_vals = all_vals[all_vals < np.inf]
    # Soft min via log-sum-exp
    m = all_vals.min()
    if m < 0:
        return m  # invalid
    log_sum = np.log(np.sum(np.exp(-beta * (all_vals - m))))
    soft_min = m - np.log(log_sum) / beta
    return soft_min


def _soft_min_grad(pos, beta=50.0):
    """Gradient of the soft-min score w.r.t. positions."""
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    half_dist = 0.5 * dist
    wall = 0.5 - np.abs(pos)

    all_vals = np.concatenate([half_dist.flatten(), wall.flatten()])
    all_vals = all_vals[all_vals < np.inf]
    m = all_vals.min()

    # Weights: w_i = exp(-beta*(v_i - m)) / sum(exp(-beta*(v_j - m)))
    log_w = -beta * (all_vals - m)
    log_w_max = log_w.max()
    exp_w = np.exp(log_w - log_w_max)
    w = exp_w / exp_w.sum()

    # Gradient: d(soft_min)/d(pos) = -sum_i w_i * d(v_i)/d(pos)
    # soft_min = m - log(sum exp(-beta*(v_i-m)))/beta
    # d(soft_min)/d(v_k) = -w_k * (-beta) * (1/beta) * ... let me just compute numerically
    # Actually: soft_min = m - (1/beta)*log(sum exp(-beta*(v_i - m)))
    # d(soft_min)/d(v_k) = delta_{k, argmin} - (1/beta) * d/d(v_k) log(sum exp(-beta*(v_i-m)))
    # For k that is the min: delta = 1, and the log term derivative = -beta * w_k
    # So d(soft_min)/d(v_k) = 1 - w_k (for the min element)
    # For others: 0 - w_k * (-beta) * (-1/beta) ... this is getting complex.
    # Let me just use finite differences for the gradient in the polish.
    return None  # Use numerical gradient instead


def _force_step_vec(pos, r, alpha, rep_exp=5):
    """Vectorized force-directed relaxation step."""
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, 1.0)

    # Repulsive force magnitude: (2r - dist)^+ / dist^rep_exp
    overlap_amount = np.maximum(2 * r - dist, 0.0)
    # Avoid division by zero
    safe_dist = np.maximum(dist, 1e-10)
    mag_matrix = overlap_amount / (safe_dist ** rep_exp)
    np.fill_diagonal(mag_matrix, 0.0)

    # Force on i from j: -mag * (pos_i - pos_j) / dist  (direction is along d[i,j])
    # Actually force[i] -= mag[i,j] * d[i,j] / dist[i,j]
    # = -mag[i,j] * (pos_i - pos_j) / dist
    # So force[i] = sum_j mag[i,j] * (pos_j - pos_i) / dist[i,j]
    # = sum_j mag[i,j] * d[j,i] / dist[i,j]  (since d[j,i] = pos_j - pos_i)
    # Note: d[i,j] = pos_i - pos_j, so d[j,i] = -d[i,j]
    # force[i] = sum_j mag[i,j] * (-d[i,j]) / dist[i,j]

    # Vectorized: force[i] = sum_j mag[i,j] * d[j,i] / dist[i,j]
    # = sum_j mag[i,j] * (-d[i,j]) / dist[i,j]
    # = -sum_j mag[i,j] * d[i,j] / dist[i,j]

    unit_d = d / safe_dist[:, :, None]
    force = -np.sum(mag_matrix[:, :, None] * unit_d, axis=1)

    # Weak centering
    force -= 0.03 * pos

    # Adaptive step
    max_move = 0.3 * r * alpha
    step = force * max_move
    step_norm = np.sqrt(np.sum(step * step, axis=1))
    for i in range(n):
        if step_norm[i] > max_move:
            step[i] *= max_move / step_norm[i]

    pos = pos + step
    lim = 0.5 - r
    pos = np.clip(pos, -lim, lim)
    return pos


def _force_layout(r, iters, alpha0, alpha1, rep_exp=5):
    """Build a non-overlapping layout for radius r via force-directed relaxation."""
    np.random.seed(12345)
    pos = np.random.uniform(-0.5 + r, 0.5 - r, size=(N, 2))
    for it in range(iters):
        alpha = alpha0 * (alpha1 / alpha0) ** (it / max(1, iters - 1))
        pos = _force_step_vec(pos, r, alpha, rep_exp)
    return pos


def _sa_incremental(pos, time_budget, r_target=None):
    """Simulated annealing with incremental distance updates.
    Returns (best_pos, best_r)."""
    t_start = time.time()
    best_pos = pos.copy()
    best_r = _score(pos)
    cur_pos = pos.copy()
    cur_r = best_r

    # Precompute distance matrix
    dist_mat = _dist_matrix(cur_pos)

    np.random.seed(98765)
    temp = 0.003
    cool = 0.99995
    step = 0.015
    min_step = 0.0005
    max_step = 0.05

    accept_count = 0
    total_count = 0
    window_size = 500

    # Track the minimum distance and which pair achieves it
    # For incremental update, we need to track min over all pairs

    while time.time() - t_start < time_budget:
        # Check time more frequently
        if time.time() - t_start > time_budget:
            break

        i = np.random.randint(N)
        old = cur_pos[i].copy()
        new_pos_i = old + np.random.normal(0, step, size=2)
        lim = 0.5 - 0.07
        new_pos_i = np.clip(new_pos_i, -lim, lim)

        # Compute new distances from i to all others
        # new_pos_i is (2,), cur_pos is (N, 2).
        # d_new will be (N, 2) due to broadcasting.
        d_new = new_pos_i[None, :] - cur_pos  # shape (N, 2)
        dist_new_i = np.sqrt(np.sum(d_new * d_new, axis=1))  # shape (N,)
        dist_new_i[i] = np.inf  # self distance

        # Update distance matrix row/col i
        old_row = dist_mat[i, :].copy()
        dist_mat[i, :] = dist_new_i
        dist_mat[:, i] = dist_new_i

        # Compute new score
        # Need min over all pairs and min wall
        # Wall for i
        wall_i = (0.5 - np.abs(new_pos_i)).min()
        # Wall for others unchanged
        min_wall_others = (0.5 - np.abs(np.delete(cur_pos, i, axis=0))).min()
        min_wall = min(wall_i, min_wall_others)

        # Min pair: min over all pairs
        # The pairs involving i have new distances; others unchanged
        min_pair_with_i = dist_new_i.min()
        # Min pair not involving i: we need to track this
        # For efficiency, compute from the matrix excluding row/col i
        # This is O(N^2) which is fine for N=50
        sub = dist_mat.copy()
        sub[i, :] = np.inf
        sub[:, i] = np.inf
        min_pair_others = sub.min()
        min_pair = min(min_pair_with_i, min_pair_others)

        new_r = min(0.5 * min_pair, min_wall)

        total_count += 1
        if new_r > cur_r or np.random.rand() < np.exp((new_r - cur_r) / max(temp, 1e-9)):
            cur_r = new_r
            accept_count += 1
            cur_pos[i] = new_pos_i
            if new_r > best_r:
                best_r = new_r
                best_pos = cur_pos.copy()
        else:
            # Revert
            cur_pos[i] = old
            dist_mat[i, :] = old_row
            dist_mat[:, i] = old_row

        temp *= cool
        if temp < 1e-7:
            temp = 1e-7

        # Adapt step size
        if total_count % window_size == 0:
            rate = accept_count / max(1, window_size)
            if rate < 0.15:
                step = max(min_step, step * 0.7)
            elif rate > 0.85:
                step = min(max_step, step * 1.2)
            accept_count = 0

    return best_pos, best_r


def _polish_scipy(pos, time_budget):
    """Polish using scipy minimize on a smooth soft-min objective."""
    t_start = time.time()
    cur_pos = pos.copy()
    cur_r = _score(cur_pos)

    # Use L-BFGS-B to minimize negative soft-min score
    # soft_min_score is a smooth approximation of the min distance
    beta = 80.0

    def neg_soft_min(x):
        p = x.reshape(-1, 2)
        return -_soft_min_score(p, beta)

    # Also try direct violation-based approach with binary search on r
    # First, try to improve with soft-min
    res = minimize(
        neg_soft_min,
        cur_pos.flatten(),
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (2 * N),
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )

    if res.fun < -cur_r + 1e-6:
        new_pos = res.x.reshape(-1, 2)
        new_r = _score(new_pos)
        if new_r > cur_r:
            cur_pos = new_pos
            cur_r = new_r

    # Binary search for max feasible r with violation minimization
    lo = cur_r
    hi = cur_r * 1.08
    for _ in range(4):
        if time.time() - t_start > time_budget:
            break
        mid = (lo + hi) / 2
        res2 = minimize(
            lambda x: _violation(x.reshape(-1, 2), mid),
            cur_pos.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        if res2.success and res2.fun < 1e-10:
            new_pos = res2.x.reshape(-1, 2)
            new_r = _score(new_pos)
            if new_r >= mid - 1e-9:
                lo = mid
                cur_pos = new_pos
                cur_r = new_r
            else:
                hi = mid
        else:
            hi = mid

    return cur_pos, cur_r


def _polish_exact(pos, time_budget):
    """Final polish: try to push each circle slightly outward from constraints."""
    t_start = time.time()
    cur_pos = pos.copy()
    cur_r = _score(cur_pos)

    # Iterative local optimization: for each circle, try small perturbations
    # and keep if it improves the score
    for iteration in range(50):
        if time.time() - t_start > time_budget:
            break
        improved = False
        for i in range(N):
            if time.time() - t_start > time_budget:
                break
            old = cur_pos[i].copy()
            # Try small random perturbations
            for _ in range(20):
                perturb = np.random.normal(0, 0.002, size=2)
                new_pos_i = old + perturb
                lim = 0.5 - 0.07
                new_pos_i = np.clip(new_pos_i, -lim, lim)
                cur_pos[i] = new_pos_i
                new_r = _score(cur_pos)
                if new_r > cur_r + 1e-12:
                    cur_r = new_r
                    improved = True
                    old = new_pos_i.copy()
                    break
            cur_pos[i] = old if not improved else cur_pos[i]
        if not improved:
            break

    return cur_pos, cur_r


def construct():
    t_start = time.time()
    deadline = 130.0  # Safety margin before 150s kill

    # Baseline grid.
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_pos = grid[:N].copy()
    best_r = _score(best_pos)

    # Force-directed initialization with high-exponent repulsion.
    if time.time() - t_start < deadline - 60:
        pos = _force_layout(best_r, 8000, 1.0, 0.02, rep_exp=5)
        r = _score(pos)
        if r > best_r:
            best_r, best_pos = r, pos

    # Simulated annealing with incremental scoring (much faster per iteration).
    if time.time() - t_start < deadline - 20:
        budget = deadline - 20 - (time.time() - t_start)
        pos, r = _sa_incremental(best_pos, budget)
        if r > best_r:
            best_r, best_pos = r, pos

    # Second SA pass with different seed if time allows.
    if time.time() - t_start < deadline - 15:
        budget = deadline - 15 - (time.time() - t_start)
        np.random.seed(42)
        # Slightly perturb to escape local optima
        perturbed = best_pos + np.random.normal(0, 0.005, size=(N, 2))
        lim = 0.5 - 0.07
        perturbed = np.clip(perturbed, -lim, lim)
        pos, r = _sa_incremental(perturbed, budget)
        if r > best_r:
            best_r, best_pos = r, pos

    # Scipy polish pass.
    if time.time() - t_start < deadline - 5:
        budget = deadline - 5 - (time.time() - t_start)
        pos, r = _polish_scipy(best_pos, budget)
        if r > best_r:
            best_r, best_pos = r, pos

    # Final exact local polish.
    if time.time() - t_start < deadline - 1:
        budget = deadline - 1 - (time.time() - t_start)
        pos, r = _polish_exact(best_pos, budget)
        if r > best_r:
            best_r, best_pos = r, pos

    # Final safety: ensure validity (finite, inside square).
    best_pos = np.clip(best_pos, -0.5 + 0.07, 0.5 - 0.07)
    if not np.all(np.isfinite(best_pos)):
        return grid[:N]
    return best_pos
# EVOLVE-BLOCK-END
