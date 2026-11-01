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


def _soft_min_score(pos, beta=80.0):
    """Smooth surrogate: soft-min of all pairwise half-distances and wall distances."""
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    half_dist = 0.5 * dist
    wall = 0.5 - np.abs(pos)
    all_vals = np.concatenate([half_dist.flatten(), wall.flatten()])
    all_vals = all_vals[all_vals < np.inf]
    if len(all_vals) == 0:
        return 0.0
    m = all_vals.min()
    if m < 0:
        return m
    log_sum = np.log(np.sum(np.exp(-beta * (all_vals - m))))
    soft_min = m - np.log(log_sum) / beta
    return soft_min


def _soft_min_grad(pos, beta=80.0):
    """Gradient of the soft-min score w.r.t. positions (vectorized)."""
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    half_dist = 0.5 * dist
    wall = 0.5 - np.abs(pos)

    all_vals = np.concatenate([half_dist.flatten(), wall.flatten()])
    all_vals = all_vals[all_vals < np.inf]
    m = all_vals.min()

    # Weights
    log_w = -beta * (all_vals - m)
    log_w_max = log_w.max()
    exp_w = np.exp(log_w - log_w_max)
    w = exp_w / exp_w.sum()

    # Gradient of soft_min w.r.t. each value v_k:
    # soft_min = m - (1/beta)*log(sum exp(-beta*(v_i - m)))
    # d(soft_min)/d(v_k) = -w_k (for k != argmin) and 1 - w_k (for k = argmin)
    # Actually: d(soft_min)/d(v_k) = -w_k * (-beta) * (1/beta) ... let me derive properly.
    # Let S = sum exp(-beta*(v_i - m)), soft_min = m - log(S)/beta
    # d(soft_min)/d(v_k) = delta_{k,km} - (1/beta) * d(log S)/d(v_k)
    # where km = argmin. d(log S)/d(v_k) = -beta * exp(-beta*(v_k - m))/S = -beta * w_k
    # So d(soft_min)/d(v_k) = delta_{k,km} + w_k
    # Wait, that doesn't seem right either. Let me just use:
    # soft_min ≈ weighted average: sum w_i * v_i (for large beta)
    # d(soft_min)/d(v_k) ≈ w_k
    # But actually the exact derivative is:
    # d(soft_min)/d(v_k) = -w_k (this is the standard result for soft-min)
    # No wait. Let me think again.
    # f = min(v) is subgradient = e_k at argmin.
    # soft_min = m - (1/beta) log(sum exp(-beta(v_i - m)))
    # df/dv_k = -delta_{k,km} + (1/beta) * beta * exp(-beta(v_k-m))/S = -delta_{k,km} + w_k
    # So df/dv_k = w_k - delta_{k,km}
    # For the minimum element: 1 - 1 = 0? No, w_km = exp(0)/S = 1/S which is not 1 unless S=1.
    # Hmm, let me just compute it as: df/dv_k = w_k - (1 if k==km else 0)
    # Actually I think the standard formula is: d(soft_min)/d(v_k) = -w_k
    # Let me verify: if all v_i are equal, soft_min = v, and w_i = 1/n, so d/dv_k = -1/n? That's wrong.
    # I'll just use finite differences or the approximation d(soft_min)/d(v_k) ≈ w_k for the gradient direction.
    # Actually the correct formula: soft_min(v) = -(1/beta) log(sum exp(-beta v_i))
    # d/dv_k = (1/beta) * beta * exp(-beta v_k) / sum exp(-beta v_i) = exp(-beta v_k) / sum exp(-beta v_i) = w_k
    # So d(soft_min)/d(v_k) = w_k. That's clean!
    # And since soft_min is to be MAXIMIZED, we want gradient ascent: pos += lr * grad
    # where grad_pos = sum_k w_k * d(v_k)/d(pos)

    # Now compute d(v_k)/d(pos) for each k
    # v_k are: half_dist[i,j] for i<j (1225 values) and wall[i,c] for i,c (100 values)
    # Total: 1225 + 100 = 1325 values

    grad_pos = np.zeros((n, 2))

    # Pairwise half-distances: half_dist[i,j] = 0.5 * |pos_i - pos_j|
    # d(half_dist[i,j])/d(pos_i) = 0.5 * (pos_i - pos_j) / |pos_i - pos_j|
    # d(half_dist[i,j])/d(pos_j) = 0.5 * (pos_j - pos_i) / |pos_i - pos_j|
    # We need to map w indices back to (i,j) pairs

    # Reconstruct the ordering of all_vals
    # half_dist is (n, n) with diagonal = inf, we flatten it and remove inf
    # wall is (n, 2), we flatten it
    # Let's build the gradient contribution directly

    # For pairwise: for each i < j, contribution to grad_pos[i] += w * 0.5 * (pos_i - pos_j)/dist
    # and grad_pos[j] += w * 0.5 * (pos_j - pos_i)/dist

    # We need the weights in the same order as all_vals
    # all_vals = concat(half_dist.flatten() [with inf removed], wall.flatten())
    # half_dist.flatten() has n*n = 2500 elements, with n=50 diagonal elements = inf
    # So after removing inf, we have 2450 elements? No, we only want i<j.
    # Actually in the code above, we fill diagonal with inf and then remove inf.
    # half_dist is (50,50) = 2500 elements, 50 are inf (diagonal), so 2450 remain.
    # But each pair appears twice (i,j) and (j,i). That's fine for the soft-min.

    # Let me just compute the gradient by iterating over the non-inf entries
    # This is 2450 + 100 = 2550 entries, which is manageable

    # Actually, let me be more careful. The half_dist matrix has inf on diagonal.
    # When we flatten and remove inf, we get 2450 values.
    # wall.flatten() gives 100 values.
    # Total all_vals has 2550 elements.

    # For efficiency, let me compute the gradient contribution from pairwise and wall separately.

    # Pairwise gradient
    # For entry (i,j) in half_dist (i != j), weight is w_pair[i,j]
    # d(half_dist[i,j])/d(pos_i) = 0.5 * (pos_i - pos_j) / dist[i,j]
    # d(half_dist[i,j])/d(pos_j) = 0.5 * (pos_j - pos_i) / dist[i,j]

    # Build weight matrix for pairwise
    # The pairwise weights are the first 2450 entries of w
    # But the mapping from flattened index to (i,j) is tricky with inf removed.
    # Let me just use a different approach: compute the soft-min gradient numerically
    # for the 100-dimensional position vector. With N=50, that's 100 dimensions.
    # Central differences: 200 evaluations of soft_min_score.
    # Each evaluation is O(N^2) = O(2500). Total: 500,000 ops. Fast enough.

    # Actually, let me just do the analytic gradient properly.
    # I'll reconstruct the index mapping.

    # half_dist indices: for i in range(n), for j in range(n):
    #   if i != j: this is a valid entry
    # The flattened half_dist (before removing inf) has indices:
    #   flat_idx = i * n + j
    # After removing the n diagonal entries (where i==j), the remaining 2450 entries
    # are in row-major order skipping the diagonal.

    # This is getting complex. Let me just use the simpler approach:
    # Compute gradient via the formula d(soft_min)/d(pos_k) = sum_j w_j * d(v_j)/d(pos_k)
    # by explicitly computing d(v_j)/d(pos) for each j.

    # For performance, let me just use a semi-analytic approach:
    # The soft-min is dominated by the minimum few values. For gradient purposes,
    # we can approximate by just using the argmin element's gradient.

    # Find the argmin of all_vals
    km = np.argmin(all_vals)

    # Determine if km is in the pairwise part or wall part
    n_pair = (n * n - n)  # 2450
    if km < n_pair:
        # It's a pairwise entry. Need to find which (i,j)
        # Map flat index (skipping diagonal) to (i,j)
        # Count: for row i, there are (n-1) entries (all except column i)
        # Cumulative: row 0 has entries at cols 1..n-1 (n-1 entries)
        # row 1 has entries at cols 0, 2..n-1 (n-1 entries)
        # etc.
        # Simple approach: iterate
        count = 0
        found_i, found_j = -1, -1
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if count == km:
                    found_i, found_j = i, j
                    break
                count += 1
            if found_i >= 0:
                break
        # Gradient contribution
        diff = pos[found_i] - pos[found_j]
        dist_ij = np.linalg.norm(diff)
        if dist_ij > 1e-12:
            grad_pos[found_i] += 0.5 * diff / dist_ij
            grad_pos[found_j] -= 0.5 * diff / dist_ij
    else:
        # It's a wall entry: index km - n_pair in wall.flatten()
        wall_idx = km - n_pair
        i = wall_idx // 2
        c = wall_idx % 2
        # wall[i,c] = 0.5 - |pos[i,c]|
        # d(wall[i,c])/d(pos[i,c]) = -sign(pos[i,c])
        s = np.sign(pos[i, c])
        if s == 0:
            s = 1.0
        grad_pos[i, c] += -s

    return grad_pos


def _violation(pos, r):
    """Sum of squared violations: pairwise overlap + wall penetration."""
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, 2 * r)
    pair_viol = np.maximum(2 * r - dist, 0.0)
    wall_viol = np.maximum(np.abs(pos) - (0.5 - r), 0.0)
    return np.sum(pair_viol ** 2) + np.sum(wall_viol ** 2)


def _violation_grad(pos, r):
    """Gradient of violation w.r.t. positions."""
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, 1.0)

    # Pairwise: viol_ij = max(2r - dist_ij, 0)^2
    # d(viol_ij)/d(pos_i) = -2*(2r-dist) * d(dist)/d(pos_i) = -2*(2r-dist) * (pos_i-pos_j)/dist
    overlap = np.maximum(2 * r - dist, 0.0)
    safe_dist = np.maximum(dist, 1e-10)
    # grad_pair[i] = sum_j -2*overlap[i,j] * d[i,j] / dist[i,j]
    unit_d = d / safe_dist[:, :, None]
    grad_pair = -2.0 * np.sum(overlap[:, :, None] * unit_d, axis=1)

    # Wall: viol_ic = max(|pos_ic| - (0.5-r), 0)^2
    # d/d(pos_ic) = 2*(|pos_ic|-(0.5-r)) * sign(pos_ic)  when |pos_ic| > 0.5-r
    wall_excess = np.maximum(np.abs(pos) - (0.5 - r), 0.0)
    grad_wall = 2.0 * wall_excess * np.sign(pos)

    return grad_pair + grad_wall


def _projected_grad_descent(pos, r, lr, n_iter, time_limit):
    """Projected gradient descent to find feasible placement for radius r."""
    t_start = time.time()
    cur = pos.copy()
    best = cur.copy()
    best_viol = _violation(best, r)

    for it in range(n_iter):
        if time.time() - t_start > time_limit:
            break
        g = _violation_grad(cur, r)
        cur = cur - lr * g
        # Project to square
        lim = 0.5 - r
        cur = np.clip(cur, -lim, lim)
        v = _violation(cur, r)
        if v < best_viol:
            best_viol = v
            best = cur.copy()
        # Adaptive learning rate
        if v > best_viol * 1.5:
            lr *= 0.8
        elif v < best_viol * 0.5:
            lr = min(lr * 1.2, 0.01)

    return best, best_viol


def _binary_search_radius(pos, r_lo, r_hi, time_budget):
    """Binary search for max feasible radius using projected gradient descent."""
    t_start = time.time()
    best_pos = pos.copy()
    best_r = r_lo

    for _ in range(8):
        if time.time() - t_start > time_budget:
            break
        r_mid = (r_lo + r_hi) / 2
        # Try to find feasible placement for r_mid
        feasible_pos, viol = _projected_grad_descent(
            best_pos, r_mid, lr=0.005, n_iter=2000,
            time_limit=time_budget - (time.time() - t_start)
        )
        if viol < 1e-10:
            best_r = r_mid
            best_pos = feasible_pos
            r_lo = r_mid
        else:
            r_hi = r_mid

    return best_pos, best_r


def _sa_incremental(pos, time_budget, seed=98765):
    """Simulated annealing with incremental distance updates."""
    t_start = time.time()
    best_pos = pos.copy()
    best_r = _score(pos)
    cur_pos = pos.copy()
    cur_r = best_r

    np.random.seed(seed)
    temp = 0.003
    cool = 0.99995
    step = 0.015
    min_step = 0.0005
    max_step = 0.05

    accept_count = 0
    total_count = 0
    window_size = 500

    while time.time() - t_start < time_budget:
        i = np.random.randint(N)
        old = cur_pos[i].copy()
        new_pos_i = old + np.random.normal(0, step, size=2)
        lim = 0.5 - 0.07
        new_pos_i = np.clip(new_pos_i, -lim, lim)

        # Compute new score incrementally
        d_new = new_pos_i[None, :] - cur_pos
        dist_new_i = np.sqrt(np.sum(d_new * d_new, axis=1))
        dist_new_i[i] = np.inf

        # Save old distances involving i
        old_d_i = cur_pos[i][None, :] - cur_pos
        old_dist_i = np.sqrt(np.sum(old_d_i * old_d_i, axis=1))
        old_dist_i[i] = np.inf

        # New min pair involving i
        min_pair_with_i_new = dist_new_i.min()
        min_pair_with_i_old = old_dist_i.min()

        # Wall for i
        wall_i_new = (0.5 - np.abs(new_pos_i)).min()
        wall_i_old = (0.5 - np.abs(old)).min()

        # Min wall others (unchanged)
        min_wall_others = (0.5 - np.abs(np.delete(cur_pos, i, axis=0))).min()

        # Min pair not involving i
        # We need this efficiently. Let's just compute it.
        # For N=50, this is O(N^2) = 2500 ops per iteration, which is fine.
        all_d = cur_pos[:, None, :] - cur_pos[None, :, :]
        all_dist = np.sqrt(np.sum(all_d * all_d, axis=2))
        all_dist[i, :] = np.inf
        all_dist[:, i] = np.inf
        min_pair_others = all_dist.min()

        min_pair_new = min(min_pair_with_i_new, min_pair_others)
        min_pair_old = min(min_pair_with_i_old, min_pair_others)

        min_wall_new = min(wall_i_new, min_wall_others)
        min_wall_old = min(wall_i_old, min_wall_others)

        new_r = min(0.5 * min_pair_new, min_wall_new)
        old_r = min(0.5 * min_pair_old, min_wall_old)

        total_count += 1
        if new_r > cur_r or np.random.rand() < np.exp((new_r - cur_r) / max(temp, 1e-9)):
            cur_r = new_r
            accept_count += 1
            cur_pos[i] = new_pos_i
            if new_r > best_r:
                best_r = new_r
                best_pos = cur_pos.copy()
        else:
            cur_pos[i] = old

        temp *= cool
        if temp < 1e-7:
            temp = 1e-7

        if total_count % window_size == 0:
            rate = accept_count / max(1, window_size)
            if rate < 0.15:
                step = max(min_step, step * 0.7)
            elif rate > 0.85:
                step = min(max_step, step * 1.2)
            accept_count = 0

    return best_pos, best_r


def _lbfgs_polish(pos, time_budget):
    """Polish using L-BFGS-B on soft-min objective."""
    t_start = time.time()
    cur_pos = pos.copy()
    cur_r = _score(cur_pos)
    beta = 80.0

    def neg_soft_min(x):
        p = x.reshape(-1, 2)
        return -_soft_min_score(p, beta)

    def neg_soft_min_grad(x):
        p = x.reshape(-1, 2)
        g = _soft_min_grad(p, beta)
        return -g.flatten()

    remaining = time_budget - (time.time() - t_start)
    if remaining < 2:
        return cur_pos, cur_r

    res = minimize(
        neg_soft_min,
        cur_pos.flatten(),
        jac=neg_soft_min_grad,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (2 * N),
        options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )

    if res.fun < -cur_r + 1e-6:
        new_pos = res.x.reshape(-1, 2)
        new_r = _score(new_pos)
        if new_r > cur_r:
            cur_pos = new_pos
            cur_r = new_r

    # Binary search push
    remaining = time_budget - (time.time() - t_start)
    if remaining > 5:
        pos2, r2 = _binary_search_radius(cur_pos, cur_r, cur_r * 1.05, remaining - 1)
        if r2 > cur_r:
            cur_pos = pos2
            cur_r = r2

    return cur_pos, cur_r


def _force_layout(r, iters, alpha0, alpha1, rep_exp=5):
    """Build a non-overlapping layout for radius r via force-directed relaxation."""
    np.random.seed(12345)
    pos = np.random.uniform(-0.5 + r, 0.5 - r, size=(N, 2))
    for it in range(iters):
        alpha = alpha0 * (alpha1 / alpha0) ** (it / max(1, iters - 1))
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, 1.0)
        overlap_amount = np.maximum(2 * r - dist, 0.0)
        safe_dist = np.maximum(dist, 1e-10)
        mag_matrix = overlap_amount / (safe_dist ** rep_exp)
        np.fill_diagonal(mag_matrix, 0.0)
        unit_d = d / safe_dist[:, :, None]
        force = -np.sum(mag_matrix[:, :, None] * unit_d, axis=1)
        force -= 0.03 * pos
        max_move = 0.3 * r * alpha
        step = force * max_move
        step_norm = np.sqrt(np.sum(step * step, axis=1))
        for i in range(N):
            if step_norm[i] > max_move:
                step[i] *= max_move / step_norm[i]
        pos = pos + step
        lim = 0.5 - r
        pos = np.clip(pos, -lim, lim)
    return pos


def construct():
    t_start = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Baseline grid.
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_pos = grid[:N].copy()
    best_r = _score(best_pos)

    # Force-directed initialization.
    if time.time() - t_start < deadline - 80:
        pos = _force_layout(best_r, 5000, 1.0, 0.02, rep_exp=5)
        r = _score(pos)
        if r > best_r:
            best_r, best_pos = r, pos

    # Simulated annealing pass 1.
    if time.time() - t_start < deadline - 30:
        budget = deadline - 30 - (time.time() - t_start)
        pos, r = _sa_incremental(best_pos, budget, seed=98765)
        if r > best_r:
            best_r, best_pos = r, pos

    # SA pass 2 with different seed.
    if time.time() - t_start < deadline - 20:
        budget = deadline - 20 - (time.time() - t_start)
        np.random.seed(42)
        perturbed = best_pos + np.random.normal(0, 0.005, size=(N, 2))
        lim = 0.5 - 0.07
        perturbed = np.clip(perturbed, -lim, lim)
        pos, r = _sa_incremental(perturbed, budget, seed=11111)
        if r > best_r:
            best_r, best_pos = r, pos

    # L-BFGS polish + binary search push.
    if time.time() - t_start < deadline - 5:
        budget = deadline - 5 - (time.time() - t_start)
        pos, r = _lbfgs_polish(best_pos, budget)
        if r > best_r:
            best_r, best_pos = r, pos

    # Final safety: ensure validity.
    best_pos = np.clip(best_pos, -0.5 + 0.06, 0.5 - 0.06)
    if not np.all(np.isfinite(best_pos)):
        return grid[:N]
    return best_pos
# EVOLVE-BLOCK-END
