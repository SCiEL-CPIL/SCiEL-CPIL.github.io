"""Optimized packing: smooth soft-min surrogate + L-BFGS-B + exact polish."""
import numpy as np
import time
import math

N = 200

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # 10s safety margin before the 150s kill


def _time_left():
    return _DEADLINE - time.time()


def _exact_radius(pos):
    """Exact packing radius: min(min_pair/2, min_boundary)."""
    n = len(pos)
    if n == 0:
        return 0.0
    dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0.0:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = math.sqrt(float(np.min(dist_sq)))
    return min(min_pair * 0.5, min_boundary)


def _soft_min_objective(pos, beta):
    """
    Smooth surrogate: maximize r = min(min_pair/2, min_boundary).
    We minimize f = -log-sum-exp(-beta * [pair_dists/2, boundary_dists]) / beta
    which approximates -min(...) as beta -> inf, so maximizing r = minimizing f.
    Returns (f, grad).
    """
    n = len(pos)
    # Boundary distances
    b = 0.5 - np.abs(pos)
    bd = np.min(b, axis=1)  # shape (n,)

    # Pairwise distances / 2
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    pd = 0.5 * np.sqrt(dist_sq)  # shape (n, n), off-diagonal

    # Collect all "radius candidates": n boundary values + n*(n-1)/2 pair values
    # To keep it manageable, we work with the full (n, n) matrix (diagonal is inf).
    # log-sum-exp over all entries of pd and bd.

    # Stack: we want LSE of (-beta * x) for x in {pd (off-diag), bd}.
    # Use a numerically stable log-sum-exp.

    # For gradient, we need d/dpos of the LSE.
    # Let S = sum_k exp(-beta * x_k), where x_k are the candidate radii.
    # f = -log(S)/beta
    # df/dpos_i = -1/beta * d log S / dpos_i
    #           = -1/beta * (1/S) * dS/dpos_i
    #           = -1/beta * (1/S) * sum_k exp(-beta*x_k) * (-beta) * dx_k/dpos_i
    #           = sum_k [exp(-beta*x_k)/S] * dx_k/dpos_i
    # So the gradient is a weighted sum of dx_k/dpos_i with weights w_k = exp(-beta*x_k)/S.

    # Compute weights for boundary terms: w_bd_j = exp(-beta*bd_j)/S
    # Compute weights for pair terms: w_pd_ij = exp(-beta*pd_ij)/S

    # Numerical stability: subtract max of (-beta*x) = -beta*min(x) ... but min(x) is what we approximate.
    # Let m = min of all x_k (the true min radius). Then -beta*x_k <= beta*m.
    # We'll compute with a shift.

    # Gather all x values for the shift computation.
    # To avoid huge memory, we compute in parts.

    # Shift: use a reference value. Let ref = min of all candidate radii.
    # Actually, for stability, shift by the minimum of (-beta * x) = -beta * max(x)? No.
    # LSE of a_k = -beta * x_k. Shift by max(a_k) = -beta * min(x_k).
    # Let m_x = min over all x_k. Then a_k - a_max = -beta*x_k + beta*m_x = beta*(m_x - x_k) <= 0.
    # exp(a_k - a_max) <= 1. Good.

    # Compute m_x = min(min(bd), min(pd off-diag)).
    m_x = min(float(np.min(bd)), float(np.min(pd)))

    # Compute shifted exponents
    # Boundary: exp(beta*(m_x - bd_j))
    exp_bd = np.exp(beta * (m_x - bd))  # shape (n,)
    # Pair: exp(beta*(m_x - pd_ij)) for off-diagonal
    exp_pd = np.exp(beta * (m_x - pd))  # shape (n, n)
    np.fill_diagonal(exp_pd, 0.0)

    S_shifted = float(np.sum(exp_bd)) + float(np.sum(exp_pd))
    if S_shifted <= 0.0:
        S_shifted = 1e-300

    # f = -log(S)/beta, where S = sum exp(-beta*x_k) = exp(-beta*m_x) * S_shifted
    # log S = -beta*m_x + log(S_shifted)
    # f = -(-beta*m_x + log(S_shifted))/beta = m_x - log(S_shifted)/beta
    f = m_x - math.log(S_shifted) / beta

    # Weights: w_k = exp(-beta*x_k)/S = exp(beta*(m_x - x_k)) / S_shifted
    w_bd = exp_bd / S_shifted  # shape (n,)
    w_pd = exp_pd / S_shifted  # shape (n, n)

    # Gradient: grad = sum_k w_k * dx_k/dpos
    # Boundary: x_j = min(0.5 - |pos_jx|, 0.5 - |pos_jy|)
    # dx_j/dpos_jx = -sign(pos_jx) if (0.5-|pos_jx|) is the min, else 0
    # dx_j/dpos_jy = -sign(pos_jy) if (0.5-|pos_jy|) is the min, else 0
    grad = np.zeros_like(pos)
    # For each point, determine which boundary constraint is active
    bx = 0.5 - np.abs(pos[:, 0])
    by = 0.5 - np.abs(pos[:, 1])
    active_x = bx <= by
    active_y = ~active_x
    # dx_j/dpos_jx = -sign(pos_jx) * active_x
    grad[:, 0] += w_bd * (-np.sign(pos[:, 0])) * active_x
    # dx_j/dpos_jy = -sign(pos_jy) * active_y
    grad[:, 1] += w_bd * (-np.sign(pos[:, 1])) * active_y

    # Pair: x_ij = 0.5 * ||pos_i - pos_j||
    # dx_ij/dpos_i = 0.5 * (pos_i - pos_j)/||pos_i - pos_j|| = 0.5 * diff_ij / dist_ij
    # dx_ij/dpos_j = 0.5 * (pos_j - pos_i)/||pos_j - pos_i|| = -0.5 * diff_ij / dist_ij
    # Note: w_pd is symmetric (w_pd_ij = w_pd_ji), so we can compute the contribution.
    # dist_ij = 2 * pd_ij (since pd_ij = 0.5*dist)
    # dx_ij/dpos_i = diff_ij / (2 * dist_ij) = diff_ij / (4 * pd_ij)
    # Let's use: for each pair (i,j), contribution to grad[i] += w_pd_ij * diff_ij/(4*pd_ij)
    # and grad[j] += w_pd_ij * (-diff_ij/(4*pd_ij)) = w_pd_ji * diff_ji/(4*pd_ji)
    # Since w_pd is symmetric and diff_ji = -diff_ij, pd_ji = pd_ij, this is consistent.

    # To compute efficiently:
    # grad_pair[i] = sum_j w_pd_ij * diff_ij / (4 * pd_ij)
    # = sum_j w_pd_ij * (pos_i - pos_j) / (4 * pd_ij)
    # Note pd_ij = 0.5 * ||pos_i - pos_j||, so 4*pd_ij = 2*||pos_i-pos_j||
    # = sum_j w_pd_ij * (pos_i - pos_j) / (2 * ||pos_i - pos_j||)

    # Avoid division by zero on diagonal (already zeroed in w_pd).
    # Compute unit vectors
    dist_mat = 2.0 * pd  # shape (n, n)
    np.fill_diagonal(dist_mat, 1.0)  # avoid div by zero
    unit = diff / dist_mat[:, :, np.newaxis]  # shape (n, n, 2)
    # grad_pair[i] = sum_j w_pd_ij * unit_ij / 2
    # = 0.5 * (w_pd * unit).sum(axis=1)
    # w_pd is (n, n), unit is (n, n, 2)
    grad_pair = 0.5 * np.sum(w_pd[:, :, np.newaxis] * unit, axis=1)  # shape (n, 2)
    grad += grad_pair

    # f is the objective to minimize; gradient is df/dpos.
    return f, grad


def _hex_initial(spacing):
    """Generate hexagonal lattice points inside [-0.5, 0.5]^2."""
    dy = spacing * math.sqrt(3.0) / 2.0
    max_coord = 0.5
    points = []
    row = 0
    y = -max_coord
    while y <= max_coord:
        x_offset = 0.0 if (row % 2 == 0) else spacing / 2.0
        x = -max_coord + x_offset
        while x <= max_coord:
            if -max_coord <= x <= max_coord and -max_coord <= y <= max_coord:
                points.append((x, y))
            x += spacing
        y += dy
        row += 1
    if not points:
        return np.zeros((0, 2))
    arr = np.array(points, dtype=float)
    return arr


def _select_subset(candidates, n_select, seed=0):
    """Select n_select points from candidates via greedy max-min distance."""
    m = len(candidates)
    if m == 0:
        return np.zeros((0, 2))
    if m <= n_select:
        return candidates.copy()

    rng = np.random.RandomState(seed)
    # Start with a random point
    boundary = np.min(0.5 - np.abs(candidates), axis=1)
    score = boundary.copy()
    sel_idx = []

    for _ in range(n_select):
        if _time_left() < 15.0:
            break
        best = int(np.argmax(score))
        sel_idx.append(best)
        p = candidates[best]
        diff = candidates - p
        d = np.sqrt(np.sum(diff * diff, axis=1))
        d[best] = np.inf
        score = np.minimum(score, d)

    sel = candidates[sel_idx]
    return sel


def _l_bfgs_optimize(pos0, beta_schedule, time_budget):
    """
    Run L-BFGS-B on the soft-min objective with an increasing beta schedule.
    """
    from scipy.optimize import minimize

    pos = np.array(pos0, dtype=float, copy=True)
    pos = np.clip(pos, -0.4999, 0.4999)
    best_pos = pos.copy()
    best_r = _exact_radius(pos)
    start_time = time.time()

    for beta in beta_schedule:
        if _time_left() < 20.0:
            break
        if (time.time() - start_time) > time_budget:
            break

        def obj_and_grad(x):
            x = x.reshape(-1, 2)
            x = np.clip(x, -0.4999, 0.4999)
            f, g = _soft_min_objective(x, beta)
            return f, g.reshape(-1)

        x0 = pos.reshape(-1)
        try:
            res = minimize(
                obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8, 'maxls': 50},
            )
            if res.success or res.fun < 1e10:
                new_pos = res.x.reshape(-1, 2)
                new_pos = np.clip(new_pos, -0.4999, 0.4999)
                new_r = _exact_radius(new_pos)
                if new_r > best_r:
                    best_r = new_r
                    best_pos = new_pos.copy()
                pos = new_pos
        except Exception:
            pass

    return best_pos, best_r


def _exact_polish(pos, max_iter=2000):
    """
    High-precision coordinate ascent on the exact objective.
    Iteratively move the bottleneck points to increase the min radius.
    """
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    best_r = _exact_radius(pos)
    best_pos = pos.copy()

    for it in range(max_iter):
        if _time_left() < 10.0:
            break

        r = _exact_radius(pos)
        if r <= 0:
            break

        # Find bottleneck
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        min_pair_dist = math.sqrt(dist_sq[min_idx])

        dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
        min_bnd_idx = int(np.argmin(dist_boundary))
        min_bnd_dist = dist_boundary[min_bnd_idx]

        made_progress = False

        if min_pair_dist / 2.0 <= min_bnd_dist:
            # Pair is bottleneck
            i, j = min_idx
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.random.randn(2) * 1e-8
                dist = np.linalg.norm(d)
            d_norm = d / dist
            # Move i and j apart
            step = 0.01 * (dist - 2 * r)
            if step > 1e-12:
                pos[i] += d_norm * step / 2
                pos[j] -= d_norm * step / 2
                made_progress = True
        else:
            # Boundary is bottleneck
            p = pos[min_bnd_idx]
            # Move toward center
            step = 0.01 * (min_bnd_dist - r)
            if step > 1e-12:
                norm_p = np.linalg.norm(p)
                if norm_p > 1e-10:
                    pos[min_bnd_idx] = p + (-p / norm_p) * step
                else:
                    pos[min_bnd_idx] += np.random.randn(2) * step
                made_progress = True

        pos = np.clip(pos, -0.4999, 0.4999)

        new_r = _exact_radius(pos)
        if new_r > best_r:
            best_r = new_r
            best_pos = pos.copy()

        if not made_progress:
            # Small random perturbation to escape
            pos += np.random.randn(n, 2) * 0.0005
            pos = np.clip(pos, -0.4999, 0.4999)

    return best_pos, best_r


def construct():
    best_pos = None
    best_r = 0.0

    # Beta schedule: start low (smooth), increase to sharpen
    beta_schedule = [20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1500.0]

    # Generate diverse initializations
    initializations = []

    # Hexagonal lattices at various spacings
    spacings = [0.035, 0.032, 0.030, 0.028, 0.025, 0.022]
    for sp in spacings:
        if _time_left() < 30.0:
            break
        try:
            candidates = _hex_initial(sp)
            if len(candidates) >= N:
                sel = _select_subset(candidates, N, seed=int(sp * 1000))
                if len(sel) == N:
                    initializations.append(sel)
        except Exception:
            continue

    # Add a few randomized perturbations of the best hex
    if initializations:
        base = initializations[0]
        for k in range(3):
            if _time_left() < 25.0:
                break
            perturbed = base + np.random.randn(N, 2) * 0.005
            perturbed = np.clip(perturbed, -0.4999, 0.4999)
            initializations.append(perturbed)

    # Fallback: uniform grid
    if not initializations:
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        initializations.append(grid)

    # Optimize each initialization
    for init in initializations:
        if _time_left() < 25.0:
            break
        try:
            opt_pos, opt_r = _l_bfgs_optimize(
                init,
                beta_schedule,
                time_budget=_time_left() - 15.0,
            )
            if opt_r > best_r:
                best_r = opt_r
                best_pos = opt_pos.copy()
        except Exception:
            continue

    # Final exact polish
    if best_pos is not None and _time_left() > 15.0:
        polished, pol_r = _exact_polish(best_pos, max_iter=1000)
        if pol_r > best_r:
            best_r = pol_r
            best_pos = polished

    # Safety: ensure valid shape and bounds
    best_pos = np.asarray(best_pos, dtype=float)
    if best_pos.shape != (N, 2):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]
    best_pos = np.clip(best_pos, -0.49999, 0.49999)
    if not np.all(np.isfinite(best_pos)):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]

    return best_pos
