"""Pack 50 equal circles in a unit square; maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
EPS = 1e-12


def _pairwise_min_dist(c):
    # c: (N,2)
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d.min()


def _wall_dist(c):
    # distance from each centre to the nearest wall
    return (HALF - np.abs(c)).min()


def _radius(c):
    return min(_pairwise_min_dist(c) * 0.5, _wall_dist(c))


def _neg_radius(c):
    return -_radius(c)


def _neg_radius_grad(c):
    # Subgradient-ish: use the active constraint (min of pairwise half-dist and wall)
    # For a smooth-ish objective we can compute gradient of the min by taking
    # the gradient of the active term. For simplicity, use a soft min via log-sum-exp.
    return None  # we'll use numerical or rely on L-BFGS-B with finite diff


def _soft_neg_radius(c, beta=200.0):
    # Smooth surrogate: -logsumexp of the "constraints"
    # We want to maximize r = min( d_ij/2 for i<j, wall_i )
    # Equivalent to maximize t s.t. t <= d_ij/2 and t <= wall_i
    # Use log-sum-exp soft min: softmin(x) = -(1/beta)*log(sum(exp(-beta*x)))
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    d_half = d / 2.0
    walls = HALF - np.abs(c)  # shape (N,)

    # Gather all constraint values
    # pairwise: N*(N-1)/2 values, walls: N values
    iu = np.triu_indices(N, k=1)
    pair_vals = d_half[iu]
    wall_vals = walls.flatten()
    all_vals = np.concatenate([pair_vals, wall_vals])

    # softmin
    m = all_vals.min()
    logsum = m + np.log(np.exp(-beta * (all_vals - m)).sum())
    softmin_val = -logsum / beta
    return -softmin_val


def _soft_neg_radius_grad(c, beta=200.0):
    diff = c[:, None, :] - c[None, :, :]
    d2 = (diff ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    d_half = d / 2.0
    walls = HALF - np.abs(c)

    iu = np.triu_indices(N, k=1)
    pair_vals = d_half[iu]
    wall_vals = walls.flatten()
    all_vals = np.concatenate([pair_vals, wall_vals])

    m = all_vals.min()
    exps = np.exp(-beta * (all_vals - m))
    Z = exps.sum()
    # d/dx of softmin = - (1/beta) * d/dx logsumexp(-beta * x)
    # = - (1/beta) * sum(-beta * exp(-beta*x_i) / Z) * dx_i/dx
    # = sum(exp(-beta*x_i)/Z) * dx_i/dx
    # So gradient of -softmin = - sum(w_i * dx_i/dx) where w_i = exp(-beta*x_i)/Z

    weights = exps / Z
    pair_w = weights[:len(pair_vals)]
    wall_w = weights[len(pair_vals):]

    grad = np.zeros_like(c)

    # Pairwise contributions
    # For each pair (i,j), value = d_ij/2, derivative wrt c_i = (c_i - c_j)/d_ij
    # weight = pair_w[k]
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        w = pair_w[k]
        dij = d[i, j]
        if dij < EPS:
            continue
        vec = (c[i] - c[j]) / dij
        grad[i] -= w * vec  # d(-softmin)/dc_i = -w * d(d_ij/2)/dc_i = -w * (c_i-c_j)/d_ij
        grad[j] += w * vec

    # Wall contributions
    # wall_i = min over coords of (HALF - |c_i_k|)
    # Actually walls array is per-coordinate: (N,2) flattened
    # wall_vals[k] = HALF - |c[coord]|
    # d(-softmin)/dc = -w * d(wall)/dc
    # d(HALF - |x|)/dx = -sign(x)
    # So d(-softmin)/dc = -w * (-sign(x)) = w * sign(x)
    wall_w_2d = wall_w.reshape(N, 2)
    signs = np.sign(c)
    grad += wall_w_2d * signs

    return grad


def _grid_start(m):
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _random_start(scale=0.45):
    return (np.random.rand(N, 2) - 0.5) * 2 * scale


def _optimize_from_start(x0, beta=200.0, maxiter=2000, tol=1e-10):
    """Optimize using L-BFGS-B on the soft objective."""
    res = minimize(
        _soft_neg_radius,
        x0,
        jac=_soft_neg_radius_grad,
        args=(beta,),
        method='L-BFGS-B',
        bounds=[(-HALF + 1e-6, HALF - 1e-6)] * (2 * N),
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-8, 'maxfun': 5000},
    )
    return res.x


def _hard_disk_expand(c, target_r, max_iter=5000, tol=1e-12):
    """Iteratively expand circles (move centers apart) to reach target radius."""
    c = c.copy()
    for _ in range(max_iter):
        moved = False
        # Compute all distances
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)

        # Find pairs closer than 2*target_r
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < 2 * target_r - tol:
                    # Push apart
                    vec = c[i] - c[j]
                    dist = np.linalg.norm(vec)
                    if dist < EPS:
                        vec = np.array([1.0, 0.0])
                        dist = 1.0
                    vec = vec / dist
                    # Move each by half the overlap
                    overlap = (2 * target_r - d[i, j]) * 0.5
                    c[i] += vec * overlap
                    c[j] -= vec * overlap
                    moved = True

        # Clamp to square
        c = np.clip(c, -HALF + 1e-6, HALF - 1e-6)

        if not moved:
            break

    return c


def _score(c):
    return _radius(c)


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill

    np.random.seed(42)

    best_c = _grid_start(7)
    best_r = _score(best_c)

    # Try several random starts
    for trial in range(30):
        if time.time() > deadline:
            break

        x0 = _random_start()
        # First do a quick L-BFGS-B on soft objective
        try:
            c_opt = _optimize_from_start(x0, beta=100.0, maxiter=1000)
            r_opt = _score(c_opt)
            if r_opt > best_r:
                best_c = c_opt.copy()
                best_r = r_opt
        except Exception:
            pass

    # Now do a more careful polish on the best solution
    if time.time() < deadline - 10:
        try:
            c_polish = _optimize_from_start(best_c, beta=500.0, maxiter=3000, tol=1e-12)
            r_polish = _score(c_polish)
            if r_polish > best_r:
                best_c = c_polish
                best_r = r_polish
        except Exception:
            pass

    # Try hard-disk expansion from best
    if time.time() < deadline - 5:
        # Estimate a slightly larger target
        current_r = _score(best_c)
        target = current_r * 1.01
        c_expanded = _hard_disk_expand(best_c, target, max_iter=2000)
        r_expanded = _score(c_expanded)
        if r_expanded > best_r:
            best_c = c_expanded
            best_r = r_expanded

        # Polish again
        if time.time() < deadline - 2:
            try:
                c_final = _optimize_from_start(best_c, beta=1000.0, maxiter=2000, tol=1e-12)
                r_final = _score(c_final)
                if r_final > best_r:
                    best_c = c_final
                    best_r = r_final
            except Exception:
                pass

    # Ensure validity: clip to square
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)

    return best_c
