"""Differentiable circle packing via soft contact graph + smooth min surrogate."""
import time
import numpy as np
from scipy.optimize import minimize

N = 32
DEADLINE = 140.0  # safety margin before 150s kill

def _smooth_min(values, k):
    """Soft minimum via log-sum-exp trick: min_i v_i ~ -logsumexp(-k*v)/k."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    m = np.min(values)
    # stable: -logsumexp(-k*v)/k = -m - log(sum(exp(-k*(v-m))))/k
    s = np.sum(np.exp(-k * (values - m)))
    return -m - np.log(s) / k

def _smooth_min_grad(values, k):
    """Gradient of smooth_min w.r.t. each input (weights)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros_like(values)
    m = np.min(values)
    exps = np.exp(-k * (values - m))
    s = np.sum(exps)
    weights = exps / s
    return weights  # d(smooth_min)/d(v_i) = w_i (since -1/k * d/dv_i log(sum exp(-k(v-m))) = w_i)

def _pairwise_distances(P):
    """P: (n,2) centers. Returns (n,n) distance matrix."""
    diff = P[:, None, :] - P[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1) + 1e-30)

def _objective_and_grad(z, k_soft, w_wall):
    """
    z: (3n,) flattened (x,y,r) for n circles.
    Objective: minimize  -sum(r) - lambda * smooth_min(margins)
    where margins = min over pairs (dist_ij - r_i - r_j) and walls (x-r, 1-x-r, y-r, 1-y-r)
    Actually we maximize sum r subject to margins >= 0. Use penalty:
    L = -sum(r) - mu * smooth_min(all_margins, k)
    Gradient computed analytically.
    """
    n = len(z) // 3
    P = z[:2*n].reshape(n, 2)
    R = z[2*n:]

    # margins to walls
    wall_margins = np.concatenate([
        P[:, 0] - R,          # left
        1.0 - P[:, 0] - R,    # right
        P[:, 1] - R,          # bottom
        1.0 - P[:, 1] - R,    # top
    ])

    # pairwise margins
    D = _pairwise_distances(P)
    pair_margin_mat = D - R[:, None] - R[None, :]
    iu = np.triu_indices(n, k=1)
    pair_margins = pair_margin_mat[iu]

    all_margins = np.concatenate([wall_margins, pair_margins])

    sm = _smooth_min(all_margins, k_soft)
    obj = -np.sum(R) - w_wall * sm
    return obj, sm, all_margins

def _grad(z, k_soft, w_wall):
    n = len(z) // 3
    P = z[:2*n].reshape(n, 2)
    R = z[2*n:]

    wall_margins = np.concatenate([
        P[:, 0] - R,
        1.0 - P[:, 0] - R,
        P[:, 1] - R,
        1.0 - P[:, 1] - R,
    ])

    D = _pairwise_distances(P)
    pair_margin_mat = D - R[:, None] - R[None, :]
    iu = np.triu_indices(n, k=1)
    pair_margins = pair_margin_mat[iu]

    all_margins = np.concatenate([wall_margins, pair_margins])
    weights = _smooth_min_grad(all_margins, k_soft)

    grad = np.zeros_like(z)

    # d(-sum R)/dR = -1
    grad[2*n:] += -1.0

    # Penalty part: -w * sm, d sm / d margin_i = weights[i]
    # So contribution to grad = -w * weights[i] * (d margin_i / d z)

    # Wall margins: for circle i
    # margin[4i+0] = x_i - r_i  -> d/dx_i=1, d/dr_i=-1
    # margin[4i+1] = 1 - x_i - r_i -> d/dx_i=-1, d/dr_i=-1
    # margin[4i+2] = y_i - r_i -> d/dy_i=1, d/dr_i=-1
    # margin[4i+3] = 1 - y_i - r_i -> d/dy_i=-1, d/dr_i=-1
    w_wall_arr = weights[:4*n]
    for i in range(n):
        wi0 = w_wall_arr[4*i]
        wi1 = w_wall_arr[4*i+1]
        wi2 = w_wall_arr[4*i+2]
        wi3 = w_wall_arr[4*i+3]
        grad[2*i + 0] += -w_wall * (wi0 - wi1)   # x_i
        grad[2*i + 1] += -w_wall * (wi2 - wi3)   # y_i
        grad[2*n + i] += -w_wall * (-(wi0) - (wi1) - (wi2) - (wi3))  # r_i

    # Pair margins: index mapping
    # pair_margins[k] for k in 0..(n(n-1)/2 -1) corresponds to (iu[0][k], iu[1][k])
    # margin = D_ij - r_i - r_j
    # d margin / d x_i = (x_i - x_j)/D_ij, similarly for others
    w_pair = weights[4*n:]
    idx_i = iu[0]
    idx_j = iu[1]
    for k in range(len(idx_i)):
        i = idx_i[k]
        j = idx_j[k]
        w = w_pair[k]
        dij = D[i, j]
        if dij < 1e-12:
            continue
        dx = (P[i, 0] - P[j, 0]) / dij
        dy = (P[i, 1] - P[j, 1]) / dij
        grad[2*i + 0] += -w_wall * w * dx
        grad[2*i + 1] += -w_wall * w * dy
        grad[2*j + 0] += -w_wall * w * (-dx)
        grad[2*j + 1] += -w_wall * w * (-dy)
        grad[2*n + i] += -w_wall * w * (-1.0)
        grad[2*n + j] += -w_wall * w * (-1.0)

    return grad

def _is_valid(z, tol=1e-7):
    n = len(z) // 3
    P = z[:2*n].reshape(n, 2)
    R = z[2*n:]
    if np.any(R < 0):
        return False
    if np.any(P[:, 0] - R < -tol) or np.any(1 - P[:, 0] - R < -tol):
        return False
    if np.any(P[:, 1] - R < -tol) or np.any(1 - P[:, 1] - R < -tol):
        return False
    D = _pairwise_distances(P)
    if np.any(D - R[:, None] - R[None, :] < -tol):
        return False
    return True

def _score(z):
    n = len(z) // 3
    R = z[2*n:]
    return float(np.sum(R))

def _hex_init(n, jitter=0.0, seed=0):
    """Hexagonal lattice packing in unit square, take first n."""
    rng = np.random.default_rng(seed)
    pts = []
    # Hexagonal packing: rows offset
    # radius r, spacing 2r horizontally, sqrt(3)*r vertically
    # We'll generate a grid of candidate centers with hex spacing and pick n that fit
    # Use a fine hex grid and select centers that fit in [r, 1-r]
    # Start with an estimate of r ~ 0.08
    r_est = 0.08
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    # Generate hex points
    rows = 0
    cols = 0
    candidates = []
    y = r_est
    row = 0
    while y <= 1 - r_est:
        x = r_est + (0 if row % 2 == 0 else r_est)
        while x <= 1 - r_est:
            candidates.append([x, y])
            x += dx
        y += dy
        row += 1
    candidates = np.array(candidates)
    # Shuffle deterministically and take n
    idx = rng.permutation(len(candidates))
    sel = candidates[idx[:n]]
    if jitter > 0:
        sel = sel + rng.normal(0, jitter, size=sel.shape)
    # Clip to feasible region with small radius
    R = np.full(n, r_est * 0.9)
    P = np.clip(sel, R, 1 - R)
    z = np.concatenate([P.flatten(), R])
    return z

def _grid_init(n, seed=0):
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.01
            cells.append([x, y, r])
    cells = np.array(cells[:n])
    if len(cells) < n:
        # pad
        extra = np.zeros((n - len(cells), 3))
        extra[:, 0] = rng.uniform(0.1, 0.9, n - len(cells))
        extra[:, 1] = rng.uniform(0.1, 0.9, n - len(cells))
        extra[:, 2] = 0.05
        cells = np.vstack([cells, extra])
    P = cells[:, :2]
    R = cells[:, 2]
    z = np.concatenate([P.flatten(), R])
    return z

def construct():
    start = time.time()
    best_z = None
    best_score = -np.inf

    # Initial candidates
    inits = []
    for s in range(6):
        inits.append(_hex_init(N, jitter=0.005, seed=100 + s))
    for s in range(3):
        inits.append(_grid_init(N, seed=200 + s))

    # Optimization schedule: anneal k_soft and w_wall
    # Phase 1: moderate k, low penalty weight (allow some overlap to find good layout)
    # Phase 2: high k, high penalty weight (enforce constraints tightly)
    schedules = [
        # (k_soft, w_wall, maxiter)
        (5.0, 50.0, 300),
        (15.0, 200.0, 300),
        (40.0, 1000.0, 400),
    ]

    for init in inits:
        if time.time() - start > DEADLINE - 10:
            break
        z = init.copy()
        for (k_soft, w_wall, maxiter) in schedules:
            if time.time() - start > DEADLINE - 10:
                break
            try:
                res = minimize(
                    lambda zz: _objective_and_grad(zz, k_soft, w_wall)[0],
                    z,
                    jac=lambda zz: _grad(zz, k_soft, w_wall),
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * (2 * N) + [(1e-6, 0.5)] * N,
                    options={'maxiter': maxiter, 'maxfun': 5000, 'gtol': 1e-6},
                )
                z = res.x
            except Exception:
                pass

        # Validate and score
        if _is_valid(z, tol=1e-6):
            sc = _score(z)
            if sc > best_score:
                best_score = sc
                best_z = z.copy()

    # Polish: high-precision local solve on best
    if best_z is not None and time.time() - start < DEADLINE - 15:
        z = best_z.copy()
        for (k_soft, w_wall, maxiter) in [(100.0, 5000.0, 500), (200.0, 20000.0, 500)]:
            if time.time() - start > DEADLINE - 5:
                break
            try:
                res = minimize(
                    lambda zz: _objective_and_grad(zz, k_soft, w_wall)[0],
                    z,
                    jac=lambda zz: _grad(zz, k_soft, w_wall),
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * (2 * N) + [(1e-6, 0.5)] * N,
                    options={'maxiter': maxiter, 'maxfun': 10000, 'gtol': 1e-9},
                )
                z = res.x
            except Exception:
                pass
        if _is_valid(z, tol=1e-7):
            sc = _score(z)
            if sc > best_score:
                best_score = sc
                best_z = z.copy()

    if best_z is None:
        # Fallback to grid
        return _grid_init(N, seed=0).reshape(N, 3)

    # Format output
    n = len(best_z) // 3
    P = best_z[:2*n].reshape(n, 2)
    R = best_z[2*n:]
    out = np.column_stack([P, R])
    return out
