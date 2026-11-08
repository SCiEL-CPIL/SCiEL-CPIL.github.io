"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii.

Combined approach:
  * Multi-start diverse initializations (grid / hex / corner-heavy / random).
  * Fast force-based relaxation with radius growth (current program machinery).
  * Rigid-body contact-graph polish: maximize the MINIMUM GAP using a
    differentiable log-sum-exp (min-pooling) over pairwise and wall gaps,
    so L-BFGS-B drives the system toward a fully-stressed configuration.
  * Binary-search radius growth on the contact graph (measured speedup idea).
  * Final smooth soft-penalty polish to repair any residual overlap.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_START = time.time()
DEADLINE = _START + 142.0  # safety margin under the 150 s kill
RNG = np.random.default_rng(20240517)

R_MIN = 1e-6
R_MAX = 0.5

# ---------------------------------------------------------------------------
# validation / scoring
# ---------------------------------------------------------------------------
def _is_valid(c, tol=1e-9):
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    if r.min() < -tol:
        return False
    if np.any(x + r > 1 + tol) or np.any(x - r < -tol):
        return False
    if np.any(y + r > 1 + tol) or np.any(y - r < -tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    s = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d[mask] < s[mask] - tol):
        return False
    return True


def _score(c):
    return float(np.sum(c[:, 2])) if _is_valid(c) else -np.inf


def _clamp(c):
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    r = np.clip(r, R_MIN, R_MAX)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    return np.column_stack([x, y, r])


# ---------------------------------------------------------------------------
# force-based relaxation (current program machinery)
# ---------------------------------------------------------------------------
def _vectorized_forces(c):
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    fx = np.zeros(N); fy = np.zeros(N); fr = np.zeros(N)
    fx -= 500.0 * np.clip(r - x, 0, None)
    fx += 500.0 * np.clip(x + r - 1, 0, None)
    fy -= 500.0 * np.clip(r - y, 0, None)
    fy += 500.0 * np.clip(y + r - 1, 0, None)
    fr += 500.0 * (np.clip(r - x, 0, None) + np.clip(x + r - 1, 0, None) +
                   np.clip(r - y, 0, None) + np.clip(y + r - 1, 0, None))

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, 1.0)
    s = r[:, None] + r[None, :]
    overlap = np.clip(s - d, 0, None)
    active = overlap > 1e-8
    for i in range(N):
        for j in range(i + 1, N):
            if active[i, j]:
                ux = dx[i, j] / d[i, j]
                uy = dy[i, j] / d[i, j]
                f = 40.0 * overlap[i, j]
                fx[i] += f * ux; fy[i] += f * uy
                fx[j] -= f * ux; fy[j] -= f * uy
                fr[i] += f * 0.5; fr[j] += f * 0.5
    fr += 10.0
    return fx, fy, fr


def _relax_and_grow(c0, max_iters=1200, grow_rate=0.0004):
    c = c0.copy()
    best = _score(c)
    best_c = c.copy()
    step = 0.03
    no_improve = 0
    for it in range(max_iters):
        if time.time() > DEADLINE - 4.0:
            break
        c = _clamp(c)
        fx, fy, fr = _vectorized_forces(c)
        fmag = np.hypot(fx, fy)
        adaptive = step / (1.0 + 0.01 * fmag.max())
        x = c[:, 0] + adaptive * fx
        y = c[:, 1] + adaptive * fy
        r = c[:, 2] + adaptive * fr + grow_rate
        c = _clamp(np.column_stack([x, y, r]))
        val = _score(c)
        if val > best:
            best = val; best_c = c.copy(); no_improve = 0
            step = min(step * 1.05, 0.1)
        else:
            no_improve += 1
            step *= 0.95
            if it > 5:
                c = best_c.copy()
            if step < 1e-6 or no_improve > 120:
                break
    return best_c, best


# ---------------------------------------------------------------------------
# initializations
# ---------------------------------------------------------------------------
def _seed_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])


def _seed_hex():
    positions = []
    r = 0.18
    dx = 2 * r
    dy = r * np.sqrt(3)
    for row in range(5):
        for col in range(5):
            x = r + col * dx + (row % 2) * dx / 2
            y = r + row * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                positions.append((x, y, r))
    while len(positions) < N:
        positions.append((RNG.uniform(0.1, 0.9), RNG.uniform(0.1, 0.9),
                          RNG.uniform(0.05, 0.1)))
    return np.array(positions[:N])


def _seed_random():
    x = RNG.uniform(0.08, 0.92, N)
    y = RNG.uniform(0.08, 0.92, N)
    r = RNG.uniform(0.03, 0.1, N)
    return np.column_stack([x, y, r])


def _seed_corner_heavy():
    positions = []; radii = []
    for cx, cy in [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]:
        positions.append((cx, cy)); radii.append(0.13)
    for cx, cy in [(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]:
        positions.append((cx, cy)); radii.append(0.1)
    positions.append((0.5, 0.5)); radii.append(0.12)
    while len(positions) < N:
        positions.append((RNG.uniform(0.1, 0.9), RNG.uniform(0.1, 0.9)))
        radii.append(RNG.uniform(0.04, 0.08))
    x = np.array([p[0] for p in positions[:N]])
    y = np.array([p[1] for p in positions[:N]])
    r = np.array(radii[:N])
    return np.column_stack([x, y, r])


# ---------------------------------------------------------------------------
# contact-graph (min-gap) objective -- differentiable min-pooling
# ---------------------------------------------------------------------------
def _contact_objective(v, T=0.02):
    """Minimize -min_gap.  min_gap = min over all pairwise and wall gaps.
    Smoothed with a log-sum-exp (min-pooling) so L-BFGS-B pushes every
    contact toward the tightest one -> fully stressed configuration."""
    c = v.reshape(N, 3)
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]

    # wall gaps (should be >= 0)
    gw = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    # pairwise gaps
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    s = r[:, None] + r[None, :]
    gp = d - s
    idx = np.triu_indices(N, k=1)
    gaps = np.concatenate([gw, gp[idx]])
    gaps = np.clip(gaps, -30.0, 30.0)

    # smooth min = -T * log( sum( exp(-g/T) ) )
    m = gaps.min()
    z = np.exp(-(gaps - m) / T)
    min_gap = m - T * np.log(z.sum())
    return -min_gap


def _contact_polish(c0, T=0.02, max_iter=250):
    if time.time() > DEADLINE - 10.0:
        return c0, _score(c0)
    v0 = c0.flatten()
    bounds = [(0.0, 1.0), (0.0, 1.0), (R_MIN, R_MAX)] * N
    try:
        res = minimize(_contact_objective, v0, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-13, 'gtol': 1e-9})
        cp = _clamp(res.x.reshape(N, 3))
    except Exception:
        cp = c0
    val = _score(cp)
    if val > _score(c0):
        return cp, val
    return c0, _score(c0)


# ---------------------------------------------------------------------------
# smooth soft-penalty objective (inspiration) for final overlap repair
# ---------------------------------------------------------------------------
def _smooth_objective(v, beta=60.0):
    c = v.reshape(N, 3)
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    h_left = r - x; h_right = x + r - 1
    h_bottom = r - y; h_top = y + r - 1

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    d = np.where(d < 1e-12, 1e-12, d)
    s = r[:, None] + r[None, :]
    idx = np.triu_indices(N, k=1)
    h_overlap = s[idx] - d[idx]

    hb = np.concatenate([h_left, h_right, h_bottom, h_top])
    hb = np.clip(hb, -50, 50); ho = np.clip(h_overlap, -50, 50)
    sb = np.log1p(np.exp(beta * hb)) / beta
    so = np.log1p(np.exp(beta * ho)) / beta
    obj = -np.sum(r) + 12.0 * (sb.sum() + so.sum())

    grad = np.zeros(3 * N)
    grad[2::3] = -1.0
    gb = 12.0 * (np.exp(beta * hb) / (1.0 + np.exp(beta * hb)))
    go = 12.0 * (np.exp(beta * ho) / (1.0 + np.exp(beta * ho)))
    for i in range(N):
        grad[i * 3 + 0] += -gb[i] + gb[N + i]
        grad[i * 3 + 1] += -gb[2 * N + i] + gb[3 * N + i]
        grad[i * 3 + 2] += gb[i] + gb[N + i] + gb[2 * N + i] + gb[3 * N + i]
    for k in range(len(go)):
        i, j = idx[0][k], idx[1][k]
        ux = (x[i] - x[j]) / d[i, j]
        uy = (y[i] - y[j]) / d[i, j]
        grad[i * 3 + 0] += -go[k] * ux
        grad[i * 3 + 1] += -go[k] * uy
        grad[i * 3 + 2] += go[k]
        grad[j * 3 + 0] += go[k] * ux
        grad[j * 3 + 1] += go[k] * uy
        grad[j * 3 + 2] += go[k]
    return obj, grad


def _smooth_polish(c0, beta=60.0, max_iter=300):
    if time.time() > DEADLINE - 6.0:
        return c0, _score(c0)
    v0 = c0.flatten()
    bounds = [(0.0, 1.0), (0.0, 1.0), (R_MIN, R_MAX)] * N
    try:
        res = minimize(_smooth_objective, v0, method='L-BFGS-B', jac=True,
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-13, 'gtol': 1e-10})
        cp = _clamp(res.x.reshape(N, 3))
    except Exception:
        cp = c0
    val = _score(cp)
    if val > _score(c0):
        return cp, val
    return c0, _score(c0)


# ---------------------------------------------------------------------------
# binary-search radius growth on the contact graph
# ---------------------------------------------------------------------------
def _grow_radii(c0, iters=12):
    c = c0.copy()
    best = _score(c); best_c = c.copy()
    for _ in range(iters):
        if time.time() > DEADLINE - 8.0:
            break
        lo, hi = 0.0, 0.02
        ok = c.copy()
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            cand = c.copy()
            cand[:, 2] = np.clip(cand[:, 2] + mid, R_MIN, R_MAX)
            cand = _contact_polish(cand, T=0.02, max_iter=80)
            if _is_valid(cand, tol=1e-9):
                ok = cand; lo = mid
            else:
                hi = mid
        if lo <= 0:
            break
        val = _score(ok)
        if val > best:
            best = val; best_c = ok.copy(); c = ok.copy()
        else:
            break
    return best_c, best


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def construct():
    initializers = [_seed_grid, _seed_hex, _seed_corner_heavy, _seed_random]
    best_c = None
    best_val = -np.inf

    # Phase 1: relax + grow from each diverse initialization
    for init_func in initializers:
        if time.time() > DEADLINE - 40.0:
            break
        c0 = init_func()
        cc, val = _relax_and_grow(c0, max_iters=1000, grow_rate=0.0004)
        if val > best_val:
            best_val = val; best_c = cc.copy()

    # Phase 2: a few extra random starts
    for i in range(8):
        if time.time() > DEADLINE - 30.0:
            break
        cc, val = _relax_and_grow(_seed_random(), max_iters=600, grow_rate=0.0003)
        if val > best_val:
            best_val = val; best_c = cc.copy()

    if best_c is None:
        best_c = _seed_grid()
        best_val = _score(best_c)

    # Phase 3: contact-graph min-gap polish (fully stressed configuration)
    if time.time() < DEADLINE - 15.0:
        cc, val = _contact_polish(best_c, T=0.02, max_iter=250)
        if val > best_val:
            best_val = val; best_c = cc.copy()

    # Phase 4: binary-search radius growth on the contact graph
    if time.time() < DEADLINE - 10.0:
        cc, val = _grow_radii(best_c, iters=10)
        if val > best_val:
            best_val = val; best_c = cc.copy()

    # Phase 5: final smooth soft-penalty repair
    if time.time() < DEADLINE - 4.0:
        cc, val = _smooth_polish(best_c, beta=60.0, max_iter=200)
        if val > best_val:
            best_val = val; best_c = cc.copy()

    # Safety fallback
    if best_c is None or best_val <= 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_c = np.array(cells[:N])

    return best_c
