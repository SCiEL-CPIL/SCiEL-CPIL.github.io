"""Force-directed circle packing in unit square for N=26 circles."""
import numpy as np
import time

N = 26
DEADLINE = 110.0  # seconds, safety margin under 120s


def _pairwise(d, n):
    """Return (dx, dy, dist) upper-triangular arrays for n points in d (n,2)."""
    dx = d[:, None, 0] - d[None, :, 0]
    dy = d[:, None, 1] - d[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    iu = np.triu_indices(n, k=1)
    return dx[iu], dy[iu], dist[iu]


def _energy(d, r, lam=100.0, mu=50.0):
    """Energy: collision penalty + boundary penalty. Lower is better."""
    n = len(d)
    dx, dy, dist = _pairwise(d, n)
    req = r[iu_i] + r[iu_j] if False else None  # placeholder
    # recompute properly
    iu = np.triu_indices(n, k=1)
    i, j = iu
    req = r[i] + r[j]
    overlap = np.maximum(req - dist, 0.0)
    E_col = lam * np.sum(overlap * overlap)
    # boundary: circle must stay in [0,1]^2
    E_b = mu * (np.sum(np.maximum(r - d[:, 0], 0.0) ** 2) +
                np.sum(np.maximum(r - d[:, 1], 0.0) ** 2) +
                np.sum(np.maximum(r - (1.0 - d[:, 0]), 0.0) ** 2) +
                np.sum(np.maximum(r - (1.0 - d[:, 1]), 0.0) ** 2))
    return E_col + E_b


def _forces(d, r, lam=100.0, mu=50.0):
    """Gradient of energy w.r.t. positions (n,2)."""
    n = len(d)
    iu = np.triu_indices(n, k=1)
    i, j = iu
    dx = d[i, 0] - d[j, 0]
    dy = d[i, 1] - d[j, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.where(dist > 1e-12, dist, 1e-12)
    req = r[i] + r[j]
    overlap = np.maximum(req - dist, 0.0)
    # d(E_col)/d(pos) = -2*lam*overlap * (pos_i - pos_j)/dist
    f = np.zeros((n, 2))
    coef = -2.0 * lam * overlap / dist_safe
    f[i, 0] += coef * dx
    f[j, 0] -= coef * dx
    f[i, 1] += coef * dy
    f[j, 1] -= coef * dy
    # boundary forces
    f[:, 0] += 2.0 * mu * np.maximum(r - d[:, 0], 0.0) * (-1.0) + \
               2.0 * mu * np.maximum(r - (1.0 - d[:, 0]), 0.0) * (1.0)
    f[:, 1] += 2.0 * mu * np.maximum(r - d[:, 1], 0.0) * (-1.0) + \
               2.0 * mu * np.maximum(r - (1.0 - d[:, 1]), 0.0) * (1.0)
    return f


def _clamp(d, r):
    """Clamp positions so circles stay inside [0,1]^2."""
    d = d.copy()
    d[:, 0] = np.clip(d[:, 0], r, 1.0 - r)
    d[:, 1] = np.clip(d[:, 1], r, 1.0 - r)
    return d


def _separate(d, r, iters=200):
    """Iteratively separate overlapping circles (position projection)."""
    n = len(d)
    iu = np.triu_indices(n, k=1)
    i, j = iu
    for _ in range(iters):
        dx = d[i, 0] - d[j, 0]
        dy = d[i, 1] - d[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        req = r[i] + r[j]
        overlap = req - dist
        mask = overlap > 1e-10
        if not np.any(mask):
            break
        im, jm = i[mask], j[mask]
        dmx = d[im, 0] - d[jm, 0]
        dmy = d[im, 1] - d[jm, 1]
        dm = np.sqrt(dmx * dmx + dmy * dmy)
        dm = np.where(dm > 1e-12, dm, 1e-12)
        push = (overlap[mask] + 1e-9) / 2.0
        d[im, 0] += push * dmx / dm
        d[jm, 0] -= push * dmx / dm
        d[im, 1] += push * dmy / dm
        d[jm, 1] -= push * dmy / dm
        d = _clamp(d, r)
    return d


def _is_valid(d, r, tol=1e-6):
    n = len(d)
    if np.any(d[:, 0] < r - tol) or np.any(d[:, 0] > 1.0 - r + tol):
        return False
    if np.any(d[:, 1] < r - tol) or np.any(d[:, 1] > 1.0 - r + tol):
        return False
    iu = np.triu_indices(n, k=1)
    i, j = iu
    dx = d[i, 0] - d[j, 0]
    dy = d[i, 1] - d[j, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < (r[i] + r[j]) - tol):
        return False
    return True


def _grow_radii(d, r, iters=300):
    """Grow all radii uniformly while maintaining no overlaps. Returns new d, r."""
    n = len(d)
    for _ in range(iters):
        # find the maximum uniform growth delta such that no overlap
        # approximate: for each pair, slack = dist - (r_i + r_j)
        iu = np.triu_indices(n, k=1)
        i, j = iu
        dx = d[i, 0] - d[j, 0]
        dy = d[i, 1] - d[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        slack_pair = dist - (r[i] + r[j])
        # boundary slack
        slack_b = np.minimum(np.minimum(d[:, 0], 1.0 - d[:, 0]),
                             np.minimum(d[:, 1], 1.0 - d[:, 1])) - r
        # uniform growth t: need r_i + t + r_j + t <= dist for all pairs
        # => 2t <= min_pair slack_pair  => t <= min_pair_slack/2
        t_pair = np.min(slack_pair) / 2.0
        t_b = np.min(slack_b)
        t = max(0.0, min(t_pair, t_b))
        if t < 1e-8:
            break
        r = r + t
        d = _clamp(d, r)
        d = _separate(d, r, iters=50)
    return d, r


def _grid_init(n):
    side = int(np.ceil(np.sqrt(n)))
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    d = np.array(cells[:n])
    r = np.full(n, 0.5 / side - 0.001)
    return d, r


def _hex_init(n):
    # hexagonal-ish packing
    d = []
    r = 0.12
    row = 0
    while len(d) < n:
        x0 = r
        if row % 2 == 0:
            xs = np.arange(x0, 1.0 - r + 1e-9, 2 * r)
        else:
            xs = np.arange(x0 + r, 1.0 - r + 1e-9, 2 * r)
        y = r + row * np.sqrt(3) * r
        if y > 1.0 - r:
            break
        for x in xs:
            if len(d) < n:
                d.append((x, y))
        row += 1
    if len(d) < n:
        # fill remaining with grid
        side = int(np.ceil(np.sqrt(n)))
        for i in range(side):
            for j in range(side):
                if len(d) < n:
                    d.append(((i + 0.5) / side, (j + 0.5) / side))
    d = np.array(d[:n])
    r = np.full(n, 0.10)
    return d, r


def _random_init(n, rng):
    d = rng.uniform(0.1, 0.9, size=(n, 2))
    r = np.full(n, 0.08)
    return d, r


def _anneal(d, r, rng, t0=0.05, t1=0.0005, steps=2000, lam=100.0, mu=50.0):
    """Simulated annealing on positions."""
    n = len(d)
    best_d = d.copy()
    best_r = r.copy()
    best_score = -np.sum(r)
    for s in range(steps):
        T = t0 * (t1 / t0) ** (s / max(steps - 1, 1))
        # random perturbation
        d_new = d + rng.normal(0, T, size=(n, 2))
        d_new = _clamp(d_new, r)
        # try to improve: separate and grow
        d_new = _separate(d_new, r, iters=30)
        d_new, r_new = _grow_radii(d_new, r.copy(), iters=30)
        score = np.sum(r_new)
        if score > best_score:
            best_score = score
            best_d = d_new.copy()
            best_r = r_new.copy()
        # accept with probability
        E_old = _energy(d, r, lam, mu)
        E_new = _energy(d_new, r_new, lam, mu)
        if E_new < E_old or rng.random() < np.exp(-(E_new - E_old) / max(T, 1e-9)):
            d, r = d_new, r_new
    return best_d, best_r, best_score


def _polish(d, r, lam=100.0, mu=50.0, steps=500):
    """Gradient descent polish on positions with growing radii."""
    n = len(d)
    for s in range(steps):
        f = _forces(d, r, lam, mu)
        lr = 0.01 * (1.0 - s / steps) + 0.001
        d = d - lr * f
        d = _clamp(d, r)
        d = _separate(d, r, iters=30)
        d, r = _grow_radii(d, r, iters=10)
    return d, r


def construct():
    start = time.time()
    rng = np.random.default_rng(42)
    best_score = -np.inf
    best_d = None
    best_r = None

    # multiple initializations
    inits = [
        ("grid", _grid_init),
        ("hex", _hex_init),
    ]
    for _ in range(3):
        inits.append(("rand", lambda n, rng=rng: _random_init(n, rng)))

    for name, init_fn in inits:
        if time.time() - start > DEADLINE - 30:
            break
        d, r = init_fn(N)
        # anneal
        d_a, r_a, score_a = _anneal(d, r, rng, t0=0.04, t1=0.0003, steps=1500)
        if score_a > best_score:
            best_score = score_a
            best_d = d_a
            best_r = r_a
        # polish
        if time.time() - start > DEADLINE - 10:
            break
        d_p, r_p = _polish(d_a, r_a, steps=300)
        if _is_valid(d_p, r_p):
            score_p = np.sum(r_p)
            if score_p > best_score:
                best_score = score_p
                best_d = d_p
                best_r = r_p
        else:
            # try to repair
            d_p = _separate(d_p, r_p, iters=200)
            d_p = _clamp(d_p, r_p)
            if _is_valid(d_p, r_p):
                score_p = np.sum(r_p)
                if score_p > best_score:
                    best_score = score_p
                    best_d = d_p
                    best_r = r_p

    # final: try to grow radii more aggressively on best
    if best_d is not None and time.time() - start < DEADLINE - 5:
        d_f, r_f = _grow_radii(best_d, best_r, iters=500)
        d_f = _separate(d_f, r_f, iters=200)
        d_f = _clamp(d_f, r_f)
        if _is_valid(d_f, r_f):
            if np.sum(r_f) > best_score:
                best_score = np.sum(r_f)
                best_d = d_f
                best_r = r_f

    # fallback to grid if something went wrong
    if best_d is None or not _is_valid(best_d, best_r, tol=1e-4):
        d, r = _grid_init(N)
        best_d, best_r = d, r

    # final safety: ensure validity
    best_d = _clamp(best_d, best_r)
    best_d = _separate(best_d, best_r, iters=100)
    best_d = _clamp(best_d, best_r)

    out = np.column_stack([best_d[:, 0], best_d[:, 1], best_r])
    return out
