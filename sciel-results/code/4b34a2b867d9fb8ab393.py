"""Circle packing in unit square n=32 -- gradient-based continuous optimizer."""
import time
import numpy as np

N = 32
RNG = np.random.default_rng(20240611)

# ---------------------------------------------------------------------------
# Objective helpers (differentiable)
# ---------------------------------------------------------------------------
def _eval(z, lam, rep):
    z = z.reshape(-1, 3)
    x = z[:, 0]
    y = z[:, 1]
    r = z[:, 2]
    n = x.shape[0]
    obj = np.sum(r)
    g = np.zeros_like(z)
    g[:, 2] = 1.0

    # boundary: keep circles inside [0,1]^2
    for i in range(n):
        if x[i] < r[i]:
            d = x[i] - r[i]
            obj += lam * d * d
            g[i, 0] += 2.0 * lam * d
            g[i, 2] += -2.0 * lam * d
        if x[i] > 1.0 - r[i]:
            d = 1.0 - r[i] - x[i]
            obj += lam * d * d
            g[i, 0] += -2.0 * lam * d
            g[i, 2] += 2.0 * lam * d
        if y[i] < r[i]:
            d = y[i] - r[i]
            obj += lam * d * d
            g[i, 1] += 2.0 * lam * d
            g[i, 2] += -2.0 * lam * d
        if y[i] > 1.0 - r[i]:
            d = 1.0 - r[i] - y[i]
            obj += lam * d * d
            g[i, 1] += -2.0 * lam * d
            g[i, 2] += 2.0 * lam * d

    # pairwise overlap + repulsion
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-12
            rsum = r[i] + r[j]
            if dist < rsum:
                d = dist - rsum
                obj += lam * d * d
                g[i, 0] += 2.0 * lam * d * (dx / dist)
                g[i, 1] += 2.0 * lam * d * (dy / dist)
                g[i, 2] += -2.0 * lam * d
                g[j, 0] += -2.0 * lam * d * (dx / dist)
                g[j, 1] += -2.0 * lam * d * (dy / dist)
                g[j, 2] += -2.0 * lam * d
            # soft repulsion to spread circles (helps grow small ones)
            if rep > 0.0 and dist < 0.6:
                f = rep * (0.6 - dist)
                g[i, 0] += f * (dx / dist)
                g[i, 1] += f * (dy / dist)
                g[j, 0] += -f * (dx / dist)
                g[j, 1] += -f * (dy / dist)

    return -obj, g.ravel()


def _valid(z):
    z = z.reshape(-1, 3)
    x = z[:, 0]
    y = z[:, 1]
    r = z[:, 2]
    if np.any(r < 0):
        return False
    if np.any(x < -1e-9) or np.any(x > 1.0 + 1e-9) or np.any(y < -1e-9) or np.any(y > 1.0 + 1e-9):
        return False
    if np.any(r > x + 1e-9) or np.any(r > 1.0 - x + 1e-9) or np.any(r > y + 1e-9) or np.any(r > 1.0 - y + 1e-9):
        return False
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if np.hypot(x[i] - x[j], y[i] - y[j]) < r[i] + r[j] - 1e-9:
                return False
    return True


def _grid_init(seed):
    rng = np.random.default_rng(seed)
    side = 6
    xs = np.linspace(0.5 / side, 1.0 - 0.5 / side, side)
    ys = np.linspace(0.5 / side, 1.0 - 0.5 / side, side)
    base = 0.5 / side
    z = []
    for i in range(side):
        for j in range(side):
            if len(z) >= N:
                break
            r = base * (0.9 + 0.2 * rng.random())
            z.append([xs[i] + 0.02 * (rng.random() - 0.5),
                      ys[j] + 0.02 * (rng.random() - 0.5),
                      r])
    return np.array(z[:N])


def _random_init(seed):
    rng = np.random.default_rng(seed)
    z = np.array([[rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8),
                   rng.uniform(0.05, 0.15)] for _ in range(N)])
    return z


def _optimize(z0, deadline):
    from scipy.optimize import minimize
    z = z0.copy()
    best = z.copy()
    best_val = _eval(z, 5.0, 0.0)[0]
    lam = 5.0
    rep = 0.0
    while time.time() < deadline:
        res = minimize(_eval, z, args=(lam, rep), method='L-BFGS-B',
                       jac=True,
                       bounds=[(0, 1), (0, 1), (0, 0.5)] * N,
                       options={'maxiter': 120, 'gtol': 1e-9, 'maxls': 40})
        if res.x is not None:
            val = _eval(res.x, lam, rep)[0]
            if val < best_val:
                best_val = val
                best = res.x.copy()
            z = res.x.copy()
        if time.time() >= deadline:
            break
        lam *= 2.5
        rep = 0.03
    return best, best_val


def construct():
    deadline = time.time() + 115.0   # safety margin vs 120s limit

    # guaranteed-valid fallback
    side = 6
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    fallback = np.array(cells[:N])

    starts = [_grid_init(0)]
    for s in range(1, 8):
        starts.append(_random_init(1000 + s))

    best = fallback
    best_sum = np.sum(fallback[:, 2])

    for k, z0 in enumerate(starts):
        if time.time() >= deadline:
            break
        z, val = _optimize(z0, deadline)
        z = z.reshape(-1, 3)
        if _valid(z):
            s = np.sum(z[:, 2])
            if s > best_sum:
                best_sum = s
                best = z.copy()

    # final safety: ensure returned packing is valid
    if not _valid(best):
        best = fallback
    return best
