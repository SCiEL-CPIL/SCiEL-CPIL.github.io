# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class _Timeout(Exception):
    pass


def _P(support, weights):
    """Exact P[X1+X2+X3 < 2 X4] for i.i.d. draws from the discrete measure.
    O(n^2 log n). Returns a value in [0, 1]."""
    s = np.asarray(support, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    n = s.size
    if n == 0:
        return 0.0
    total = w.sum()
    if total <= 1e-300:
        return 0.0
    p = w / total
    # pairwise sums and their weights
    PS = s[:, None] + s[None, :]          # (n, n)
    PW = p[:, None] * p[None, :]          # (n, n)
    ps_flat = PS.ravel()
    pw_flat = PW.ravel()
    order = np.argsort(ps_flat, kind="stable")
    ps_sorted = ps_flat[order]
    prefix = np.empty(ps_sorted.size + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(pw_sorted[order], out=prefix[1:]) if False else None
    prefix[1:] = np.cumsum(pw_sorted)
    # thresholds 2*s_l - s_k over (k, l)
    thresh = 2.0 * s[None, :] - s[:, None]   # [k, l]
    counts = np.searchsorted(ps_sorted, thresh.ravel(), side="left")
    inner = prefix[counts]
    wkwl = (p[:, None] * p[None, :]).ravel()
    return float(np.sum(wkwl * inner))


def _nm(objective, x0, deadline, maxiter=3000):
    """Nelder-Mead minimization with a wall-clock guard."""
    def cb(_xk):
        if time.time() > deadline:
            raise _Timeout()
    try:
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-9, "fatol": 1e-12},
                       callback=cb)
        return np.asarray(res.x, dtype=np.float64)
    except _Timeout:
        return None


def _free_weights(x):
    w = np.exp(x - x.max())
    return w


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start = time.time()
    deadline = start + 540.0          # safety margin under the 630 s kill
    rng = np.random.default_rng(12345)

    def time_left():
        return deadline - time.time()

    # ---- baseline / fallback: two-point {0, L} optimal ----
    def two_point_P(q):
        # q = mass at L, (1-q) at 0
        return q * (1.0 - q) ** 2 * (1.0 + 2.0 * q)
    grid = np.linspace(0.05, 0.95, 200)
    q_best = grid[np.argmax([two_point_P(q) for q in grid])]
    best_s = np.array([0.0, 1.0], dtype=np.float64)
    best_w = np.array([1.0 - q_best, q_best], dtype=np.float64)
    best_P = two_point_P(q_best)

    def consider(s, w):
        nonlocal best_s, best_w, best_P
        s = np.asarray(s, dtype=np.float64).ravel()
        w = np.asarray(w, dtype=np.float64).ravel()
        if s.size != w.size or s.size == 0:
            return
        if np.any(np.isnan(s)) or np.any(np.isnan(w)):
            return
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return
        if w.sum() <= 1e-9:
            return
        P = _P(s, w)
        if P > best_P:
            best_P = P
            best_s = s.copy()
            best_w = w.copy()

    # ---- free-weight optimization on fixed grids ----
    def run_free(support, n_restarts, maxiter):
        support = np.asarray(support, dtype=np.float64).ravel()
        n = support.size
        def obj(x):
            return -_P(support, _free_weights(x))
        starts = [np.zeros(n)]
        # a few structured + random starts
        starts.append(np.concatenate([np.array([3.0]), np.full(n - 1, -1.0)]))
        for _ in range(min(n_restarts, 6)):
            starts.append(rng.normal(0.0, 1.5, size=n))
        for x0 in starts:
            if time_left() < 8.0:
                return
            xopt = _nm(obj, x0, deadline, maxiter=maxiter)
            if xopt is not None:
                consider(support, _free_weights(xopt))

    uniform_grids = [np.arange(m, dtype=np.float64) for m in (8, 12, 16)]
    geo_grids = [np.concatenate([[0.0], 2.0 ** np.arange(1, m)]) for m in (8, 12, 16)]
    for g in uniform_grids + geo_grids:
        if time_left() < 25.0:
            break
        run_free(g, n_restarts=6, maxiter=4000)

    # ---- parametric geometric family (s_i = exp(a + (i-1) r), w_i = exp(-alpha (i/(n-1))^gamma)) ----
    def param_decode(x):
        n = int(np.clip(x[4], 4, 40))
        a, r, alpha, gamma = x[0], x[1], x[2], x[3]
        s = np.zeros(n, dtype=np.float64)
        if n > 1:
            s[1:] = np.exp(a + (np.arange(1, n)) * r)
        idx = np.arange(n, dtype=np.float64)
        t = idx / max(n - 1, 1)
        w = np.exp(-alpha * np.power(t, gamma))
        return s, w

    def param_obj(x):
        s, w = param_decode(x)
        return -_P(s, w)

    param_starts = [
        np.array([0.0, 0.0, 2.0, 1.0, 12.0]),
        np.array([0.5, 0.3, 4.0, 1.5, 16.0]),
        np.array([-1.0, 0.5, 1.0, 2.0, 20.0]),
    ]
    for _ in range(8):
        param_starts.append(rng.normal([0.0, 0.2, 2.0, 1.2, 14.0],
                                       [1.0, 0.5, 1.5, 0.8, 6.0]))
    for x0 in param_starts:
        if time_left() < 10.0:
            break
        xopt = _nm(param_obj, x0, deadline, maxiter=2500)
        if xopt is not None:
            consider(*param_decode(xopt))

    # ---- grow-and-polish the best solution on a finer grid ----
    if time_left() > 60.0:
        s0 = np.sort(best_s)
        # build a finer candidate grid: keep original points + midpoints, cap size
        extra = (s0[:-1] + s0[1:]) / 2.0
        cand = np.unique(np.concatenate([s0, extra]))
        if cand.size > 32:
            # thin to 32 evenly by index
            keep = np.linspace(0, cand.size - 1, 32).round().astype(int)
            cand = np.unique(cand[keep])
        if cand.size >= 2:
            run_free(cand, n_restarts=4, maxiter=5000)

    # ---- final validity cleanup + normalization ----
    s = np.clip(best_s, 0.0, None)
    w = np.clip(best_w, 0.0, None)
    if w.sum() <= 1e-9:
        s = np.array([0.0], dtype=np.float64)
        w = np.array([1.0], dtype=np.float64)
    w = w / w.sum()
    return s, w


# EVOLVE-BLOCK-END
