"""Autocorrelation-inequality lower-bound maximizer (fresh design)."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 50
_DEADLINE = 140.0  # safety margin under the 150s wall-clock kill
_t0 = time.time()

# Precompute constants (independent of h)
W = 1.0 / 51.0          # spacing w = 1/(N+1)
K = W / 3.0
C = 1.0 / (N + 1)       # 1/(m+1) for norm_1
# P: g[j] = sum_i h[i] h[i+j], j=0..N-1
P = [np.arange(0, N - j) for j in range(N)]
P = [np.asarray(p, dtype=np.float64) for p in P]
# Q: y[i] = g[i-1] for i=1..N, else 0
Q = [np.zeros(N) for _ in range(N + 2)]
for i in range(1, N + 1):
    Q[i][i - 1] = 1.0
Q = [np.asarray(q, dtype=np.float64) for q in Q]


def _elapsed():
    return time.time() - _t0


def compute_value(h):
    h = np.asarray(h, dtype=np.float64).ravel()
    if h.size != N or not np.all(np.isfinite(h)) or np.any(h < 0):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m + 1] = g
    l2 = K * np.sum(y * y + y * np.roll(y, -1) + np.roll(y, -1) * np.roll(y, -1))
    norm_1 = np.sum(np.abs(g)) * C
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return float(l2 / (norm_1 * norm_inf))


def _l2(h):
    y = np.array([Q[i] @ h for i in range(N + 2)])
    return K * float(np.dot(y, y) + 2.0 * float(np.dot(y, y[1:])))


def _max_g(h):
    return float(np.max([P[j] @ h for j in range(N)]))


def _neg_value(h):
    l2 = _l2(h)
    s = float(np.sum(h))
    mg = _max_g(h)
    if s <= 0 or mg <= 0:
        return 0.0
    return -(l2 / (s * s * mg))


def _neg_value_grad(h):
    s = float(np.sum(h))
    g = np.array([P[j] @ h for j in range(N)])
    mg = float(np.max(g))
    if s <= 0 or mg <= 0:
        return 0.0, np.zeros(N)
    y = np.array([Q[i] @ h for i in range(N + 2)])
    d_l2 = 2.0 * K * (y + np.roll(y, 1) + np.roll(y, -1))
    g_l2 = np.array([Q[i] @ d_l2 for i in range(N)])
    val = l2 = K * float(np.dot(y, y) + 2.0 * float(np.dot(y, y[1:])))
    # value = l2 / (s^2 * mg);  -dvalue = -(1/(s^2 mg)) d l2 + l2 * d(1/(s^2 mg))
    # d(1/(s^2 mg)) = -(2 s mg + s^2 dm_g)/(s^2 mg)^2 * ... = -(2/(s mg) + dm_g/mg)/(s mg)
    dm_g = np.zeros(N)
    for j in range(N):
        if g[j] >= mg - 1e-12:
            dm_g += P[j] / float(np.sum(np.abs(g[j] - mg) < 1e-12))
    d_inv = -(2.0 / (s * mg) + float(dm_g @ (np.ones(N))) / mg) / (s * mg)
    # careful: dm_g @ 1 = sum of active P[j] summed
    d_inv = -(2.0 / (s * mg) + float(np.sum(dm_g)) / mg) / (s * mg)
    grad_val = (1.0 / (s * s * mg)) * g_l2 + l2 * d_inv * np.ones(N)
    return -val, -grad_val


def _smooth_neg(h, eps=1e-6):
    s = float(np.sum(h))
    g = np.array([P[j] @ h for j in range(N)])
    mg = float(np.max(g))
    if s <= 0 or mg <= 0:
        return 0.0
    l2 = _l2(h)
    return -(l2 / (s * s * (mg + eps)))


def _smooth_grad(h, eps=1e-6):
    s = float(np.sum(h))
    g = np.array([P[j] @ h for j in range(N)])
    mg = float(np.max(g))
    if s <= 0 or mg <= 0:
        return 0.0, np.zeros(N)
    y = np.array([Q[i] @ h for i in range(N + 2)])
    d_l2 = 2.0 * K * (y + np.roll(y, 1) + np.roll(y, -1))
    g_l2 = np.array([Q[i] @ d_l2 for i in range(N)])
    l2 = K * float(np.dot(y, y) + 2.0 * float(np.dot(y, y[1:])))
    sm = mg + eps
    # value = l2 / (s^2 * sm)
    grad_val = (1.0 / (s * s * sm)) * g_l2 + l2 * (-(2.0 / (s * sm)) / (s * sm)) * np.ones(N)
    return -l2 / (s * s * sm), -grad_val


def _polish(h, eps=1e-6):
    h = np.asarray(h, dtype=np.float64).ravel().copy()
    try:
        res = minimize(_smooth_neg, h, args=(eps,), method='L-BFGS-B',
                       jac=_smooth_grad, bounds=[(0.0, None)] * N,
                       options={'maxiter': 400, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
        if res.success or res.fun < _neg_value(res.x):
            h = np.clip(res.x, 0.0, None)
    except Exception:
        pass
    return h


def _exact(h):
    h = np.asarray(h, dtype=np.float64).ravel().copy()
    try:
        res = minimize(_neg_value, h, method='L-BFGS-B', jac=_neg_value_grad,
                       bounds=[(0.0, None)] * N,
                       options={'maxiter': 300, 'maxfun': 4000, 'ftol': 1e-14, 'gtol': 1e-12})
        if res.success or _neg_value(res.x) < _neg_value(h):
            h = np.clip(res.x, 0.0, None)
    except Exception:
        pass
    return h


def _perturb_local(h, rng, scale=0.05):
    base = compute_value(h)
    h = np.asarray(h, dtype=np.float64).ravel().copy()
    for _ in range(200):
        if _elapsed() > _DEADLINE - 5:
            break
        step = scale * rng.standard_normal(N)
        cand = np.clip(h + step, 0.0, None)
        v = compute_value(cand)
        if v > base + 1e-12:
            base = v
            h = cand
        scale *= 0.97
    return h


def _flat_max_search(h, rng, scale=0.05, iters=300):
    """Directly push to reduce the max lag while keeping l2 up."""
    base = compute_value(h)
    h = np.asarray(h, dtype=np.float64).ravel().copy()
    for _ in range(iters):
        if _elapsed() > _DEADLINE - 5:
            break
        g = np.array([P[j] @ h for j in range(N)])
        mg = float(np.max(g))
        if mg <= 0:
            break
        # gradient of max g w.r.t. h
        dmax = np.zeros(N)
        act = np.where(np.abs(g - mg) < 1e-9)[0]
        for j in act:
            dmax += P[j]
        dmax /= float(len(act))
        # direction: reduce max g, keep sum roughly constant (project)
        s = float(np.sum(h))
        if s <= 0:
            break
        dmax = dmax - np.mean(dmax)  # keep sum ~ constant
        step = scale * dmax / (np.linalg.norm(dmax) + 1e-12)
        cand = np.clip(h - step, 0.0, None)
        v = compute_value(cand)
        if v > base + 1e-12:
            base = v
            h = cand
        scale *= 0.99
    return h


def _random_starts(rng, count):
    starts = []
    for _ in range(count):
        if _elapsed() > _DEADLINE - 5:
            break
        r = rng.random()
        if r < 0.3:
            h = np.ones(N)
        elif r < 0.5:
            h = np.ones(N) * (0.5 + 0.5 * rng.random())
        elif r < 0.65:
            h = np.exp(-((np.arange(N) - N / 2) ** 2) / (2.0 * (N / 4.0) ** 2))
            h = h / np.max(h)
        elif r < 0.8:
            # broad flat with small random ripple
            h = np.ones(N) + 0.2 * rng.standard_normal(N)
            h = np.clip(h, 0.1, None)
        else:
            # random smooth via random frequencies
            f = rng.integers(1, 4, size=3)
            ph = rng.uniform(0, 2 * np.pi, size=3)
            h = 1.0 + 0.3 * np.sin(2 * np.pi * f[0] * np.arange(N) / N + ph[0]) \
                + 0.2 * np.cos(2 * np.pi * f[1] * np.arange(N) / N + ph[1])
            h = np.clip(h, 0.0, None)
            if np.max(h) > 0:
                h = h / np.max(h)
        h = np.clip(h, 0.0, None)
        starts.append(h)
    return starts


def construct():
    rng = np.random.default_rng(12345)
    best_h = np.ones(N, dtype=np.float64)
    best_v = compute_value(best_h)

    def consider(h):
        nonlocal best_h, best_v
        h = np.clip(np.asarray(h, dtype=np.float64).ravel(), 0.0, None)
        v = compute_value(h)
        if v > best_v + 1e-12:
            best_v = v
            best_h = h.copy()

    # 1) random restarts with smooth polish + exact refine
    starts = _random_starts(rng, 12)
    for h0 in starts:
        if _elapsed() > _DEADLINE - 20:
            break
        hp = _polish(h0, eps=1e-5)
        consider(hp)
        he = _exact(hp)
        consider(he)

    # 2) flat-max directed search from best
    if _elapsed() < _DEADLINE - 20:
        consider(_flat_max_search(best_h, rng))

    # 3) perturbation-based local refinement from best
    if _elapsed() < _DEADLINE - 10:
        consider(_perturb_local(best_h, rng, scale=0.05))

    # 4) final exact polish from best
    if _elapsed() < _DEADLINE - 3:
        consider(_exact(best_h))

    # 5) one more random restart + polish if time
    if _elapsed() < _DEADLINE - 10:
        h0 = np.ones(N) + 0.15 * rng.standard_normal(N)
        h0 = np.clip(h0, 0.0, None)
        consider(_exact(_polish(h0, eps=1e-6)))

    return [float(x) for x in best_h]


if __name__ == "__main__":
    print(compute_value(construct()))
