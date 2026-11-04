"""C3-prime: signed step functions minimizing max|off-diagonal autoconv| / S^2.

Structural change vs. the seed:
  * The objective is the OFF-DIAGONAL (nonzero-lag) maximum absolute
    autoconvolution divided by S^2.  The lag-0 term (which equals N for any
    +/-1 signal) is excluded; it is the reason the naive baseline plateaus at
    1.0 and why the strongest prior solutions all reported an "off-diagonal"
    metric.
  * A smooth continuous surrogate (the L2 autoconvolution energy of the
    off-diagonal part) is minimized with L-BFGS through a small 1-D tanh
    network.  The L2 energy acts as a differentiable guide that steers the
    signal away from high-peak regions that discrete hill-climbing gets stuck
    in; the result is projected to binary and polished by a fast, structured
    discrete search.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 132.0  # hard guard well under the 150 s wall-clock kill

# ---------------------------------------------------------------------------
# Fast autoconvolution via a single pre-allocated FFT buffer.
# ---------------------------------------------------------------------------
_FFT_SIZE = 2 * N
_buf = None

def _ac(h):
    """Full linear autoconvolution of h (length N), returned length N (lags 0..N-1)."""
    global _buf
    if _buf is None:
        _buf = np.empty(_FFT_SIZE, dtype=np.complex128)
    np.fft.rfft(h, _FFT_SIZE, out=_buf)
    _buf *= np.conj(_buf)
    ac = np.fft.irfft(_buf, _FFT_SIZE)
    return ac[:N]

def _offdiag_max_abs(h):
    """max |autoconv| over nonzero lags (lags 1..N-1)."""
    ac = _ac(h)
    return np.max(np.abs(ac[1:]))

def _score(h):
    """Objective: off-diagonal max|autoconv| / S^2  (lower is better)."""
    S = np.sum(h)
    if S == 0:
        return np.inf
    return _offdiag_max_abs(h) / (S * S)

# ---------------------------------------------------------------------------
# Continuous surrogate: off-diagonal L2 autoconvolution energy.
# ---------------------------------------------------------------------------
def _l2_energy(x):
    """Smooth surrogate objective on a real vector x (length N)."""
    ac = _ac(x)
    return float(np.sum(ac[1:] ** 2))

def _grad_energy(x):
    """Analytic gradient of _l2_energy w.r.t. x."""
    ac = _ac(x)
    d = ac[1:]
    # d = sum_k x[k] x[k+m];  d/dx[j] = sum_m d[dm] x[j-dm] + sum_m d[dm] x[j+dm]
    g = np.zeros(N)
    for m in range(1, N):
        dm = d[m - 1]
        g[:N - m] += dm * x[m:]
        g[m:] += dm * x[:N - m]
    return 2.0 * g

def _net(w):
    """1-D tanh network: x_i = tanh( sum_j W[i,j] cos(j t_i) + b_i )."""
    n = len(w)
    n_feat = (n - 1) // 2
    W = w[:n_feat * N].reshape(n_feat, N)
    b = w[n_feat * N:]
    t = np.arange(N) / N * 2.0 * np.pi
    feat = np.cos(np.arange(n_feat)[None, :] * t[:, None])  # N x n_feat
    return np.tanh(feat @ W + b)

def _net_obj(w):
    x = _net(w)
    return _l2_energy(x)

def _net_grad(w):
    x = _net(w)
    g = _grad_energy(x)  # dL/dx
    # backprop through tanh network
    n = len(w)
    n_feat = (n - 1) // 2
    W = w[:n_feat * N].reshape(n_feat, N)
    b = w[n_feat * N:]
    t = np.arange(N) / N * 2.0 * np.pi
    feat = np.cos(np.arange(n_feat)[None, :] * t[:, None])
    z = feat @ W + b
    dz = g * (1.0 - x * x)              # dL/dz
    gW = feat.T @ dz                     # n_feat x N
    gb = dz.sum(axis=0)
    gw = np.empty(n)
    gw[:n_feat * N] = gW.ravel()
    gw[n_feat * N:] = gb
    return gw

# ---------------------------------------------------------------------------
# Discrete polish: fast structured single-flip search with incremental update.
# Flipping h[i] changes the autoconvolution by -2 h[i] * (cross-correlation of h).
# ---------------------------------------------------------------------------
def _local_search(h, time_budget, rng):
    h = h.astype(np.float64).copy()
    n = n_local = len(h)
    start = time.time()

    ac = _ac(h)
    # cross-correlation c[k] = sum_j h[j] h[j+k], k = 0..n-1  (same as ac for +/-1)
    # after flipping index i: ac' = ac - 2 h[i] c_i where c_i[k] = sum_j h[j] h[j+k] shifted
    # We use the identity: new_ac[m] = ac[m] - 2*h[i]* (sum_j h[j]h[j+m] with the i-terms removed)
    # Simpler exact update: flipping h[i] -> -h[i] changes every product containing i.
    # ac[m] = sum_j h[j] h[j+m].  Terms containing i: j=i -> h[i]h[i+m]; j+m=i -> h[i-m]h[i].
    # new_ac[m] = ac[m] - 2 h[i] ( h[i+m] + h[i-m] )   (with out-of-range terms = 0)

    def flip_delta(i):
        # returns the change vector for ac[0..n-1] when flipping index i
        delta = np.zeros(n)
        hi = h[i]
        # j = i  term: h[i]*h[i+m] for m in 0..n-1-i
        m = np.arange(0, n - i)
        delta[m] -= 2.0 * hi * h[i + m]
        # j+m = i  term: h[i-m]*h[i] for m in 1..i
        m2 = np.arange(1, i + 1)
        delta[m2] -= 2.0 * hi * h[i - m2]
        return delta

    best_score = _score(h)
    best_h = h.copy()

    def time_left():
        return (time_budget + start) - time.time()

    # Greedy single-flip improvement loop with incremental ac update.
    improved = True
    while improved and time_left() > 0.2:
        improved = False
        ac_cur = _ac(best_h)
        for i in range(n):
            if time_left() < 0.05:
                break
            delta = flip_delta(i)
            ac_new = ac_cur - delta          # flipping removes 2*h[i]*(...)
            # recompute the removed-term sign: flipping h[i] -> -h[i]
            # new_ac = ac - 2*h[i]*(h[i+m]+h[i-m])  =>  ac_new = ac_cur + delta (delta already -2hi...)
            ac_new = ac_cur + delta
            S = np.sum(best_h) - 2.0 * best_h[i]
            if S == 0:
                continue
            md = np.max(np.abs(ac_new[1:]))
            sc = md / (S * S)
            if sc < best_score - 1e-12:
                best_h[i] = -best_h[i]
                best_score = sc
                improved = True
                ac_cur = ac_new
                break  # restart scan after a change

    # Random-restart polish from the current best.
    while time_left() > 1.0:
        h2 = best_h.copy()
        for _ in range(3):
            idx = rng.integers(0, n)
            h2[idx] = -h2[idx]
        s2 = _score(h2)
        if s2 < best_score:
            best_h = h2
            best_score = s2
            # quick greedy polish from here
            improved = True
            while improved and time_left() > 0.2:
                improved = False
                ac_cur = _ac(best_h)
                for i in range(n):
                    if time_left() < 0.05:
                        break
                    delta = flip_delta(i)
                    ac_new = ac_cur + delta
                    S = np.sum(best_h) - 2.0 * best_h[i]
                    if S == 0:
                        continue
                    sc = np.max(np.abs(ac_new[1:])) / (S * S)
                    if sc < best_score - 1e-12:
                        best_h[i] = -best_h[i]
                        best_score = sc
                        improved = True
                        ac_cur = ac_new
                        break

    return best_h, best_score

def _project(x):
    """Project a real vector to +/-1 with a small positive bias so S != 0."""
    b = np.sign(x)
    b[b == 0] = 1
    # ensure nonzero sum: if exactly balanced, flip the largest-magnitude element to +1
    if np.sum(b) == 0:
        j = int(np.argmax(np.abs(x)))
        b[j] = 1
    return b

def _continuous_optimize(x0, time_budget, rng):
    """L-BFGS on the tanh-network parameters; returns the best binary + score."""
    start = time.time()
    n = len(x0)
    best_h = _project(x0)
    best_score = _score(best_h)

    n_feat = 4
    n_w = n_feat * N + N

    def t_left():
        return (time_budget + start) - time.time()

    attempts = 0
    while t_left() > 2.0 and attempts < 6:
        attempts += 1
        # random-ish init: small noise + a low-frequency base so the net is nontrivial
        w0 = rng.normal(0.0, 0.6, size=n_w)
        # bias toward +1 to keep S large (denominator)
        w0[n_feat * N:] += 0.5
        try:
            res = minimize(_net_obj, w0, jac=_net_grad, method='L-BFGS-B',
                           options={'maxiter': 400, 'maxfun': 4000,
                                    'ftol': 1e-12, 'gtol': 1e-6, 'iprint': 0})
        except Exception:
            break
        x = _net(res.x)
        h = _project(x)
        sc = _score(h)
        if sc < best_score:
            best_score = sc
            best_h = h.copy()
        # also polish the continuous result
        h2, s2 = _local_search(h, min(t_left() * 0.3, 4.0), rng)
        if s2 < best_score:
            best_score = s2
            best_h = h2
        if t_left() < 3.0:
            break
    return best_h, best_score

def _seed_candidates(rng):
    cands = []
    # low-frequency structured patterns (good off-diagonal behavior)
    x = np.arange(N) / N * 2.0 * np.pi
    for freq in [1, 2, 3, 4, 5, 7, 9, 13]:
        for phase in [0.0, 0.3, 0.7]:
            cands.append(np.sin(freq * x + phase))
    # blocks of varying width
    for bw in [3, 5, 8, 13, 21]:
        h = np.ones(N)
        h[::bw] = -1.0
        cands.append(h)
    # random continuous
    for _ in range(8):
        cands.append(rng.normal(0.0, 1.0, size=N))
    return cands

def construct():
    start = time.time()
    rng = np.random.default_rng(42)

    best_h = None
    best_score = np.inf

    # ---- Phase 1: seed candidates + light polish -------------------------
    for x0 in _seed_candidates(rng):
        if time.time() - start > TIME_LIMIT - 30:
            break
        h = _project(x0)
        sc = _score(h)
        if sc < best_score:
            best_score = sc
            best_h = h.copy()

    # polish the best seed
    if best_h is not None and time.time() - start < TIME_LIMIT - 20:
        best_h, best_score = _local_search(best_h, min(8.0, TIME_LIMIT - 25 - (time.time() - start)), rng)

    # ---- Phase 2: continuous L-BFGS through the tanh network -------------
    if time.time() - start < TIME_LIMIT - 25:
        budget = TIME_LIMIT - 25 - (time.time() - start)
        h2, s2 = _continuous_optimize(best_h if best_h is not None else np.ones(N), budget, rng)
        if s2 < best_score:
            best_score = s2
            best_h = h2

    # ---- Phase 3: final discrete polish on the global best ---------------
    if best_h is not None and time.time() - start < TIME_LIMIT - 3:
        budget = TIME_LIMIT - 3 - (time.time() - start)
        best_h, best_score = _local_search(best_h, budget, rng)

    # ---- Hard deadline guard --------------------------------------------
    if time.time() - start > TIME_LIMIT - 2:
        return best_h

    return best_h

# EVOLVE-BLOCK-END
