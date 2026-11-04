"""Second autocorrelation inequality -- soft-max model-and-optimize rewrite.

Discrete setting: 50 nonnegative step heights on equal intervals of [-1/4, 1/4],
dh = 1/20.  Discrete autoconvolution sampled on the same grid:
    ac_i = dh * sum_{j+k = i + OFFSET} h_j h_k,   OFFSET = 10.

Objective (scale-invariant):  R = ||h||_2^2 / (||h||_1 * ||ac||_inf).

We model h = max(0, B @ a) with a smooth Gaussian-bump basis (3M parameters) and
optimize log R with a SMOOTH soft-max surrogate for log ||ac||_inf (log-sum-exp),
which makes the whole objective differentiable.  The analytic gradient of the
soft-max objective is used with L-BFGS-B from many diverse random starts.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # (1/2) / DH

_IDX = np.arange(N, dtype=float)
_JK = np.arange(2 * N - 1)  # full-convolution index j+k, length 2N-1

_DEADLINE = 150.0
_SAFETY = 15.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


# --------------------------------------------------------------------------- #
#  Basis (model)
# --------------------------------------------------------------------------- #
def _make_basis():
    M = 12
    centers = np.linspace(2.0, N - 3.0, M)
    widths = [1.7, 3.3, 6.0]
    basis = []
    for c in centers:
        for w in widths:
            basis.append(np.exp(-0.5 * ((_IDX - c) / w) ** 2))
    B = np.stack(basis, axis=1)  # (N, P)
    return B


B = _make_basis()
P = B.shape[1]


def _forward(params):
    h = B @ params
    np.maximum(h, 0.0, out=h)
    return h


def _ratio_from_h(h):
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return 0.0
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    acmax = float(np.max(ac))
    if acmax <= 1e-300:
        return 0.0
    return l2sq / (l1 * acmax)


# --------------------------------------------------------------------------- #
#  Smooth objective + analytic gradient (soft-max surrogate)
# --------------------------------------------------------------------------- #
def _logR(params, beta=80.0):
    h = _forward(params)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return -1e30
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    amax = float(np.max(ac))
    if amax <= 1e-300:
        return -1e30
    S = float(np.sum(np.exp(beta * (ac - amax))))
    log_ac = amax + math.log(S) / beta
    R = l2sq / (l1 * math.exp(log_ac))
    if R <= 1e-300:
        return -1e30
    return math.log(R)


def _logR_and_grad(params, beta=80.0):
    h_raw = B @ params
    pos = h_raw > 0.0
    h = np.where(pos, h_raw, 0.0)

    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return -1e30, np.zeros_like(params)

    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    amax = float(np.max(ac))
    if amax <= 1e-300:
        return -1e30, np.zeros_like(params)

    # soft-max weights over the N autoconvolution values
    w = np.exp(beta * (ac - amax))
    S = float(np.sum(w))
    if S <= 1e-300:
        return -1e30, np.zeros_like(params)
    wp = w / S  # d log_ac / d ac_i

    # d log_ac / d h_i = sum_j wp_j * d ac_j / d h_i
    # d ac_j / d h_i = 2 DH h_{j-i} (j-i in [0,N))
    #   => d ac_j / d h_i = 2 DH * h[ j - i ]  for j in [0, 2N-2]
    # vectorize: for each i, sum over j of wp[j] * 2DH * h[j-i] where valid.
    # Build via: for offset t = j - i in [-(N-1), N-1], contribution 2DH*h[t] if 0<=t<N.
    # dlog_ac_dh[i] = 2DH * sum_{t:0<=t<N} wp[i + t] * h[t]
    # i in [0,N), t in [0,N) -> index i+t in [0, 2N-2].
    # Use matrix: M[i,t] = wp[i+t].  Build with broadcasting.
    ii = np.arange(N)[:, None]
    tt = np.arange(N)[None, :]
    idx = ii + tt  # (N,N)
    M = wp[idx] * (2.0 * DH)  # (N,N)
    dlog_ac_dh = M @ h  # (N,)

    dlogR_dh = (2.0 * h / l2sq) - (1.0 / l1) - dlog_ac_dh

    dlogR_dh_active = np.where(pos, dlogR_dh, 0.0)
    grad = B.T @ dlogR_dh_active

    if not np.all(np.isfinite(grad)):
        grad = np.zeros_like(params)
        eps = 1e-6
        for p in range(P):
            pp = params.copy(); pp[p] += eps
            pm = params.copy(); pm[p] -= eps
            grad[p] = (_logR(pp) - _logR(pm)) / (2.0 * eps)

    R = l2sq / (l1 * math.exp(amax + math.log(S) / beta))
    if R <= 1e-300:
        return -1e30, grad
    return math.log(R), grad


def _neg_logR(params):
    return -_logR(params)


def _neg_logR_jac(params):
    v, g = _logR_and_grad(params)
    return v, -g


# --------------------------------------------------------------------------- #
#  Local polish
# --------------------------------------------------------------------------- #
def _polish(params0, budget):
    best_p = np.array(params0, dtype=float)
    best_v = _logR(best_p)
    t0 = time.time()
    bounds = [(None, None)] * P
    try:
        res = minimize(
            _neg_logR, best_p,
            method='L-BFGS-B',
            jac=_neg_logR_jac,
            bounds=bounds,
            options={'maxiter': 250, 'maxfun': 2500, 'ftol': 1e-12,
                     'gtol': 1e-9, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > budget else x),
        )
        if res.x is not None:
            rv = _logR(res.x)
            if rv > best_v:
                best_v = rv
                best_p = res.x.copy()
    except Exception:
        pass
    return best_p, best_v, math.exp(best_v) if best_v > -1e29 else 0.0


# --------------------------------------------------------------------------- #
#  Global starts
# --------------------------------------------------------------------------- #
def _global_starts():
    rng = np.random.default_rng(20240501)
    starts = []

    # (a) single dominant basis vectors
    for p in range(P):
        a = np.zeros(P); a[p] = 1.0
        starts.append(a)

    # (b) many random sign-magnitude vectors -> diverse multi-peak candidates
    for _ in range(160):
        a = rng.choice([-1.0, 1.0], size=P) * rng.uniform(0.4, 1.6)
        starts.append(a)

    # (c) gaussian random
    for _ in range(60):
        starts.append(rng.normal(0.0, 1.0, size=P))

    # (d) structured: alternating blocks of signs
    for _ in range(40):
        a = np.zeros(P)
        block = rng.integers(2, 6)
        sgn = 1.0
        for p in range(P):
            if (p // block) % 2 == 0:
                sgn = 1.0
            else:
                sgn = -1.0
            a[p] = sgn * rng.uniform(0.4, 1.6)
        starts.append(a)

    return starts


def construct():
    global _start
    _start = time.time()

    best_params = None
    best_log = -1e30
    best_R = 0.0

    starts = _global_starts()

    scored = []
    for a in starts:
        v = _logR(a)
        scored.append((v, a))
        if v > best_log:
            best_log = v
            best_params = a.copy()
            best_R = math.exp(v) if v > -1e29 else 0.0
    scored.sort(key=lambda t: -t[0])

    top_k = min(len(scored), 90)
    for k in range(top_k):
        v0, a = scored[k]
        if _time_left() <= 6.0:
            break
        if v0 < best_log - 2.0 and best_log > -1e29:
            continue
        budget = min(3.5, max(0.8, _time_left() - 8.0))
        try:
            p_opt, v_opt, R_opt = _polish(a, budget=budget)
            if v_opt > best_log:
                best_log = v_opt
                best_params = p_opt.copy()
                best_R = R_opt
        except Exception:
            pass
        if _time_left() <= 20.0:
            break

    # Final high-precision polish on the global best.
    if best_params is not None and _time_left() > 12.0:
        try:
            p_f, v_f, R_f = _polish(best_params, budget=max(1.0, _time_left() - 18.0))
            if v_f > best_log:
                best_log = v_f
                best_params = p_f.copy()
                best_R = R_f
        except Exception:
            pass

    if best_params is None:
        h = np.ones(N)
    else:
        h = _forward(best_params)
        if not np.all(np.isfinite(h)) or float(np.sum(h)) <= 0.0:
            h = np.ones(N)
    h = np.maximum(h, 0.0)
    return [float(x) for x in h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio_from_h(h))
    print("time =", time.time() - t0)
