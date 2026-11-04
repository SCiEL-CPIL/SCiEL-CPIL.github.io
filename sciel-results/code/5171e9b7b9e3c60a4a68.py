"""Second autocorrelation inequality -- combined model-and-optimize.

Discrete setting: 50 nonnegative step heights on equal intervals of [-1/4, 1/4],
dh = 1/20.  The discrete autoconvolution sampled on the same grid is
    (h*h)(x_i) = dh * sum_{j+k = i + OFFSET} h_j h_k,   OFFSET = 10.

Objective (scale-invariant):  R = ||h||_2^2 / (||h||_1 * ||h*h||_inf).

We MODEL h as a composition of smooth Gaussian bumps
    h(x) = max(0, sum_m a_m * exp(-0.5*((x-c_m)/w_m)^2))
with 20 parameters.  We fit/optimise the parameters directly against the
task objective using the analytic gradient of log R w.r.t. the parameters,
which keeps the search smooth and fast.  Multiple global starts (coarse grid
+ random) each get a local L-BFGS-B polish; the best is returned.

We also add a direct height-space optimization path using a smooth soft-max
surrogate for the peak term, and a DFT-based parameterization as an
additional search avenue.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

# Grid positions (cell centers / sample positions) in index space.
_IDX = np.arange(N, dtype=float)

_DEADLINE = 150.0
_SAFETY = 18.0


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


# --------------------------------------------------------------------------- #
#  Basis (model) definition
# --------------------------------------------------------------------------- #
def _make_basis():
    """Gaussian bump basis.  M centers spread over the grid, 2 widths each."""
    M = 10
    centers = np.linspace(3.0, N - 4.0, M)
    widths = [2.2, 4.5]
    basis = []
    for c in centers:
        for w in widths:
            basis.append(np.exp(-0.5 * ((_IDX - c) / w) ** 2))
    B = np.stack(basis, axis=1)  # shape (N, P)
    return B


B = _make_basis()
P = B.shape[1]  # number of parameters


def _forward(params):
    """h = max(0, B @ params)."""
    h = B @ params
    np.maximum(h, 0.0, out=h)
    return h


def _autoconv(h):
    """Compute discrete autoconvolution of h (length N) -> length 2N-1."""
    return np.convolve(h, h, mode='full')


def _ratio_from_h(h):
    """R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) from a precomputed nonneg h."""
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


def _ratio(params):
    return _ratio_from_h(_forward(params))


# --------------------------------------------------------------------------- #
#  Soft-max ratio for direct height-space optimization
# --------------------------------------------------------------------------- #
def _ratio_soft(h, beta=50.0):
    """Compute ||h||^2_2 / (||h||_1 * soft_max(h*h)) using soft-max for peak."""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    ac = DH * conv[OFFSET:OFFSET + N]
    max_val = np.max(ac)
    if max_val <= 0:
        return 0.0
    # Soft-max approximation of max using log-sum-exp
    shifted = beta * (ac - max_val)
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_max = (log_sum_exp + beta * max_val) / beta
    if soft_max <= 0:
        return 0.0
    return l2_sq / soft_max


def _logR_soft(h, beta=50.0):
    """Log of soft-max ratio."""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return -1e30
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    ac = DH * conv[OFFSET:OFFSET + N]
    max_val = np.max(ac)
    if max_val <= 0:
        return -1e30
    shifted = beta * (ac - max_val)
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    log_sum_exp = max_shifted + np.log(np.sum(exp_shifted))
    soft_max = (log_sum_exp + beta * max_val) / beta
    if soft_max <= 0:
        return -1e30
    R = l2_sq / soft_max
    if R <= 0:
        return -1e30
    return math.log(R)


def _logR_soft_grad(h, beta=50.0):
    """Analytic gradient of log soft-max ratio w.r.t. h (before max(0,.))."""
    h_pos = np.maximum(h, 0.0)
    l1 = np.sum(h_pos)
    if l1 == 0:
        return -1e30, np.zeros(N)
    h_norm = h_pos / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    ac = DH * conv[OFFSET:OFFSET + N]
    max_val = np.max(ac)
    if max_val <= 0:
        return -1e30, np.zeros(N)
    shifted = beta * (ac - max_val)
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum_exp = max_shifted + np.log(sum_exp)
    soft_max = (log_sum_exp + beta * max_val) / beta
    if soft_max <= 0:
        return -1e30, np.zeros(N)
    R = l2_sq / soft_max
    if R <= 0:
        return -1e30, np.zeros(N)
    logR = math.log(R)

    # Gradient of log R = log l2_sq - log soft_max w.r.t. h (normalized)
    # Then chain rule to h_pos, then to h
    # d log l2_sq / d h_norm_i = 2 h_norm_i / l2_sq
    # d log soft_max / d h_norm_i: need derivative of soft_max w.r.t. ac, then ac w.r.t. h_norm
    # soft_max = (log_sum_exp + beta * max_val) / beta
    # d soft_max / d ac_j = exp_shifted_j / sum_exp * (1 - beta * ac_j / soft_max + beta * ac_j / soft_max) ... 
    # Actually: soft_max = (1/beta) log sum exp(beta ac_k)
    # d soft_max / d ac_j = exp(beta ac_j) / sum exp(beta ac_k) = exp_shifted_j / sum_exp
    # d ac_j / d h_norm_i: ac_j = DH * sum_{p+q=j} h_norm_p h_norm_q
    # d ac_j / d h_norm_i = DH * sum_{q: i+q=j} 2 h_norm_q = 2 DH h_norm[j-i] if valid

    dlogR_dh_norm = np.zeros(N)
    # d log l2_sq / d h_norm
    dlogR_dh_norm += 2.0 * h_norm / l2_sq
    # d log soft_max / d h_norm = sum_j (1/soft_max) * (d soft_max/d ac_j) * (d ac_j/d h_norm)
    weights = exp_shifted / sum_exp  # d soft_max / d ac_j
    # d ac_j / d h_norm_i = 2 DH h_norm[j-i] if 0 <= j-i < N
    # So d log soft_max / d h_norm_i = (1/soft_max) * sum_j weights_j * 2 DH h_norm[j-i]
    # This is a convolution: (weights * 2 DH) convolved with h_norm, at index i
    # Specifically: sum_j weights_j * 2 DH h_norm[j-i] = 2 DH * (weights * h_norm)[i] in some sense
    # Let's compute directly:
    dlog_softmax_dh_norm = np.zeros(N)
    for i in range(N):
        s = 0.0
        for j in range(N):
            k = j - i
            if 0 <= k < N:
                s += weights[j] * 2.0 * DH * h_norm[k]
        dlog_softmax_dh_norm[i] = s / soft_max
    dlogR_dh_norm -= dlog_softmax_dh_norm

    # Chain: d logR / d h_pos = dlogR_dh_norm / l1  (since h_norm = h_pos / l1)
    # d logR / d h = dlogR_dh_pos * (1 if h>0 else 0)
    dlogR_dh_pos = dlogR_dh_norm / l1
    dlogR_dh = np.where(h_pos > 0, dlogR_dh_pos, 0.0)

    if not np.all(np.isfinite(dlogR_dh)):
        dlogR_dh = np.zeros(N)

    return logR, dlogR_dh


# --------------------------------------------------------------------------- #
#  Log R and gradient for basis parameterization
# --------------------------------------------------------------------------- #
def _logR(params):
    h = _forward(params)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e300:
        return -1e30
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    acmax = float(np.max(ac))
    if acmax <= 1e-300:
        return -1e30
    R = l2sq / (l1 * acmax)
    if R <= 1e-300:
        return -1e30
    return math.log(R)


def _logR_and_grad(params):
    """Analytic gradient of log R = log l2sq - log l1 - log acmax w.r.t. params."""
    h_raw = B @ params
    pos = h_raw > 0.0
    h = np.where(pos, h_raw, 0.0)

    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return -1e30, np.zeros_like(params)

    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    acmax = float(np.max(ac))
    if acmax <= 1e-300:
        return -1e30, np.zeros_like(params)

    R = l2sq / (l1 * acmax)
    logR = math.log(R)

    m = int(np.argmax(ac))
    dac_dh = np.zeros(N)
    j = m - int(np.arange(N))
    valid = (j >= 0) & (j < N)
    dac_dh[valid] = 2.0 * DH * h[j[valid]]

    dlogR_dh = (2.0 * h / l2sq) - (1.0 / l1) - (dac_dh / acmax)

    dlogR_dh_active = np.where(pos, dlogR_dh, 0.0)
    grad = B.T @ dlogR_dh_active

    if not np.all(np.isfinite(grad)):
        grad = np.zeros_like(params)
        eps = 1e-6
        for p in range(P):
            pp = params.copy(); pp[p] += eps
            pm = params.copy(); pm[p] -= eps
            lp = _logR(pp); lm = _logR(pm)
            grad[p] = (lp - lm) / (2.0 * eps)
    return logR, grad


def _neg_logR(params):
    v, _ = _logR_and_grad(params)
    return -v


def _neg_logR_jac(params):
    v, g = _logR_and_grad(params)
    return v, -g


# --------------------------------------------------------------------------- #
#  Optimization routines
# --------------------------------------------------------------------------- #
def _polish(params0, budget):
    """Local L-BFGS-B polish of log R w.r.t. params; return best params/value."""
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
            options={'maxiter': 300, 'maxfun': 4000, 'ftol': 1e-12,
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


def _polish_direct(h0, budget, beta=50.0):
    """Local L-BFGS-B polish on height space using soft-max surrogate."""
    h = np.maximum(h0, 0.0).copy()
    best_h = h.copy()
    best_v = _logR_soft(h, beta)
    t0 = time.time()

    def neg_obj(h_flat):
        v = _logR_soft(h_flat, beta)
        return -v

    def neg_jac(h_flat):
        v, g = _logR_soft_grad(h_flat, beta)
        return -v, -g

    try:
        res = minimize(
            neg_obj, h,
            method='L-BFGS-B',
            jac=neg_jac,
            bounds=[(0, None)] * N,
            options={'maxiter': 300, 'maxfun': 4000, 'ftol': 1e-12,
                     'gtol': 1e-9, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > budget else x),
        )
        if res.x is not None:
            h_opt = np.maximum(res.x, 0.0)
            rv = _logR_soft(h_opt, beta)
            if rv > best_v:
                best_v = rv
                best_h = h_opt.copy()
    except Exception:
        pass
    return best_h, best_v, math.exp(best_v) if best_v > -1e29 else 0.0


def _coordinate_descent(params, budget):
    """Simple cyclic coordinate descent fallback on log R w.r.t. params."""
    best_p = np.array(params, dtype=float)
    best_v = _logR(best_p)
    t0 = time.time()
    step = 0.5
    for _ in range(4000):
        if (time.time() - t0) > budget:
            break
        improved = False
        for p in range(P):
            lo = best_p[p] - step
            hi = best_p[p] + step
            for _ in range(24):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                tm1 = best_p.copy(); tm1[p] = m1
                tm2 = best_p.copy(); tm2[p] = m2
                v1 = _logR(tm1); v2 = _logR(tm2)
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
            cand = best_p.copy()
            cand[p] = (lo + hi) / 2.0
            vc = _logR(cand)
            if vc > best_v:
                best_v = vc
                best_p = cand
                improved = True
        if not improved:
            break
    return best_p, best_v, math.exp(best_v) if best_v > -1e29 else 0.0


# --------------------------------------------------------------------------- #
#  DFT-based parameterization
# --------------------------------------------------------------------------- #
def _dft_parameterize(z):
    """Parameterize heights as |DFT(z)|^2 where z is a complex vector."""
    H = np.fft.fft(z)
    h = np.abs(H) ** 2
    return h


def _dft_inverse(h):
    """Given h = |DFT(z)|^2, find a z (phase choice)."""
    H_mags = np.sqrt(h + 1e-30)
    H = H_mags.astype(complex)
    z = np.fft.ifft(H)
    return z


def _optimize_dft(z0, budget):
    """Optimize over complex z to maximize ratio via |DFT(z)|^2."""
    z = z0.astype(complex).copy()

    def objective(z_flat):
        z_c = z_flat[:N] + 1j * z_flat[N:]
        h = _dft_parameterize(z_c)
        return -_logR_soft(h, beta=50.0)

    z_flat = np.concatenate([z.real, z.imag])
    t0 = time.time()
    try:
        res = minimize(objective, z_flat, method='L-BFGS-B',
                       options={'maxiter': 200, 'gtol': 1e-8, 'maxls': 20},
                       callback=lambda x: (np.nan if (time.time() - t0) > budget else x))
        z_opt = res.x[:N] + 1j * res.x[N:]
        h_opt = _dft_parameterize(z_opt)
        v_opt = _logR_soft(h_opt, beta=50.0)
        return h_opt, v_opt, math.exp(v_opt) if v_opt > -1e29 else 0.0
    except Exception:
        h_opt = _dft_parameterize(z)
        v_opt = _logR_soft(h_opt, beta=50.0)
        return h_opt, v_opt, math.exp(v_opt) if v_opt > -1e29 else 0.0


# --------------------------------------------------------------------------- #
#  Global search over the parameter space
# --------------------------------------------------------------------------- #
def _global_starts():
    """Yield diverse starting points in parameter space."""
    rng = np.random.default_rng(20240501)
    starts = []

    # (a) Sparse single dominant bump (positive alpha on one basis, zero else).
    for p in range(P):
        a = np.zeros(P)
        a[p] = 1.0
        starts.append(a)

    # (b) Pairs of bumps (spreads support to reduce peak overlap).
    for p1 in range(0, P, 2):
        for p2 in range(p1 + 2, P, 2):
            a = np.zeros(P)
            a[p1] = 1.0
            a[p2] = 1.0
            starts.append(a)

    # (c) Random Latin-hypercube-ish spreads.
    for _ in range(60):
        a = rng.normal(0.0, 1.0, size=P)
        starts.append(a)

    # (d) Alternating sign patterns (to create multi-peak after max(0,.)).
    for _ in range(40):
        a = rng.choice([-1.0, 1.0], size=P)
        a *= rng.uniform(0.5, 1.5)
        starts.append(a)

    return starts


def _direct_initializations():
    """Generate diverse height initializations for direct optimization."""
    inits = []
    inits.append(np.ones(N))

    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N

    # Single Gaussians
    for center in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        for width in [0.05, 0.1, 0.15, 0.2]:
            h = np.exp(-((x - center) ** 2) / (2 * width ** 2))
            inits.append(np.maximum(h, 0.0))

    # Two Gaussians
    for c1, c2 in [(-0.15, 0.15), (-0.1, 0.1), (-0.2, 0.2)]:
        for w in [0.05, 0.08, 0.12]:
            h1 = np.exp(-((x - c1) ** 2) / (2 * w ** 2))
            h2 = np.exp(-((x - c2) ** 2) / (2 * w ** 2))
            inits.append(np.maximum(h1 + h2, 0.0))

    # Random
    rng = np.random.RandomState(42)
    for _ in range(10):
        h_rand = rng.random(N)
        inits.append(h_rand)

    return inits


def construct():
    global _start
    _start = time.time()

    best_params = None
    best_log = -1e30
    best_R = 0.0
    best_h = np.ones(N)

    # --- Basis parameterization path ---
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

    top_k = min(len(scored), 60)
    for k in range(top_k):
        v0, a = scored[k]
        if _time_left() <= 6.0:
            break
        if v0 < best_log - 2.0 and best_log > -1e29:
            continue
        budget = min(4.0, max(1.0, _time_left() - 8.0))
        try:
            p_opt, v_opt, R_opt = _polish(a, budget=budget)
            if v_opt > best_log:
                best_log = v_opt
                best_params = p_opt.copy()
                best_R = R_opt
                best_h = _forward(p_opt)
        except Exception:
            pass
        if _time_left() <= 20.0:
            break

    # Coordinate-descent fallback
    if best_params is not None and _time_left() > 15.0:
        try:
            p_cd, v_cd, R_cd = _coordinate_descent(best_params, budget=min(10.0, _time_left() - 18.0))
            if v_cd > best_log:
                best_log = v_cd
                best_params = p_cd.copy()
                best_R = R_cd
                best_h = _forward(p_cd)
        except Exception:
            pass

    # Final high-precision polish on the global best.
    if best_params is not None and _time_left() > 12.0:
        try:
            p_f, v_f, R_f = _polish(best_params, budget=max(1.0, _time_left() - 18.0))
            if v_f > best_log:
                best_log = v_f
                best_params = p_f.copy()
                best_R = R_f
                best_h = _forward(p_f)
        except Exception:
            pass

    # --- Direct height-space optimization path ---
    if _time_left() > 20.0:
        inits = _direct_initializations()
        scored_h = []
        for h_init in inits:
            h_init = np.maximum(h_init, 0.0)
            s = _logR_soft(h_init, beta=50.0)
            scored_h.append((s, h_init))
        scored_h.sort(key=lambda t: -t[0])

        top_k_h = min(len(scored_h), 10)
        for k in range(top_k_h):
            if _time_left() <= 8.0:
                break
            v0, h_init = scored_h[k]
            if v0 < best_log - 2.0 and best_log > -1e29:
                continue
            budget = min(5.0, max(1.0, _time_left() - 10.0))
            try:
                h_opt, v_opt, R_opt = _polish_direct(h_init, budget=budget)
                if v_opt > best_log:
                    best_log = v_opt
                    best_R = R_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    # --- DFT-based optimization path ---
    if _time_left() > 15.0:
        h_start = best_h.copy()
        if np.sum(h_start) <= 0:
            h_start = np.ones(N)
        z_init = _dft_inverse(h_start)
        budget = min(5.0, max(1.0, _time_left() - 12.0))
        try:
            h_dft, v_dft, R_dft = _optimize_dft(z_init, budget=budget)
            if v_dft > best_log:
                best_log = v_dft
                best_R = R_dft
                best_h = h_dft.copy()
        except Exception:
            pass

    # Materialize the height vector.
    h = np.maximum(best_h, 0.0)
    if not np.all(np.isfinite(h)) or float(np.sum(h)) <= 0.0:
        h = np.ones(N)
    return [float(x) for x in h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio_from_h(h))
    print("time =", time.time() - t0)
