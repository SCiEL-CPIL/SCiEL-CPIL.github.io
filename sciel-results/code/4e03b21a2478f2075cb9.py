"""Model-and-optimize program for third autocorrelation inequality.

Strategy:
  * h = baseline + zero-mean DCT-II oscillatory perturbation (low-frequency basis).
  * Soft-max (log-sum-exp) surrogate of the signed max of the full autoconvolution,
    with an exact analytic gradient computed via FFT (convolution, not correlation).
  * L-BFGS-B from many random DCT restarts, then full-400-dim polish with a
    two-stage beta schedule (broad -> sharp), then hard-objective coordinate polish.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fftpack import dct, idct

N = 400
N_FULL = 2 * N - 1
SIZE = 2 * N


def _autoconv(h):
    f = np.fft.rfft(h, n=SIZE)
    c = np.fft.irfft(f * f, n=SIZE)
    return c[:N_FULL]


def _hard(h):
    s = h.sum()
    if abs(s) < 1e-12:
        return 1e15
    conv = _autoconv(h)
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))


def _softmax_grad(h, beta):
    conv = _autoconv(h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500.0, 500.0)
    ez = np.exp(shifted)
    z = ez.sum()
    sm = m + np.log(z) / beta
    w = ez / z
    # d(sm)/d(h_i) = sum_k w_k * d(conv_k)/d(h_i) = (h * w)_i (convolution)
    fw = np.fft.rfft(w, n=SIZE)
    fh = np.fft.rfft(h, n=SIZE)
    g = np.fft.irfft(fh * fw, n=SIZE)
    return sm, g[:N]


def _param_to_h(p):
    baseline = p[0]
    m = len(p) - 1
    coeffs = np.zeros(N)
    if m > 0:
        coeffs[1:m + 1] = p[1:m + 1]
    h_osc = idct(coeffs, type=2, norm='ortho')
    h_osc = h_osc - h_osc.mean()
    return baseline + h_osc


def _param_obj(p, beta):
    h = _param_to_h(p)
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e15
    sm, _ = _softmax_grad(h, beta)
    return abs(2.0 * N * sm / (s * s))


def _full_obj_grad(x, beta):
    s = x.sum()
    if abs(s) < 1e-9:
        return 1e15, np.zeros_like(x)
    sm, g_sm = _softmax_grad(x, beta)
    val = 2.0 * N * sm / (s * s)
    g = 2.0 * N * (g_sm * s - 2.0 * sm) / (s ** 3)
    return val, g


def construct():
    t0 = time.time()
    DEADLINE = 140.0

    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(20240611)
    m_dct = 40

    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)

    # ---- Phase 1: DCT-parameterized optimization with many restarts ----
    n_restarts = 24
    baselines = [0.5, 1.0, 1.5, 2.0, -1.0, 0.75, 1.25]
    for r in range(n_restarts):
        if time_left() < 45.0:
            break
        scale = 0.3 / (np.arange(1, m_dct + 1) ** 0.5)
        coeffs = rng.normal(0, 1.0, size=m_dct) * scale
        baseline = baselines[r % len(baselines)] + rng.normal(0, 0.15)
        p0 = np.concatenate([[baseline], coeffs])

        h0 = _param_to_h(p0)
        v0 = _hard(h0)
        if v0 < best_val:
            best_val = v0
            best_h = h0.copy()

        try:
            res = minimize(_param_obj, p0, args=(15.0,), method='L-BFGS-B',
                           options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxfun': 2500})
            h_new = _param_to_h(res.x)
            v_new = _hard(h_new)
            if v_new < best_val:
                best_val = v_new
                best_h = h_new.copy()
        except Exception:
            pass

    # ---- Phase 2: Full 400-dim polish, broad -> sharp beta ----
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s

    beta_seq = [8.0, 15.0, 30.0, 60.0, 120.0]
    for beta in beta_seq:
        if time_left() < 30.0:
            break
        try:
            res = minimize(lambda x: _full_obj_grad(x, beta)[0], h_curr,
                           jac=lambda x: _full_obj_grad(x, beta)[1],
                           method='L-BFGS-B',
                           options={'maxiter': 350, 'ftol': 1e-14, 'gtol': 1e-12,
                                    'maxfun': 5000})
            v = _hard(res.x)
            if v < best_val:
                best_val = v
                best_h = res.x.copy()
                h_curr = res.x.copy()
        except Exception:
            pass

    # ---- Phase 3: Hard-objective coordinate / subgradient polish ----
    h_curr = best_h.copy()
    best_local = _hard(h_curr)
    max_iters = 40
    it = 0
    while time_left() > 20.0 and it < max_iters:
        it += 1
        conv = _autoconv(h_curr)
        kmax = int(np.argmax(conv))
        lag = kmax - (N - 1)
        # subgradient of conv[lag]: d/dh_i = h[lag-i] + h[i-lag]
        sg = np.zeros(N)
        a = lag - np.arange(N)
        b = np.arange(N) - lag
        sg += np.where((a >= 0) & (a < N), h_curr[np.clip(a, 0, N - 1)], 0.0)
        sg += np.where((b >= 0) & (b < N), h_curr[np.clip(b, 0, N - 1)], 0.0)
        sg_centered = sg - sg.mean()
        improved = False
        for alpha in [2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]:
            cand = h_curr - alpha * sg_centered
            vv = _hard(cand)
            if vv < best_local:
                best_local = vv
                h_curr = cand
                improved = True
                if vv < best_val:
                    best_val = vv
                    best_h = cand.copy()
                break
        if not improved:
            break

    # ---- Phase 4: Final high-precision polish ----
    if time_left() > 12.0:
        h_curr = best_h.copy()
        s = h_curr.sum()
        if abs(s) > 1e-9:
            h_curr = h_curr / s
        for beta in [200.0, 400.0]:
            if time_left() < 7.0:
                break
            try:
                res = minimize(lambda x: _full_obj_grad(x, beta)[0], h_curr,
                               jac=lambda x: _full_obj_grad(x, beta)[1],
                               method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13,
                                        'maxfun': 8000})
                v = _hard(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
            except Exception:
                pass

    # ---- Sanity checks ----
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        best_val = _hard(best_h)
    if abs(best_h.sum()) < 1e-10:
        best_h = np.ones(N)
        best_val = _hard(best_h)

    return list(best_h)
