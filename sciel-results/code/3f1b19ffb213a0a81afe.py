"""Improved program for the third autocorrelation inequality.

Strategy:
- Use a smooth log-sum-exp surrogate of the max(autoconv) term to make the objective differentiable.
- Optimize in a low-dimensional DCT subspace (first ~20 modes) with L-BFGS-B, multiple restarts.
- Perform full-dimensional coordinate descent with exact value evaluation for local refinement.
- Use Nelder-Mead as a final polish step.
- Explicit deadline guard returns best solution before 150s.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
_t0 = time.time()
_DEADLINE = 145.0

# Precompute
_i_arr = np.arange(N, dtype=float)
_conv_len = 2 * N - 1
_fft_size = 2 * N

def _autoconv(h):
    H = np.fft.rfft(h, _fft_size)
    c = np.fft.irfft(H * H, _fft_size)[:_conv_len]
    return c

def _value(h):
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _autoconv(h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _smooth_value(h, beta=200.0):
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _autoconv(h)
    m = conv.max()
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -500.0, 500.0)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * lse / (s * s))

# DCT-II basis (orthonormal, first K modes)
_K = 20
_n = np.arange(N)
_k = np.arange(_K)
_angles = np.pi * (2 * _n[:, None] + 1) * _k[None, :] / (2 * N)
_DCT = np.cos(_angles)
_norms = np.sqrt(np.sum(_DCT ** 2, axis=0))
_norms[_norms == 0] = 1.0
_DCT = _DCT / _norms
_DCT_T = _DCT.T

def _h_from_dct(c):
    return _DCT @ c

def _dct_obj(c):
    return _smooth_value(_h_from_dct(c), beta=200.0)

def _project_dct(h):
    mean = float(np.mean(h))
    coeffs = _DCT_T @ (h - mean)
    return coeffs, mean

def _local_refine_exact(h, max_rounds=8, step=0.02):
    """Per-step coordinate descent on exact value."""
    h = h.copy()
    v = _value(h)
    s = float(np.sum(h))
    if s == 0 or not np.all(np.isfinite(h)):
        return h
    H = np.fft.rfft(h, _fft_size)
    H2 = H * H

    for rnd in range(max_rounds):
        improved = False
        for i in range(N):
            old = h[i]
            for delta in (step, -step):
                new = old + delta
                if new == old:
                    continue
                H[i] = H[i] + delta
                H2[i] = H[i] * H[i]
                conv = np.fft.irfft(H2, _fft_size)[:_conv_len]
                s_new = s + delta
                if abs(s_new) < 1e-12:
                    H[i] = old
                    continue
                v_new = abs(2.0 * N * conv.max() / (s_new * s_new))
                if v_new < v - 1e-12:
                    v = v_new
                    h[i] = new
                    s = s_new
                    improved = True
                    break
            else:
                H[i] = old
        if not improved:
            step *= 0.5
            if step < 1e-5:
                break
    return h

def _local_refine_smooth(h, max_rounds=6, step=0.03, beta=200.0):
    """Per-step coordinate descent on smooth surrogate."""
    h = h.copy()
    s = float(np.sum(h))
    if s == 0 or not np.all(np.isfinite(h)):
        return h
    H = np.fft.rfft(h, _fft_size)
    H2 = H * H
    conv = np.fft.irfft(H2, _fft_size)[:_conv_len]
    m = conv.max()
    scaled = np.clip(beta * (conv - m), -500.0, 500.0)
    exp_sum = np.sum(np.exp(scaled))
    v = abs(2.0 * N * (m + np.log(exp_sum) / beta) / (s * s))

    for rnd in range(max_rounds):
        improved = False
        for i in range(N):
            old = h[i]
            for delta in (step, -step):
                new = old + delta
                if new == old:
                    continue
                H[i] = H[i] + delta
                H2[i] = H[i] * H[i]
                conv_new = np.fft.irfft(H2, _fft_size)[:_conv_len]
                m_new = conv_new.max()
                scaled_new = np.clip(beta * (conv_new - m_new), -500.0, 500.0)
                exp_sum_new = np.sum(np.exp(scaled_new))
                s_new = s + delta
                if abs(s_new) < 1e-12:
                    H[i] = old
                    continue
                v_new = abs(2.0 * N * (m_new + np.log(exp_sum_new) / beta) / (s_new * s_new))
                if v_new < v - 1e-12:
                    v = v_new
                    h[i] = new
                    m = m_new
                    s = s_new
                    improved = True
                    break
            else:
                H[i] = old
        if not improved:
            step *= 0.5
            if step < 1e-5:
                break
    return h

def construct():
    global _t0
    _t0 = time.time()

    best_h = np.ones(N)
    best_v = _value(best_h)

    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return best_v
        s = float(np.sum(h))
        if s == 0:
            return best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v

    rng = np.random.default_rng(42)

    # Phase 1: DCT spectral shaping with L-BFGS-B, multiple restarts
    if deadline_ok():
        try:
            for restart in range(15):
                if not deadline_ok():
                    break
                if restart == 0:
                    coeffs = np.zeros(_K)
                else:
                    n_active = int(rng.integers(3, min(12, _K)))
                    active = rng.choice(_K, size=n_active, replace=False)
                    coeffs = np.zeros(_K)
                    coeffs[active] = rng.normal(0, 2.0, size=n_active)
                res = minimize(_dct_obj, coeffs, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12})
                if res.x is not None:
                    h = _h_from_dct(res.x)
                    update_best(h)
        except Exception:
            pass

    # Phase 2: Local smooth refinement
    if deadline_ok():
        try:
            h_ref = _local_refine_smooth(best_h, max_rounds=8, step=0.04, beta=200.0)
            update_best(h_ref)
        except Exception:
            pass

    # Phase 3: More DCT restarts with different initializations
    if deadline_ok():
        try:
            for restart in range(15, 35):
                if not deadline_ok():
                    break
                n_active = int(rng.integers(4, min(15, _K)))
                active = rng.choice(_K, size=n_active, replace=False)
                coeffs = np.zeros(_K)
                coeffs[active] = rng.normal(0, 2.5, size=n_active)
                res = minimize(_dct_obj, coeffs, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12})
                if res.x is not None:
                    h = _h_from_dct(res.x)
                    update_best(h)
        except Exception:
            pass

    # Phase 4: Local exact refinement
    if deadline_ok():
        try:
            h_ref = _local_refine_exact(best_h, max_rounds=10, step=0.02)
            update_best(h_ref)
        except Exception:
            pass

    # Phase 5: Nelder-Mead polish on DCT coefficients
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dct(best_h)
            res = minimize(_dct_obj, coeffs_init, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.x is not None:
                h = _h_from_dct(res.x)
                update_best(h)
        except Exception:
            pass

    # Phase 6: Final local smooth refinement
    if deadline_ok():
        try:
            h_ref = _local_refine_smooth(best_h, max_rounds=6, step=0.02, beta=300.0)
            update_best(h_ref)
        except Exception:
            pass

    # Phase 7: Final local exact refinement
    if deadline_ok():
        try:
            h_ref = _local_refine_exact(best_h, max_rounds=8, step=0.01)
            update_best(h_ref)
        except Exception:
            pass

    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
