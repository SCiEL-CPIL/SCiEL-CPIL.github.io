"""Model-and-optimize program for the third autocorrelation inequality.

Strategy:
- Parameterize h via low-frequency DCT + mean (smooth spectral model) and a full
  per-step model (h = mean + A*cos(w*i+phi) + sum_k b_k * DCT_k(i)).
- Atom-based sparse parameterization: h = sum_j a_j * W(i - o_j) * s_j
- Fit/optimize parameters directly against a smooth log-sum-exp surrogate of
  max(autoconv) / sum^2 using L-BFGS-B, with many restarts.
- Fast local per-step refinement using in-place FFT updates.
- Nelder-Mead polish as final step.
- Explicit deadline guard returns the best solution found before 150s.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
_t0 = time.time()
_DEADLINE = 140.0

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

def _smooth_value(h, beta=100.0):
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
_K = 14
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
    return _smooth_value(_h_from_dct(c), beta=100.0)

# Full model: h = base + A*cos(w*i + phi) + DCT_coeffs
# params = [base, A, w, phi, c_0..c_{K-1}]  (total 4 + K)
def _full_model(params):
    base = params[0]
    A = params[1]
    w = params[2]
    phi = params[3]
    coeffs = params[4:]
    h = _h_from_dct(coeffs) + base + A * np.cos(w * _i_arr + phi)
    return h

def _full_obj(params):
    return _smooth_value(_full_model(params), beta=100.0)

def _project_dct(h):
    mean = float(np.mean(h))
    coeffs = _DCT_T @ (h - mean)
    return coeffs, mean

# Fast local per-step refinement using precomputed FFT and per-step update
def _local_refine(h, max_rounds=6, step=0.05):
    """Greedy per-step descent on smooth surrogate using in-place FFT updates."""
    h = h.copy()
    s = float(np.sum(h))
    if s == 0 or not np.all(np.isfinite(h)):
        return h

    H = np.fft.rfft(h, _fft_size)
    H2 = H * H
    conv = np.fft.irfft(H2, _fft_size)[:_conv_len]
    m = conv.max()
    beta = 100.0
    scaled = np.clip(beta * (conv - m), -500.0, 500.0)
    exp_sum = np.sum(np.exp(scaled))
    v = abs(2.0 * N * (m + np.log(exp_sum) / beta) / (s * s))

    for rnd in range(max_rounds):
        improved = False
        for i in range(N):
            old = h[i]
            old_H = H[i]
            old_H2 = H2[i]
            best_delta = None
            best_v = v
            for delta in (step, -step):
                new = old + delta
                if new == old:
                    continue
                new_H = old_H + delta
                new_H2 = new_H * new_H
                H2[i] = new_H2
                conv_new = np.fft.irfft(H2, _fft_size)[:_conv_len]
                m_new = conv_new.max()
                scaled_new = np.clip(beta * (conv_new - m_new), -500.0, 500.0)
                exp_sum_new = np.sum(np.exp(scaled_new))
                s_new = s + delta
                if abs(s_new) < 1e-12:
                    H2[i] = old_H2
                    continue
                v_new = abs(2.0 * N * (m_new + np.log(exp_sum_new) / beta) / (s_new * s_new))
                if v_new < best_v - 1e-12:
                    best_v = v_new
                    best_delta = delta
                    best_H = new_H
                    best_H2 = new_H2
                    best_m = m_new
                    best_s = s_new
            H2[i] = old_H2
            if best_delta is not None:
                H[i] = best_H
                H2[i] = best_H2
                h[i] = old + best_delta
                v = best_v
                m = best_m
                s = best_s
                improved = True
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break
    return h

def _local_refine_exact(h, max_rounds=4, step=0.03):
    """Per-step descent on exact value using in-place FFT (no surrogate bias)."""
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
            old_H = H[i]
            old_H2 = H2[i]
            best_delta = None
            best_v = v
            for delta in (step, -step):
                new = old + delta
                if new == old:
                    continue
                new_H = old_H + delta
                new_H2 = new_H * new_H
                H2[i] = new_H2
                conv = np.fft.irfft(H2, _fft_size)[:_conv_len]
                s_new = s + delta
                if abs(s_new) < 1e-12:
                    H2[i] = old_H2
                    continue
                v_new = abs(2.0 * N * conv.max() / (s_new * s_new))
                if v_new < best_v - 1e-12:
                    best_v = v_new
                    best_delta = delta
                    best_H = new_H
                    best_H2 = new_H2
                    best_s = s_new
            H2[i] = old_H2
            if best_delta is not None:
                H[i] = best_H
                H2[i] = best_H2
                h[i] = old + best_delta
                v = best_v
                s = best_s
                improved = True
        if not improved:
            step *= 0.5
            if step < 1e-4:
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

    rng = np.random.default_rng(12345)

    # Phase 1: DCT-only spectral shaping with L-BFGS-B, multiple restarts
    if deadline_ok():
        try:
            for restart in range(12):
                if not deadline_ok():
                    break
                if restart == 0:
                    coeffs = np.zeros(_K)
                else:
                    n_active = int(rng.integers(3, min(10, _K)))
                    active = rng.choice(_K, size=n_active, replace=False)
                    coeffs = np.zeros(_K)
                    coeffs[active] = rng.normal(0, 1.5, size=n_active)
                res = minimize(_dct_obj, coeffs, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.x is not None:
                    h = _h_from_dct(res.x)
                    update_best(h)
        except Exception:
            pass

    # Phase 2: Full model (DCT + chirp + base) with L-BFGS-B
    if deadline_ok():
        try:
            for restart in range(10):
                if not deadline_ok():
                    break
                base = 0.0
                A = 0.0
                w = 0.0
                phi = 0.0
                if restart > 0:
                    base = float(rng.normal(0, 0.5))
                    A = float(rng.uniform(0.1, 2.0)) * (1 if rng.random() < 0.5 else -1)
                    w = float(rng.uniform(0.5, 30.0)) * (1 if rng.random() < 0.5 else -1)
                    phi = float(rng.uniform(-np.pi, np.pi))
                coeffs = np.zeros(_K)
                if restart > 0:
                    n_active = int(rng.integers(2, min(8, _K)))
                    active = rng.choice(_K, size=n_active, replace=False)
                    coeffs[active] = rng.normal(0, 1.0, size=n_active)
                params = np.concatenate([[base, A, w, phi], coeffs])
                res = minimize(_full_obj, params, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.x is not None:
                    h = _full_model(res.x)
                    update_best(h)
        except Exception:
            pass

    # Phase 2.5: Atom-based sparse parameterization
    # h = sum_j a_j * W((i - o_j) mod N) * s_j
    # params: for each atom j: [a_j, o_j, s_j] where s_j in {-1, 1}
    # We use a Hamming window of length 10 as the atom
    if deadline_ok():
        try:
            atom_len = 10
            W = np.hamming(atom_len)
            
            def _atom_model(params, n_atoms):
                h = np.zeros(N)
                for j in range(n_atoms):
                    a = params[j * 3]
                    o = int(round(params[j * 3 + 1])) % N
                    s = 1.0 if params[j * 3 + 2] >= 0 else -1.0
                    # Place atom centered at position o
                    for k in range(atom_len):
                        pos = (o - atom_len // 2 + k) % N
                        h[pos] += a * s * W[k]
                return h
            
            def _atom_obj(params, n_atoms):
                h = _atom_model(params, n_atoms)
                return _smooth_value(h, beta=100.0)
            
            for n_atoms in [3, 5, 7]:
                if not deadline_ok():
                    break
                for restart in range(5):
                    if not deadline_ok():
                        break
                    if restart == 0:
                        params = np.concatenate([
                            [1.0, 0.0, 1.0] * n_atoms
                        ])
                    else:
                        params = np.zeros(n_atoms * 3)
                        for j in range(n_atoms):
                            params[j * 3] = rng.normal(0, 1.0)
                            params[j * 3 + 1] = rng.uniform(0, N)
                            params[j * 3 + 2] = 1.0 if rng.random() < 0.5 else -1.0
                    res = minimize(lambda p: _atom_obj(p, n_atoms), params,
                                   method='L-BFGS-B',
                                   options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                    if res.x is not None:
                        h = _atom_model(res.x, n_atoms)
                        update_best(h)
        except Exception:
            pass

    # Phase 3: Local per-step refinement (smooth surrogate)
    if deadline_ok():
        try:
            h_ref = _local_refine(best_h, max_rounds=6, step=0.05)
            update_best(h_ref)
        except Exception:
            pass

    # Phase 4: Refine on DFT/DCT basis from best solution with L-BFGS-B
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dct(best_h)
            res = minimize(_dct_obj, coeffs_init, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.x is not None:
                h = _h_from_dct(res.x)
                update_best(h)
        except Exception:
            pass

    # Phase 5: Local exact refinement
    if deadline_ok():
        try:
            h_ref = _local_refine_exact(best_h, max_rounds=4, step=0.03)
            update_best(h_ref)
        except Exception:
            pass

    # Phase 5.5: Nelder-Mead polish on full 400-dim space
    if deadline_ok():
        try:
            h_nm = best_h.copy()
            res = minimize(_value, h_nm, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.x is not None:
                update_best(res.x)
        except Exception:
            pass

    # Phase 6: More DCT restarts with different seeds if time remains
    if deadline_ok():
        try:
            for restart in range(15, 30):
                if not deadline_ok():
                    break
                n_active = int(rng.integers(3, min(10, _K)))
                active = rng.choice(_K, size=n_active, replace=False)
                coeffs = np.zeros(_K)
                coeffs[active] = rng.normal(0, 1.5, size=n_active)
                res = minimize(_dct_obj, coeffs, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.x is not None:
                    h = _h_from_dct(res.x)
                    update_best(h)
        except Exception:
            pass

    # Phase 7: Final local smooth refinement
    if deadline_ok():
        try:
            h_ref = _local_refine(best_h, max_rounds=4, step=0.03)
            update_best(h_ref)
        except Exception:
            pass

    # Phase 8: Final local exact refinement
    if deadline_ok():
        try:
            h_ref = _local_refine_exact(best_h, max_rounds=3, step=0.02)
            update_best(h_ref)
        except Exception:
            pass

    # Phase 9: Final Nelder-Mead polish if time remains
    if deadline_ok():
        try:
            h_nm = best_h.copy()
            res = minimize(_value, h_nm, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.x is not None:
                update_best(res.x)
        except Exception:
            pass

    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
