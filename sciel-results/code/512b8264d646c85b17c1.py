"""Combined program: FIR/spectral shaping with smooth-max L-BFGS-B + Golay/ZC global search + high-precision polish.

Strategy:
- Primary: DCT spectral shaping (40 modes) with smooth log-sum-exp max surrogate and L-BFGS-B.
- Secondary: Golay complementary pair (two chirps) global search via differential evolution.
- Tertiary: Windowed Zadoff-Chu global search.
- Refinement: Project best to DFT basis (20 harmonics), refine with L-BFGS-B (smooth) then Nelder-Mead (exact).
- Final: High-precision Nelder-Mead polish on DFT basis.
- Deadline guard ensures return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.fft import dct as scipy_dct

N = 400
_t0 = time.time()
_DEADLINE = 135.0  # safety margin under 150s wall-clock
_K_DCT = 40
_K_DFT = 20

# Precompute common arrays
_i_arr = np.arange(N, dtype=float)

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _smooth_max_value(h, beta=80.0):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * log_sum_exp / (s * s))

# DCT basis for spectral shaping
_n_idx = np.arange(N)
_k_idx = np.arange(_K_DCT)
_angles = np.pi * (2 * _n_idx[:, None] + 1) * _k_idx[None, :] / (2 * N)
_DCT_BASIS = np.cos(_angles)
_norms = np.sqrt(np.sum(_DCT_BASIS ** 2, axis=0))
_norms[_norms == 0] = 1.0
_DCT_BASIS = _DCT_BASIS / _norms

def _dct_h(coeffs):
    """Generate h from DCT coefficients."""
    return _DCT_BASIS @ coeffs

def _dct_smooth_obj(coeffs, beta=80.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dct_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

# DFT basis for refinement
_basis = np.zeros((2 * _K_DFT, N))
for k in range(1, _K_DFT + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)

def _dft_h(coeffs, base_val):
    """Generate h from DFT coefficients and a base value."""
    return base_val + _basis.T @ coeffs

def _dft_obj(coeffs_base):
    """Objective for DFT parameters: [coeffs..., base]."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _value(h)

def _dft_smooth_obj(coeffs_base, beta=80.0):
    """Smooth objective for DFT parameters."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _smooth_max_value(h, beta)

def _golay_h(params):
    """
    Golay complementary pair: h = base + a1*cos(q1) + a2*cos(q2)
    where q1, q2 are quadratic phases (chirps).
    """
    p1, r1, a1, p2, r2, a2, base = params
    ang1 = 2.0 * np.pi * r1 * _i_arr * (_i_arr + 1) / N + p1
    ang2 = 2.0 * np.pi * r2 * _i_arr * (_i_arr + 1) / N + p2
    return base + a1 * np.cos(ang1) + a2 * np.cos(ang2)

def _golay_obj(params):
    h = _golay_h(params)
    return _value(h)

def _golay_smooth_obj(params, beta=80.0):
    h = _golay_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zc_h(params):
    """Windowed Zadoff-Chu sequence."""
    phase, rate, amp, base, win_width, win_center = params
    ang = 2.0 * np.pi * rate * _i_arr * (_i_arr + 1) / N + phase
    rel_pos = _i_arr / N
    x = (rel_pos - win_center) / (0.5 * win_width)
    window = np.where(np.abs(x) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    return base + amp * window * np.cos(ang)

def _windowed_zc_obj(params):
    h = _windowed_zc_h(params)
    return _value(h)

def _chirp_h(params):
    """Single chirp: h = base + amp * cos(phase + rate * (i/N)^2)"""
    phase, rate, amp, base = params
    ang = phase + rate * (_i_arr / N) ** 2
    return base + amp * np.cos(ang)

def _chirp_obj(params):
    h = _chirp_h(params)
    return _value(h)

def _project_dft(h):
    """Project h onto DFT basis + base, return (coeffs, base_val)."""
    mean_h = float(np.mean(h))
    coeffs = np.zeros(2 * _K_DFT)
    target = h - mean_h
    for k in range(1, _K_DFT + 1):
        angle = 2.0 * np.pi * k * _n_idx / N
        cos_k = np.cos(angle)
        sin_k = np.sin(angle)
        A = np.column_stack([cos_k, sin_k])
        try:
            sol = np.linalg.lstsq(A, target, rcond=None)[0]
            coeffs[2 * (k - 1)] = sol[0]
            coeffs[2 * (k - 1) + 1] = sol[1]
        except:
            pass
    return coeffs, mean_h

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
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: DCT spectral shaping with multiple restarts (primary)
    if deadline_ok():
        try:
            rng = np.random.default_rng(42)
            for restart in range(10):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(4, min(15, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 1.5, size=n_active)
                
                res = minimize(_dct_smooth_obj, coeffs, method='L-BFGS-B',
                              options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 2: Golay complementary pair global search
    if deadline_ok():
        try:
            bounds_golay = [
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res = differential_evolution(
                _golay_obj, bounds_golay, seed=7, maxiter=50, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _golay_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: Windowed Zadoff-Chu global search
    if deadline_ok():
        try:
            bounds_wzc = [
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
                (0.3, 1.5),
                (0.2, 0.8),
            ]
            res = differential_evolution(
                _windowed_zc_obj, bounds_wzc, seed=13, maxiter=45, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _windowed_zc_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: Chirp global search for diversity
    if deadline_ok():
        try:
            bounds_chirp = [
                (-np.pi, np.pi),
                (-200.0, 200.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res = differential_evolution(
                _chirp_obj, bounds_chirp, seed=99, maxiter=40, tol=1e-8,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _chirp_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 5: L-BFGS-B refinement on DFT basis from best solution (smooth)
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_dft = minimize(
                lambda x: _dft_smooth_obj(x, beta=80.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_dft.x is not None:
                h = _dft_h(res_dft.x[:-1], res_dft.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 6: DCT-based L-BFGS-B refinement from best solution
    if deadline_ok():
        try:
            dct_coeffs = scipy_dct(best_h, type=2, n=_K_DCT, norm='ortho')
            
            res_dct_refine = minimize(
                lambda c: _smooth_max_value(_dct_h(c), beta=80.0),
                dct_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_dct_refine.x is not None:
                h = _dct_h(res_dct_refine.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 7: Nelder-Mead polish on DFT basis (exact objective)
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_nm = minimize(
                _dft_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-11, 'fatol': 1e-13, 'maxiter': 800, 'maxfev': 1500}
            )
            if res_nm.x is not None:
                h = _dft_h(res_nm.x[:-1], res_nm.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 8: Additional DCT refinement with different beta values
    if deadline_ok():
        try:
            for beta_val in [40.0, 100.0, 150.0]:
                if not deadline_ok():
                    break
                dct_coeffs = scipy_dct(best_h, type=2, n=_K_DCT, norm='ortho')
                res_dct_beta = minimize(
                    lambda c: _smooth_max_value(_dct_h(c), beta=beta_val),
                    dct_coeffs,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res_dct_beta.x is not None:
                    h = _dct_h(res_dct_beta.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 9: Final high-precision Nelder-Mead polish
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_nm2 = minimize(
                _dft_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-13, 'fatol': 1e-15, 'maxiter': 1000, 'maxfev': 2000}
            )
            if res_nm2.x is not None:
                h = _dft_h(res_nm2.x[:-1], res_nm2.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 10: Final refinement using exact objective with tighter tolerances
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_final = minimize(
                _dft_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 500, 'maxfev': 1000}
            )
            if res_final.x is not None:
                h = _dft_h(res_final.x[:-1], res_final.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
