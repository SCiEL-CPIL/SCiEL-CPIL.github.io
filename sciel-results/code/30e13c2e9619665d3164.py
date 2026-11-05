"""
Third autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2 over h in R^N, N=400.

Strategy:
- Primary: DCT spectral shaping with L-BFGS-B (enforces nonzero sum via base offset)
- Secondary: Golay complementary pair inspired sequences
- Tertiary: Windowed Zadoff-Chu with global search
- Final: High-dimensional L-BFGS-B polish with smooth max
- Deadline guard ensures we return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
_t0 = time.time()
_DEADLINE = 130.0  # safety margin under 150s wall-clock
_K_DCT = 25  # number of DCT basis vectors for spectral shaping
_K_DFT = 12  # number of DFT frequency components for refinement

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _smooth_max_value(h, beta=30.0):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    # Smooth max via log-sum-exp (scaled)
    shifted = beta * (conv - m)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    smooth_m = m + np.log(np.sum(np.exp(shifted))) / beta
    return abs(2.0 * N * smooth_m / (s * s))

def _dct_basis_matrix(K, N):
    """Build DCT-II basis matrix of shape (N, K) for first K modes."""
    n = np.arange(N)
    k = np.arange(K)
    angles = np.pi * (2 * n[:, None] + 1) * k[None, :] / (2 * N)
    basis = np.cos(angles)
    norms = np.sqrt(np.sum(basis ** 2, axis=0))
    norms[norms == 0] = 1.0
    basis = basis / norms
    return basis

_DCT_BASIS = _dct_basis_matrix(_K_DCT, N)  # (N, K_DCT)

def _dct_h(coeffs, base_val=1.0):
    """Generate h from DCT coefficients + base offset to ensure nonzero sum."""
    h = base_val + _DCT_BASIS @ coeffs
    return h

def _dct_smooth_obj(params, beta=30.0):
    """Smooth objective for DCT coefficients + base. params: [coeffs..., base]"""
    coeffs = params[:-1]
    base_val = params[-1]
    h = _dct_h(coeffs, base_val)
    return _smooth_max_value(h, beta)

def _windowed_zadoff_chu_h(params):
    """Windowed Zadoff-Chu: h[i] = base + amp * window[i] * cos(2*pi*rate*i*(i+1)/N + phase).
    
    params: [phase, rate, amp, base, win_width, win_center, win_type]
    - win_width: relative width of window (0.1 to 1.5), controls Gaussian sigma or Hann extent
    - win_center: center of window in [0, 1] (relative to N)
    - win_type: 0=Hann, 1=Gaussian
    """
    phase, rate, amp, base = params[0], params[1], params[2], params[3]
    win_width = params[4] if len(params) > 4 else 1.0
    win_center = params[5] if len(params) > 5 else 0.5
    win_type = params[6] if len(params) > 6 else 0.0
    
    i = np.arange(N, dtype=float)
    # Zadoff-Chu quadratic phase
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    
    # Apply smooth window to suppress sidelobes
    rel_pos = i / N  # [0, 1]
    if win_type < 0.5:
        # Hann window centered at win_center with width win_width
        # Hann: 0.5*(1+cos(pi*x)) for x in [-1,1], 0 outside
        x = (rel_pos - win_center) / (0.5 * win_width)
        window = np.where(np.abs(x) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    else:
        # Gaussian window
        sigma = win_width / 4.0  # FWHM ~ 2.355*sigma
        window = np.exp(-0.5 * ((rel_pos - win_center) / sigma) ** 2)
    
    h = base + amp * window * np.cos(ang)
    return h

def _zadoff_chu_h(params):
    """Zadoff-Chu inspired sequence: h[i] = base + amp * cos(2*pi*rate*i*(i+1)/N + phase)."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    # Zadoff-Chu uses quadratic phase with specific structure
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    h = base + amp * np.cos(ang)
    return h

def _zadoff_chu_obj(params):
    """Objective for Zadoff-Chu parameters."""
    h = _zadoff_chu_h(params)
    return _value(h)

def _zadoff_chu_smooth_obj(params, beta=50.0):
    """Smooth objective for Zadoff-Chu parameters."""
    h = _zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zc_obj(params):
    """Objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _value(h)

def _windowed_zc_smooth_obj(params, beta=50.0):
    """Smooth objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    # Quadratic phase: phase + rate * (i/N)^2
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _chirp_obj(params):
    """Objective for chirp parameters."""
    h = _chirp_h(params)
    return _value(h)

def _chirp_smooth_obj(params, beta=50.0):
    """Smooth objective for chirp parameters."""
    h = _chirp_h(params)
    return _smooth_max_value(h, beta)

# Low-frequency DFT basis for refinement
_n_idx = np.arange(N)
_basis = np.zeros((2 * _K_DFT, N))
for k in range(1, _K_DFT + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
_BASIS_T = _basis.T  # (N, 2*K_DFT)

def _dft_h(coeffs, base_val):
    """Generate h from DFT coefficients and a base value."""
    # coeffs shape: (2*K,), base_val is scalar
    h = base_val + _BASIS_T @ coeffs
    return h

def _dft_obj(coeffs_base):
    """Objective for DFT parameters: [coeffs..., base]."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _value(h)

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    def update_best(h):
        nonlocal best_h, best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: DCT spectral shaping with L-BFGS-B (primary)
    if deadline_ok():
        try:
            rng = np.random.default_rng(123)
            for restart in range(6):
                if not deadline_ok():
                    break
                # Initialize DCT coefficients + base
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(3, min(10, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 0.5, size=n_active)
                base_init = 1.0 + rng.normal(0, 0.1)
                
                params_init = np.concatenate([coeffs, [base_init]])
                
                res = minimize(
                    lambda p: _dct_smooth_obj(p, beta=30.0),
                    params_init,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * _K_DCT + [(0.1, 10.0)],
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0}
                )
                if res.fun < np.inf:
                    coeffs_opt = res.x[:-1]
                    base_opt = res.x[-1]
                    h_candidate = _dct_h(coeffs_opt, base_opt)
                    update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 2: Golay complementary pair optimization
    if deadline_ok():
        try:
            rng = np.random.default_rng(456)
            for restart in range(4):
                if not deadline_ok():
                    break
                params_init = np.array([
                    rng.uniform(-np.pi, np.pi),
                    rng.uniform(0.5, 5.0),
                    rng.uniform(-np.pi, np.pi),
                    rng.uniform(0.5, 5.0),
                    rng.uniform(0.5, 5.0),
                    1.0,
                ])
                
                res = minimize(
                    lambda p: _golay_smooth_obj(p, beta=30.0),
                    params_init,
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi), (0.1, 10.0), (-np.pi, np.pi), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0)],
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0}
                )
                if res.fun < np.inf:
                    h_candidate = _golay_complementary_h(res.x)
                    update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 3: Windowed Zadoff-Chu with limited DE
    if deadline_ok():
        try:
            bounds_wzc = [
                (-np.pi, np.pi),
                (-5.0, 5.0),
                (-10.0, 10.0),
                (0.1, 10.0),
                (0.3, 1.5),
                (0.2, 0.8),
                (0.0, 1.0),
            ]
            
            res_de = differential_evolution(
                _windowed_zc_obj,
                bounds_wzc,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-5,
                polish=True,
                init='sobol',
                updating='immediate',
                workers=1,
            )
            if res_de.fun < np.inf:
                h_candidate = _windowed_zadoff_chu_h(res_de.x)
                update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 4: DFT refinement from best_h
    if deadline_ok():
        try:
            base_val = np.mean(best_h)
            h_centered = best_h - base_val
            coeffs_init, _, _, _ = np.linalg.lstsq(_BASIS_T, h_centered, rcond=None)
            
            params_init = np.concatenate([coeffs_init, [base_val]])
            bounds_dft = [(-10.0, 10.0)] * (2 * _K_DFT) + [(0.1, 10.0)]
            
            res_dft = minimize(
                lambda p: _dft_smooth_obj(p, beta=25.0),
                params_init,
                method='L-BFGS-B',
                bounds=bounds_dft,
                options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            if res_dft.fun < np.inf:
                coeffs_opt = res_dft.x[:-1]
                base_opt = res_dft.x[-1]
                h_candidate = _dft_h(coeffs_opt, base_opt)
                update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 5: Final high-dimensional polish
    if deadline_ok():
        try:
            res_final = minimize(
                lambda hv: _smooth_max_value(hv, beta=15.0),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-12, 'iprint': 0}
            )
            if res_final.fun < np.inf:
                h_final = res_final.x
                v_final = _value(h_final)
                if v_final < best_v and np.all(np.isfinite(h_final)) and abs(np.sum(h_final)) > 1e-12:
                    best_h = h_final
                    best_v = v_final
        except Exception:
            pass
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    final_val = _value(best_h)
    if not np.isfinite(final_val) or final_val > 2.0:
        return np.ones(N)
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Score: {2.0 / _value(h)}")
