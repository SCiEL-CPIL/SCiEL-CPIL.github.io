"""Combined program: DCT spectral shaping + smooth-max surrogate + coordinate descent polish.

Strategy:
- Primary: DCT-II spectral shaping with L-BFGS-B on a smooth log-sum-exp surrogate.
  This keeps the optimization in a low-dimensional, smooth subspace while the
  surrogate captures the non-differentiable max-of-convolution structure.
- Secondary: Full-dimensional coordinate descent with the same smooth surrogate
  to escape the DCT subspace and add high-frequency corrections.
- Refinement: Nelder-Mead polish on the DCT coefficients for the best candidates.
- Deadline guard ensures we return before the 150s wall-clock limit.
"""
import time
import numpy as np
from scipy.optimize import minimize, minimize_nelder_mead

N = 400
_t0 = time.time()
_DEADLINE = 138.0  # Safety margin before 150s kill
_K_DCT = 35        # Number of DCT modes to optimize

# Precompute common arrays
_i_arr = np.arange(N, dtype=float)

def _value(h):
    """Compute the exact objective value (lower is better)."""
    h = np.asarray(h, dtype=float)
    if len(h) != N or not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if s == 0.0:
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

def _smooth_max_value(h, beta=250.0):
    """
    Smooth approximation of max of autoconvolution using log-sum-exp.
    Higher beta better approximates the true max.
    """
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -500, 500)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * log_sum_exp / (s * s))

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

_DCT_BASIS = _dct_basis_matrix(_K_DCT, N)
_DCT_BASIS_T = _DCT_BASIS.T

def _dct_h(coeffs):
    """Generate h from DCT coefficients."""
    return _DCT_BASIS @ coeffs

def _dct_smooth_obj(coeffs, beta=250.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dct_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

def _coord_descent_smooth(h0, max_iter=200, step_size=0.02, beta=250.0, n_trial=60):
    """
    Full-dimensional coordinate descent with smooth logsumexp surrogate.
    Escapes Fourier subspace restriction, allowing high-frequency corrections.
    """
    h = h0.copy()
    best_v = _smooth_max_value(h, beta)
    
    for it in range(max_iter):
        if (time.time() - _t0) > _DEADLINE:
            break
        improved = False
        # Try random coordinates
        idx = np.random.choice(N, size=n_trial, replace=False)
        for i in idx:
            old_val = h[i]
            # Try several step sizes
            for delta in [step_size, -step_size, 2*step_size, -2*step_size,
                         0.5*step_size, -0.5*step_size]:
                h[i] = old_val + delta
                v = _smooth_max_value(h, beta)
                if v < best_v:
                    best_v = v
                    improved = True
                    break
                h[i] = old_val
            if not improved:
                h[i] = old_val
        if not improved:
            # Try smaller steps on a subset
            for i in idx[:20]:
                old_val = h[i]
                for delta in [step_size*0.3, -step_size*0.3, step_size*0.1, -step_size*0.1]:
                    h[i] = old_val + delta
                    v = _smooth_max_value(h, beta)
                    if v < best_v:
                        best_v = v
                        improved = True
                        break
                    h[i] = old_val
                if not improved:
                    h[i] = old_val
        if not improved:
            break
    return h

def construct():
    global _t0
    _t0 = time.time()
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    # Known optimal: constant vector gives value = 2.0
    best_h = np.ones(N)
    best_v = 2.0
    
    rng = np.random.default_rng(42)
    
    # Try multiple initializations for DCT optimization
    initializations = []
    # 1. All positive equal weights (approximates constant)
    initializations.append(np.ones(_K_DCT) / _K_DCT)
    # 2. Random small perturbations
    for _ in range(8):
        initializations.append(rng.standard_normal(_K_DCT) * 0.1)
    # 3. Sparse random
    for _ in range(5):
        init = np.zeros(_K_DCT)
        idx = rng.choice(_K_DCT, size=max(1, _K_DCT // 3), replace=False)
        init[idx] = rng.standard_normal(len(idx)) * 0.3
        initializations.append(init)
    # 4. Structured: emphasize low frequencies
    for k_frac in [0.1, 0.3, 0.5]:
        init = np.zeros(_K_DCT)
        num_low = max(1, int(_K_DCT * k_frac))
        init[:num_low] = rng.standard_normal(num_low) * 0.2
        initializations.append(init)
    
    beta = 250.0
    
    for init in initializations:
        if not deadline_ok():
            break
        try:
            res = minimize(
                _dct_smooth_obj,
                init,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                h = _dct_h(res.x)
                v = _value(h)
                if v < best_v - 1e-12:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Coordinate descent polish on the best candidate
    if deadline_ok():
        h_cd = _coord_descent_smooth(best_h, max_iter=150, step_size=0.02, beta=beta)
        v_cd = _value(h_cd)
        if v_cd < best_v - 1e-12:
            best_v = v_cd
            best_h = h_cd.copy()
    
    # Nelder-Mead polish on DCT coefficients for the best candidate
    if deadline_ok():
        # Project best_h onto DCT basis
        coeffs_best = _DCT_BASIS_T @ best_h
        try:
            res_nm = minimize_nelder_mead(
                _dct_obj,
                coeffs_best,
                maxiter=100,
                xatol=1e-10,
                fatol=1e-10
            )
            if res_nm.x is not None:
                h_nm = _dct_h(res_nm.x)
                v_nm = _value(h_nm)
                if v_nm < best_v - 1e-12:
                    best_v = v_nm
                    best_h = h_nm.copy()
        except Exception:
            pass
    
    # Final coordinate descent pass
    if deadline_ok():
        h_cd2 = _coord_descent_smooth(best_h, max_iter=100, step_size=0.01, beta=beta)
        v_cd2 = _value(h_cd2)
        if v_cd2 < best_v - 1e-12:
            best_v = v_cd2
            best_h = h_cd2.copy()
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    
    v_final = _value(best_h)
    if v_final < 2.0 - 1e-12:
        return best_h
    else:
        return np.ones(N)
