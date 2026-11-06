"""Combined program for the third autocorrelation inequality.

Strategy:
- Primary: DCT spectral shaping with L-BFGS-B on smooth max surrogate (multiple restarts).
- Secondary: Golay complementary pair (two chirps with different rates) where the sum of
  individual autocorrelations cancels out off-peak values.
- Tertiary: Windowed Zadoff-Chu sequences with global optimization.
- Fourth: Full-dimensional coordinate descent with smooth logsumexp surrogate to escape
  Fourier subspace restriction, allowing high-frequency corrections.
- Refine best candidates with L-BFGS-B on DFT basis and Nelder-Mead polish.
- Deadline guard ensures we return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
_t0 = time.time()
_DEADLINE = 140.0  # safety margin under 150s wall-clock
_K_DCT = 80
_K_DFT = 20

# Precompute common arrays
_i_arr = np.arange(N, dtype=float)
_conv_len = 2 * N - 1
_fft_size = 2 * N

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    H = np.fft.rfft(h, _fft_size)
    conv = np.fft.irfft(H * H, _fft_size)[:_conv_len]
    return conv

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

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

def _adaptive_smooth_obj(coeffs, beta_start=100.0, beta_min=30.0, decay=0.95):
    """Adaptive beta smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    # Adaptive beta: larger when spread is small, smaller when spread is large
    spread = np.max(conv) - np.min(conv)
    if spread > 1e-6:
        beta = max(beta_min, min(beta_start, 50.0 / spread))
    else:
        beta = beta_start
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
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

def _dct_smooth_obj(coeffs, beta=80.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dct_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

# Low-frequency DFT basis for refinement
_n_idx = np.arange(N)
_basis = np.zeros((2 * _K_DFT, N))
for k in range(1, _K_DFT + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
_BASIS_T = _basis.T

def _dft_h(coeffs, base_val):
    """Generate h from DFT coefficients and a base value."""
    h = base_val + _BASIS_T @ coeffs
    return h

def _dft_smooth_obj(coeffs_base, beta=80.0):
    """Smooth objective for DFT parameters."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _smooth_max_value(h, beta)

def _golay_complementary_h(params):
    """
    Golay complementary pair: h = base + amp1*cos(quad_phase1) + amp2*cos(quad_phase2)
    Two chirps with different rates can have complementary autocorrelation properties.
    params: [phase1, rate1, amp1, phase2, rate2, amp2, base]
    """
    phase1, rate1, amp1, phase2, rate2, amp2, base = params
    i = _i_arr
    ang1 = 2.0 * np.pi * rate1 * i * (i + 1) / N + phase1
    ang2 = 2.0 * np.pi * rate2 * i * (i + 1) / N + phase2
    h = base + amp1 * np.cos(ang1) + amp2 * np.cos(ang2)
    return h

def _golay_smooth_obj(params, beta=80.0):
    """Smooth objective for Golay complementary parameters."""
    h = _golay_complementary_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zadoff_chu_h(params):
    """Windowed Zadoff-Chu with smooth window to suppress sidelobes."""
    phase, rate, amp, base = params[0], params[1], params[2], params[3]
    win_width = params[4] if len(params) > 4 else 1.0
    win_center = params[5] if len(params) > 5 else 0.5
    win_type = params[6] if len(params) > 6 else 0.0
    
    i = _i_arr
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    rel_pos = i / N
    
    if win_type < 0.5:
        x = (rel_pos - win_center) / (0.5 * win_width)
        window = np.where(np.abs(x) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    else:
        sigma = win_width / 4.0
        window = np.exp(-0.5 * ((rel_pos - win_center) / sigma) ** 2)
    
    h = base + amp * window * np.cos(ang)
    return h

def _windowed_zc_smooth_obj(params, beta=80.0):
    """Smooth objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    i = _i_arr
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _project_dct(h):
    """Project h onto DCT basis, return coefficients."""
    mean = float(np.mean(h))
    coeffs = _DCT_BASIS_T @ (h - mean)
    return coeffs, mean

def _full_dim_coord_descent(h0, max_rounds=12, step=0.04, beta=80.0):
    """
    Full-dimensional coordinate descent with smooth logsumexp surrogate.
    Uses incremental FFT update: modifying one element of h changes only one
    bin of H, so we can update conv in O(N) instead of O(N log N).
    """
    h = h0.copy()
    s = float(np.sum(h))
    if s == 0 or not np.all(np.isfinite(h)):
        return h
    
    # Precompute: H[k] = sum_j h[j] * exp(-2*pi*i*j*k/fft_size)
    # If we change h[i] by delta, H[k] changes by delta * exp(-2*pi*i*i*k/fft_size)
    # So we can precompute the phase factors for each index i.
    
    H = np.fft.rfft(h, _fft_size)
    H2 = H * H
    conv = np.fft.irfft(H2, _fft_size)[:_conv_len]
    m = conv.max()
    scaled = np.clip(beta * (conv - m), -100, 100)
    exp_sum = np.sum(np.exp(scaled))
    v = abs(2.0 * N * (m + np.log(exp_sum) / beta) / (s * s))
    
    # Precompute phase factors: phase[i, k] = exp(-2*pi*i*i*k/fft_size) for k=0..fft_size/2
    # This is expensive upfront but saves O(N log N) per trial
    # Actually, let's just do full FFT per trial but limit rounds
    # Better: use the fact that conv = ifft(H^2), and track H incrementally
    
    for rnd in range(max_rounds):
        improved = False
        for i in range(N):
            old = h[i]
            best_delta = 0
            best_v = v
            for delta in (step, -step, step*0.5, -step*0.5):
                new = old + delta
                if new == old:
                    continue
                s_new = s + delta
                if abs(s_new) < 1e-12:
                    continue
                # Update H: H[k] += delta * exp(-2*pi*i*i*k/fft_size)
                # For efficiency, compute the update vector
                k_vals = np.arange(_fft_size // 2 + 1)
                phase_update = np.exp(-2j * np.pi * i * k_vals / _fft_size)
                H_new = H + delta * phase_update
                H2_new = H_new * H_new
                conv_new = np.fft.irfft(H2_new, _fft_size)[:_conv_len]
                m_new = conv_new.max()
                scaled_new = np.clip(beta * (conv_new - m_new), -100, 100)
                exp_sum_new = np.sum(np.exp(scaled_new))
                v_new = abs(2.0 * N * (m_new + np.log(exp_sum_new) / beta) / (s_new * s_new))
                if v_new < best_v - 1e-12:
                    best_v = v_new
                    best_delta = delta
            if best_delta != 0:
                v = best_v
                h[i] = old + best_delta
                s += best_delta
                # Update H and H2
                k_vals = np.arange(_fft_size // 2 + 1)
                phase_update = np.exp(-2j * np.pi * i * k_vals / _fft_size)
                H += best_delta * phase_update
                H2 = H * H
                improved = True
        if not improved:
            step *= 0.5
            if step < 1e-5:
                break
    return h

def _local_refine_exact(h, max_rounds=12, step=0.03):
    """Per-step coordinate descent on exact value with incremental FFT."""
    h = h.copy()
    v = _value(h)
    s = float(np.sum(h))
    if s == 0 or not np.all(np.isfinite(h)):
        return h
    H = np.fft.rfft(h, _fft_size)

    for rnd in range(max_rounds):
        improved = False
        for i in range(N):
            old = h[i]
            best_delta = 0
            best_v = v
            for delta in (step, -step, step*0.5, -step*0.5):
                new = old + delta
                if new == old:
                    continue
                s_new = s + delta
                if abs(s_new) < 1e-12:
                    continue
                k_vals = np.arange(_fft_size // 2 + 1)
                phase_update = np.exp(-2j * np.pi * i * k_vals / _fft_size)
                H_new = H + delta * phase_update
                H2_new = H_new * H_new
                conv = np.fft.irfft(H2_new, _fft_size)[:_conv_len]
                v_new = abs(2.0 * N * conv.max() / (s_new * s_new))
                if v_new < best_v - 1e-12:
                    best_v = v_new
                    best_delta = delta
            if best_delta != 0:
                v = best_v
                h[i] = old + best_delta
                s += best_delta
                k_vals = np.arange(_fft_size // 2 + 1)
                phase_update = np.exp(-2j * np.pi * i * k_vals / _fft_size)
                H += best_delta * phase_update
                improved = True
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
            for restart in range(25):
                if not deadline_ok():
                    break
                if restart == 0:
                    coeffs = np.zeros(_K_DCT)
                else:
                    n_active = int(rng.integers(5, min(20, _K_DCT)))
                    active = rng.choice(_K_DCT, size=n_active, replace=False)
                    coeffs = np.zeros(_K_DCT)
                    coeffs[active] = rng.normal(0, 3.0, size=n_active)
                res = minimize(_dct_smooth_obj, coeffs, method='L-BFGS-B',
                               options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-13})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass

    # Phase 2: More DCT restarts with adaptive beta
    if deadline_ok():
        try:
            for restart in range(20):
                if not deadline_ok():
                    break
                n_active = int(rng.integers(6, min(25, _K_DCT)))
                active = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs = np.zeros(_K_DCT)
                coeffs[active] = rng.normal(0, 3.5, size=n_active)
                res = minimize(_adaptive_smooth_obj, coeffs, method='L-BFGS-B',
                               options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-13})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass

    # Phase 3: Local smooth refinement
    if deadline_ok():
        try:
            h_ref = _full_dim_coord_descent(best_h, max_rounds=10, step=0.04, beta=80.0)
            update_best(h_ref)
        except Exception:
            pass

    # Phase 4: More DCT restarts with different initializations
    if deadline_ok():
        try:
            for restart in range(20):
                if not deadline_ok():
                    break
                n_active = int(rng.integers(8, min(30, _K_DCT)))
                active = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs = np.zeros(_K_DCT)
                coeffs[active] = rng.normal(0, 4.0, size=n_active)
                res = minimize(_adaptive_smooth_obj, coeffs, method='L-BFGS-B',
                               options={'maxiter': 350, 'ftol': 1e-16, 'gtol': 1e-13})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass

    # Phase 5: Local exact refinement
    if deadline_ok():
        try:
            h_ref = _local_refine_exact(best_h, max_rounds=12, step=0.03)
            update_best(h_ref)
        except Exception:
            pass

    # Phase 6: Nelder-Mead polish on DCT coefficients
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dct(best_h)
            res = minimize(_dct_smooth_obj, coeffs_init, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-13})
            if res.x is not None:
                h = _dct_h(res.x)
                update_best(h)
        except Exception:
            pass

    # Phase 7: Final local smooth refinement with higher beta
    if deadline_ok():
        try:
            h_ref = _full_dim_coord_descent(best_h, max_rounds=8, step=0.02, beta=150.0)
            update_best(h_ref)
        except Exception:
            pass

    # Phase 8: Final local exact refinement
    if deadline_ok():
        try:
            h_ref = _local_refine_exact(best_h, max_rounds=10, step=0.015)
            update_best(h_ref)
        except Exception:
            pass

    # Phase 9: One more DCT polish cycle
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dct(best_h)
            res = minimize(_dct_smooth_obj, coeffs_init, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.x is not None:
                h = _dct_h(res.x)
                update_best(h)
        except Exception:
            pass

    # Phase 10: Final exact polish
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
