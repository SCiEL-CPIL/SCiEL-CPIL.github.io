"""Combined program: Chaotic-masked Zadoff-Chu + Golay + DCT spectral shaping +
gradient-based coordinate descent + Nesterov acceleration + high-precision polish.

Strategy:
- Phase 1: Chaotic-masked complex Zadoff-Chu global search (deterministic high-entropy
  decorrelation breaks periodic sidelobe structure).
- Phase 2: Golay complementary pair global search.
- Phase 3: Barker-like multi-harmonic search.
- Phase 4: DCT spectral shaping with L-BFGS-B on smooth max surrogate.
- Phase 5: Fast gradient-based coordinate descent on full h (analytic gradient of
  logsumexp smooth surrogate, 1 FFT per coordinate per sweep).
- Phase 6: Nesterov-accelerated gradient descent on full h vector.
- Phase 7: High-precision DFT basis polish with tight tolerances.
- Deadline guard: explicit wall-clock check, return best before 145s.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
_t0 = time.time()
_DEADLINE = 140.0  # safety margin under 150s
_K_DCT = 40
_K_DFT = 20

_i_arr = np.arange(N, dtype=float)
_n_idx = np.arange(N)


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


def _smooth_max_grad(h, beta=80.0):
    """
    Compute gradient of smooth max surrogate w.r.t. h.
    Uses analytic gradient of logsumexp and chain rule through FFT convolution.
    Returns (value, grad) where grad has same shape as h.
    """
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf, np.zeros_like(h)
    
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    if sum_exp == 0:
        return np.inf, np.zeros_like(h)
    
    # Weights for logsumexp gradient
    w = exp_scaled / sum_exp  # shape (2N-1,)
    
    # log_sum_exp = m + log(sum_exp)/beta
    # d(log_sum_exp)/d(conv[j]) = w[j] / beta
    # conv[j] = sum_i h[i]*h[i+j] (for j>=0) or h[i]*h[i-j+N-1]... 
    # Actually conv from FFT is the full linear convolution.
    # d(conv[j])/d(h[k]) = h[k-j] if 0<=k-j<N else 0  (for the linear conv)
    # More precisely: conv[j] = sum_{i} h[i] * h[i+j-(N-1)] for the full conv
    # Let's use: conv[j] = sum_{i=0}^{N-1} h[i] * h[j-i] where h is zero-padded
    # d(conv[j])/d(h[k]) = h[j-k] (zero-padded)
    
    # grad_h[k] = sum_j w[j]/beta * d(conv[j])/d(h[k])
    #           = sum_j w[j]/beta * h[j-k]  (with h zero-padded)
    # This is a cross-correlation of w with h.
    
    # Use FFT for efficiency
    n_conv = 2 * N - 1
    size_fft = 4 * N  # power of 2 > 2N-1
    W = np.fft.rfft(w, n=size_fft)
    H = np.fft.rfft(h, n=size_fft)
    # cross-correlation: sum_j w[j] * h[j-k] = ifft(conj(W) * H)
    # Actually: (w * h)[k] = sum_j w[j] h[j-k] = ifft(fft(w) * conj(fft(h)))... 
    # Let me think more carefully.
    # sum_j w[j] * h[j-k] for j=0..n_conv-1, k=0..N-1
    # This is the cross-correlation of w and h at lag k.
    # In FFT: ifft(fft(w) * conj(fft(h))) gives sum_j w[j] conj(h[j-k])
    # Since h is real: sum_j w[j] h[j-k]
    
    cross = np.fft.irfft(W * np.conj(np.fft.rfft(H, n=size_fft)), n=size_fft)[:N]
    
    # grad of log_sum_exp w.r.t. h[k] = cross[k] / beta
    # But we also need to account for the abs() and the 2N/(s^2) factor.
    # value = abs(2N * LSE / s^2) where LSE = m + log(sum_exp)/beta
    # Let's compute d(value)/d(h[k])
    # If 2N*LSE/s^2 > 0: value = 2N*LSE/s^2, d/dh = 2N*(dLSE/dh)/s^2 - 2N*LSE*2s*(ds/dh)/s^4
    # ds/dh[k] = 1
    # If 2N*LSE/s^2 < 0: value = -2N*LSE/s^2, d/dh = -2N*(dLSE/dh)/s^2 + 2N*LSE*2s/s^4
    
    LSE = m + np.log(sum_exp) / beta
    factor = 2.0 * N / (s * s)
    inner = factor * LSE
    
    if inner >= 0:
        # value = inner, d(value)/dh[k] = factor * dLSE/dh[k] - 2*inner/s
        grad_LSE = cross / beta
        grad = factor * grad_LSE - 2.0 * inner / s
    else:
        # value = -inner, d(value)/dh[k] = -factor * dLSE/dh[k] + 2*inner/s
        grad_LSE = cross / beta
        grad = -factor * grad_LSE + 2.0 * inner / s
    
    val = abs(inner)
    return val, grad


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


def _dct_h(coeffs):
    """Generate h from DCT coefficients."""
    return _DCT_BASIS @ coeffs


def _dct_smooth_obj(coeffs, beta=80.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)


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


def _dft_obj(coeffs_base):
    """Objective for DFT parameters: [coeffs..., base]."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _value(h)


# Low-frequency DFT basis for refinement
_basis = np.zeros((2 * _K_DFT, N))
for k in range(1, _K_DFT + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
_BASIS_T = _basis.T


def _project_dft(h):
    """Project h onto DFT basis + base, return (coeffs, base_val)."""
    mean_h = float(np.mean(h))
    coeffs = np.zeros(2 * _K_DFT)
    target = h - mean_h
    for k in range(1, _K_DFT + 1):
        angle = 2.0 * np.pi * k * _n_idx / N
        cos_k = np.cos(angle)
        sin_k = np.sin(angle)
        # Use dot products since basis is orthonormal (approximately)
        coeffs[2 * (k - 1)] = np.dot(cos_k, target) / np.dot(cos_k, cos_k)
        coeffs[2 * (k - 1) + 1] = np.dot(sin_k, target) / np.dot(sin_k, sin_k)
    return coeffs, mean_h


def _chaotic_zc_h(params):
    """
    Chaotic-masked Zadoff-Chu sequence.
    h = base + amp * Re( ZC * exp(i * chaotic_phase) )
    where ZC is a Zadoff-Chu sequence and chaotic_phase comes from logistic map.
    
    params: [base, amp, rate, logis_r, logis_x0, phase_offset, mask_rate]
    - rate: ZC rate parameter
    - logis_r: logistic map parameter (3.5-4.0 for chaos)
    - logis_x0: initial condition for logistic map
    - phase_offset: global phase
    - mask_rate: rate of the chaotic phase modulation
    """
    base, amp, rate, logis_r, logis_x0, phase_offset, mask_rate = params
    
    i = _i_arr
    # Zadoff-Chu: exp(-i*pi*rate*i*(i+1)/N)
    zc_phase = -np.pi * rate * i * (i + 1) / N + phase_offset
    
    # Logistic map for chaotic phase
    x = np.zeros(N)
    x[0] = logis_x0
    for j in range(1, N):
        x[j] = logis_r * x[j - 1] * (1.0 - x[j - 1])
    
    # Chaotic phase mask
    chaotic_phase = 2.0 * np.pi * mask_rate * x + phase_offset * 0.5
    
    # Combined: Re(ZC * exp(i*chaotic_phase)) = cos(zc_phase + chaotic_phase)
    h = base + amp * np.cos(zc_phase + chaotic_phase)
    return h


def _chaotic_zc_obj(params):
    h = _chaotic_zc_h(params)
    return _value(h)


def _golay_complementary_h(params):
    """
    Golay complementary pair: h = base + amp1*cos(quad_phase1) + amp2*cos(quad_phase2)
    params: [phase1, rate1, amp1, phase2, rate2, amp2, base]
    """
    phase1, rate1, amp1, phase2, rate2, amp2, base = params
    i = _i_arr
    ang1 = 2.0 * np.pi * rate1 * i * (i + 1) / N + phase1
    ang2 = 2.0 * np.pi * rate2 * i * (i + 1) / N + phase2
    h = base + amp1 * np.cos(ang1) + amp2 * np.cos(ang2)
    return h


def _golay_obj(params):
    h = _golay_complementary_h(params)
    return _value(h)


def _barker_like_h(params):
    """
    Barker-like sequence from multi-harmonic sum.
    h = base + sum_k a_k * cos(2*pi*k*i/N + phi_k)
    """
    base = params[0]
    n_harmonics = 8
    amplitudes = params[1:1 + n_harmonics]
    phases = params[1 + n_harmonics:1 + 2 * n_harmonics]
    i = _i_arr
    h = np.full(N, base)
    for k in range(1, n_harmonics + 1):
        h += amplitudes[k - 1] * np.cos(2.0 * np.pi * k * i / N + phases[k - 1])
    return h


def _barker_like_obj(params):
    h = _barker_like_h(params)
    return _value(h)


def _grad_coord_descent(h0, max_sweeps=20, beta=60.0, lr_init=0.01):
    """
    Fast gradient-based coordinate descent using analytic gradient of smooth surrogate.
    Each sweep: for each coordinate, compute gradient once, take a step.
    Uses line search with backtracking.
    """
    h = h0.copy()
    cur_val, _ = _smooth_max_grad(h, beta)
    lr = lr_init
    
    n_sweeps = 0
    while (time.time() - _t0) < _DEADLINE and n_sweeps < max_sweeps:
        improved = False
        order = np.random.default_rng(n_sweeps + 100).permutation(N)
        
        for idx in order:
            if (time.time() - _t0) >= _DEADLINE:
                return h
            # Compute gradient at current point
            _, grad = _smooth_max_grad(h, beta)
            g = grad[idx]
            if abs(g) < 1e-14:
                continue
            
            # Try step in negative gradient direction
            step = -lr * g
            h_trial = h.copy()
            h_trial[idx] += step
            v_trial, _ = _smooth_max_grad(h_trial, beta)
            
            if v_trial < cur_val:
                h = h_trial
                cur_val = v_trial
                improved = True
            else:
                # Backtrack
                for _bt in range(3):
                    step *= 0.5
                    h_trial = h.copy()
                    h_trial[idx] += step
                    v_trial, _ = _smooth_max_grad(h_trial, beta)
                    if v_trial < cur_val:
                        h = h_trial
                        cur_val = v_trial
                        improved = True
                        break
        
        n_sweeps += 1
        if not improved:
            lr *= 0.7
            if lr < 1e-8:
                break
        else:
            lr = min(lr * 1.1, 0.05)
        beta = min(beta * 1.2, 150.0)
    
    return h


def _nesterov_grad_descent(h0, max_iter=500, beta=80.0, lr_init=0.005):
    """
    Nesterov-accelerated gradient descent on full h vector.
    """
    h = h0.copy()
    v = np.zeros_like(h)  # velocity
    lr = lr_init
    cur_val, _ = _smooth_max_grad(h, beta)
    
    for it in range(max_iter):
        if (time.time() - _t0) >= _DEADLINE:
            break
        
        # Nesterov lookahead
        h_lookahead = h - v * lr
        _, grad_la = _smooth_max_grad(h_lookahead, beta)
        
        # Update velocity with momentum
        momentum = 0.9
        v = momentum * v - lr * grad_la
        
        # Update position
        h_new = h + v
        v_new_val, _ = _smooth_max_grad(h_new, beta)
        
        if v_new_val < cur_val:
            h = h_new
            cur_val = v_new_val
            # Adaptive learning rate
            lr = min(lr * 1.05, 0.02)
        else:
            # Revert and reduce lr
            lr *= 0.8
            v = momentum * v
            if lr < 1e-9:
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
        v = _value(h)
        if v < best_v and np.all(np.isfinite(h)) and np.sum(h) != 0:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: Chaotic-masked Zadoff-Chu global search
    if deadline_ok():
        try:
            bounds_czc = [
                (-2.0, 2.0),    # base
                (-3.0, 3.0),    # amp
                (-50.0, 50.0),  # rate
                (3.5, 4.0),     # logistic r
                (0.01, 0.99),   # logistic x0
                (-np.pi, np.pi),  # phase_offset
                (-20.0, 20.0),  # mask_rate
            ]
            res = differential_evolution(
                _chaotic_zc_obj, bounds_czc, seed=42, maxiter=50, tol=1e-8,
                popsize=15, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _chaotic_zc_h(res.x)
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
                h = _golay_complementary_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: Barker-like multi-harmonic search
    if deadline_ok():
        try:
            n_harm = 8
            bounds_barker = [(-2.0, 2.0)]  # base
            for _ in range(n_harm):
                bounds_barker.append((-2.0, 2.0))  # amplitude
            for _ in range(n_harm):
                bounds_barker.append((-np.pi, np.pi))  # phase
            
            res = differential_evolution(
                _barker_like_obj, bounds_barker, seed=23, maxiter=35, tol=1e-8,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _barker_like_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: DCT spectral shaping with L-BFGS-B (multiple restarts)
    if deadline_ok():
        try:
            rng = np.random.default_rng(42)
            for restart in range(6):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(4, min(14, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 1.0, size=n_active)
                
                res = minimize(_dct_smooth_obj, coeffs, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 5: Gradient-based coordinate descent on full h
    if deadline_ok():
        try:
            h_cd = _grad_coord_descent(best_h, max_sweeps=15, beta=60.0, lr_init=0.01)
            v_cd = _value(h_cd)
            if v_cd < best_v:
                best_v = v_cd
                best_h = h_cd.copy()
        except Exception:
            pass
    
    # Phase 6: Nesterov-accelerated gradient descent
    if deadline_ok():
        try:
            h_ng = _nesterov_grad_descent(best_h, max_iter=300, beta=80.0, lr_init=0.003)
            v_ng = _value(h_ng)
            if v_ng < best_v:
                best_v = v_ng
                best_h = h_ng.copy()
        except Exception:
            pass
    
    # Phase 7: DFT basis L-BFGS-B refinement
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res = minimize(
                _dft_smooth_obj, x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.x is not None:
                h = _dft_h(res.x[:-1], res.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 8: DCT L-BFGS-B refinement
    if deadline_ok():
        try:
            dct_coeffs = np.zeros(_K_DCT)
            for k in range(_K_DCT):
                dct_coeffs[k] = np.sum(best_h * _DCT_BASIS[:, k])
            
            res = minimize(
                lambda c: _dct_smooth_obj(c),
                dct_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.x is not None:
                h = _dct_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 9: Final high-precision DFT polish with exact objective
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res = minimize(
                _dft_obj, x0,
                method='Powell',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 300, 'maxfev': 500}
            )
            if res.x is not None:
                h = _dft_h(res.x[:-1], res.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 10: Final short gradient coordinate descent polish
    if deadline_ok():
        try:
            h_final = _grad_coord_descent(best_h, max_sweeps=5, beta=100.0, lr_init=0.003)
            v_final = _value(h_final)
            if v_final < best_v:
                best_v = v_final
                best_h = h_final.copy()
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
