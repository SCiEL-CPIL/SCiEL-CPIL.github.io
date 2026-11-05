"""Combined program: chirp + Zadoff-Chu + DFT basis + FIR filter + smooth-max L-BFGS-B refinement.

Strategy:
- Use FFT-based convolution for speed (enables more optimization iterations).
- Keep the chirp-based parameterization as global search space.
- Add Zadoff-Chu inspired sequences (near-zero non-zero autocorrelation) as additional candidates.
- Add FIR low-pass filter design as an additional parameterization.
- Use differential evolution for global search over chirp parameters.
- Refine using L-BFGS-B on a low-frequency DFT basis with a smooth max surrogate.
- Also try Nelder-Mead polish on chirp parameters.
- Deadline guard ensures we return before 150s.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
_t0 = time.time()
_DEADLINE = 130.0  # safety margin under 150s wall-clock

# Precompute FFT size for convolution
_CONV_N = 2 * N - 1  # length of full convolution
_FFT_N = 1
while _FFT_N < _CONV_N:
    _FFT_N *= 2

def _fft_convolve(h):
    """Fast convolution using FFT."""
    h = np.asarray(h, dtype=float)
    H = np.fft.rfft(h, _FFT_N)
    conv = np.fft.irfft(H * H, _FFT_N)[:_CONV_N]
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

def _smooth_max_value(h, alpha=50.0):
    """Smooth surrogate for the max of convolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    # Soft max via log-sum-exp
    scaled = conv * alpha
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = log_sum_exp / alpha
    return abs(2.0 * N * soft_max / (s * s))

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    # Quadratic phase: phase + rate * (i/N)^2
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _fir_lowpass_h(params):
    """Generate h from FIR low-pass filter parameters.
    
    params: [cutoff, order_factor, amp, base]
    The FIR filter coefficients approximate an ideal low-pass with smooth windowing.
    """
    cutoff, order_factor, amp, base = params
    # cutoff in [0, 1] normalized frequency
    cutoff = np.clip(cutoff, 0.01, 0.99)
    # order_factor controls the effective length
    order_factor = np.clip(order_factor, 0.1, 3.0)
    
    i = np.arange(N, dtype=float)
    # Sinc-like kernel with windowing
    x = (i - N / 2.0) / (N * order_factor)
    # Avoid division by zero
    x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)
    sinc = np.sin(np.pi * cutoff * x_safe) / (np.pi * x_safe)
    # Apply a smooth window (raised cosine)
    w = 0.5 * (1 + np.cos(2 * np.pi * i / N))
    h = base + amp * sinc * w
    return h

def _fir_lowpass_obj(params):
    """Objective for FIR low-pass parameters."""
    h = _fir_lowpass_h(params)
    return _value(h)

def _zadoff_chu_h(params):
    """Generate h from Zadoff-Chu-like parameters: [phase, rate, amp, base].
    
    Zadoff-Chu sequences have constant amplitude and near-zero non-zero
    autocorrelation. We use h[i] = base + amp * cos(phase + rate * i * (i - N/2) / N)
    which approximates the quadratic phase of ZC sequences.
    """
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    # ZC-like quadratic phase centered
    ang = phase + rate * i * (i - N / 2.0) / N
    h = base + amp * np.cos(ang)
    return h

def _zadoff_chu_obj(params):
    """Objective for Zadoff-Chu parameters."""
    h = _zadoff_chu_h(params)
    return _value(h)

def _chirp_obj(params):
    """Objective for chirp parameters."""
    h = _chirp_h(params)
    return _value(h)

# Low-frequency DFT basis for refinement
_K = 20  # number of frequency components (increased from 15)
_n_idx = np.arange(N)
_basis = np.zeros((2 * _K, N))
for k in range(1, _K + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
_BASIS_T = _basis.T  # (N, 2*K)

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

def _dft_smooth_obj(coeffs_base, alpha=50.0):
    """Smooth objective for DFT parameters with adaptive alpha."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _smooth_max_value(h, alpha=alpha)

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    # Phase 1: Global search via differential evolution on chirp parameters
    # params: [phase, rate, amp, base]
    bounds = [
        (-np.pi, np.pi),      # phase
        (-200.0, 200.0),      # rate (quadratic phase coefficient)
        (-3.0, 3.0),          # amplitude
        (-3.0, 3.0),          # base
    ]
    
    try:
        if deadline_ok():
            res = differential_evolution(
                _chirp_obj,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=10,
                init='sobol',
                polish=False,
                update_interval=1,
            )
            if res.x is not None:
                h = _chirp_h(res.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
    except Exception:
        pass
    
    # Phase 1b: Global search via differential evolution on Zadoff-Chu parameters
    if deadline_ok():
        try:
            bounds_zc = [
                (-np.pi, np.pi),      # phase
                (-50.0, 50.0),        # rate (ZC quadratic coefficient)
                (-3.0, 3.0),          # amplitude
                (-3.0, 3.0),          # base
            ]
            res_zc = differential_evolution(
                _zadoff_chu_obj,
                bounds_zc,
                seed=123,
                maxiter=40,
                tol=1e-6,
                popsize=10,
                init='sobol',
                polish=False,
                update_interval=1,
            )
            if res_zc.x is not None:
                h = _zadoff_chu_h(res_zc.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 1c: Global search via differential evolution on FIR low-pass parameters
    if deadline_ok():
        try:
            bounds_fir = [
                (0.05, 0.95),         # cutoff
                (0.1, 3.0),           # order factor
                (-3.0, 3.0),          # amplitude
                (-3.0, 3.0),          # base
            ]
            res_fir = differential_evolution(
                _fir_lowpass_obj,
                bounds_fir,
                seed=456,
                maxiter=40,
                tol=1e-6,
                popsize=10,
                init='sobol',
                polish=False,
                update_interval=1,
            )
            if res_fir.x is not None:
                h = _fir_lowpass_h(res_fir.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish on chirp parameters
    if deadline_ok():
        try:
            # Start from the best found, or from a reasonable default
            start_params = [0.0, 10.0, 1.0, 1.0]
            # If we have a better best_h, try to estimate chirp params from it
            # For simplicity, just run Nelder-Mead from a few starting points
            for start in [
                [0.0, 10.0, 1.0, 1.0],
                [0.0, -10.0, 1.0, 1.0],
                [np.pi/2, 50.0, 0.5, 1.0],
                [0.0, 100.0, 1.5, 0.5],
            ]:
                if not deadline_ok():
                    break
                res2 = minimize(
                    _chirp_obj,
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200, 'maxfev': 300},
                )
                if res2.x is not None:
                    h = _chirp_h(res2.x)
                    v = _value(h)
                    if v < best_v:
                        best_v = v
                        best_h = h.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B refinement on DFT basis with smooth max surrogate
    # Initialize DFT coefficients from best_h
    if deadline_ok():
        try:
            # Compute DFT coefficients from best_h
            h_best = best_h
            mean_h = np.mean(h_best)
            # Project onto DFT basis using FFT for speed
            H_fft = np.fft.rfft(h_best - mean_h, N)
            coeffs_init = np.zeros(2 * _K)
            for k in range(1, _K + 1):
                # For real signals, FFT[k] = (c_k - i*s_k) * N/2 approximately
                # We want h = sum c_k*cos + s_k*sin
                # FFT[k] = sum h[n]*exp(-i*2*pi*k*n/N)
                # = (c_k - i*s_k) * N/2 for k > 0
                fft_val = H_fft[k]
                coeffs_init[2 * (k - 1)] = 2.0 * np.real(fft_val) / N
                coeffs_init[2 * (k - 1) + 1] = -2.0 * np.imag(fft_val) / N
            # Full parameter vector: [coeffs..., base]
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            # Use smooth max surrogate for better gradient behavior
            res3 = minimize(
                lambda x: _dft_smooth_obj(x, alpha=50.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            if res3.x is not None:
                h = _dft_h(res3.x[:-1], res3.x[-1])
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead polish on DFT basis
    if deadline_ok():
        try:
            # Re-initialize from current best_h using FFT
            h_best = best_h
            mean_h = np.mean(h_best)
            H_fft = np.fft.rfft(h_best - mean_h, N)
            coeffs_init = np.zeros(2 * _K)
            for k in range(1, _K + 1):
                fft_val = H_fft[k]
                coeffs_init[2 * (k - 1)] = 2.0 * np.real(fft_val) / N
                coeffs_init[2 * (k - 1) + 1] = -2.0 * np.imag(fft_val) / N
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res4 = minimize(
                _dft_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 800, 'maxfev': 2000},
            )
            if res4.x is not None:
                h = _dft_h(res4.x[:-1], res4.x[-1])
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish using smooth max with tighter alpha
    if deadline_ok():
        try:
            h_best = best_h
            mean_h = np.mean(h_best)
            H_fft = np.fft.rfft(h_best - mean_h, N)
            coeffs_init = np.zeros(2 * _K)
            for k in range(1, _K + 1):
                fft_val = H_fft[k]
                coeffs_init[2 * (k - 1)] = 2.0 * np.real(fft_val) / N
                coeffs_init[2 * (k - 1) + 1] = -2.0 * np.imag(fft_val) / N
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res5 = minimize(
                lambda x: _dft_smooth_obj(x, alpha=200.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            if res5.x is not None:
                h = _dft_h(res5.x[:-1], res5.x[-1])
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 6: Final exact-value polish with very tight tolerances
    if deadline_ok():
        try:
            h_best = best_h
            mean_h = np.mean(h_best)
            H_fft = np.fft.rfft(h_best - mean_h, N)
            coeffs_init = np.zeros(2 * _K)
            for k in range(1, _K + 1):
                fft_val = H_fft[k]
                coeffs_init[2 * (k - 1)] = 2.0 * np.real(fft_val) / N
                coeffs_init[2 * (k - 1) + 1] = -2.0 * np.imag(fft_val) / N
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res6 = minimize(
                _dft_obj,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14},
            )
            if res6.x is not None:
                h = _dft_h(res6.x[:-1], res6.x[-1])
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
