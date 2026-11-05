"""
Third autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2 over h in R^N, N=400.

Structural change: Represent h as a finite impulse response (FIR) filter designed to approximate
an ideal low-pass filter with a specific transition band. The filter coefficients are directly
optimized via gradient descent to minimize the peak of the autoconvolution, exploiting the
time-domain symmetry and smooth decay of the ideal sinc-like kernel to naturally suppress sidelobes.

The key insight is that a well-designed low-pass filter has a flat frequency response in the passband
and rapid roll-off in the stopband. When applied to a signal, this suppresses high-frequency components
that contribute to sidelobes in the autocorrelation. By optimizing the FIR coefficients directly,
we can shape the autocorrelation peak more effectively than parameterized approaches.

We use:
1. A sinc-based initial FIR design (Kaiser windowed sinc)
2. Direct gradient-based optimization of the FIR coefficients
3. Smooth max surrogate for differentiable optimization
4. Deadline guard for safety
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
_t0 = time.time()
_DEADLINE = 130.0  # safety margin under 150s wall-clock
_FIR_ORDER = 32  # number of FIR filter coefficients (even number for symmetric filter)

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

def _smooth_max_value(h, beta=50.0):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    smooth_m = m + np.log(np.sum(np.exp(shifted))) / beta
    return abs(2.0 * N * smooth_m / (s * s))

def _design_kaiser_sinc_fir(n_taps, cutoff, beta):
    """
    Design a low-pass FIR filter using the windowed sinc method with a Kaiser window.
    
    Parameters:
    -----------
    n_taps : int
        Number of filter taps (should be even for Type II linear phase FIR)
    cutoff : float
        Cutoff frequency in cycles per sample (0 < cutoff < 0.5)
    beta : float
        Kaiser window beta parameter (higher = narrower main lobe, lower sidelobes)
    
    Returns:
    --------
    h : ndarray
        FIR filter coefficients of length n_taps
    """
    n = n_taps
    # For Type II linear phase FIR (even number of taps), use symmetric coefficients
    # The ideal low-pass impulse response is sinc(n*cutoff)
    m = np.arange(n) - (n - 1) / 2.0  # centered indices
    # Avoid division by zero at m=0
    with np.errstate(divide='ignore', invalid='ignore'):
        h_ideal = np.where(m == 0, cutoff, np.sin(np.pi * cutoff * m) / (np.pi * m))
    
    # Kaiser window
    alpha = np.sqrt((n - 1) ** 2 / 4.0)
    if alpha == 0:
        window = np.ones(n)
    else:
        # Kaiser window formula
        x = np.arange(n)
        x = (x - (n - 1) / 2.0) / ((n - 1) / 2.0)
        window = np.i0(beta * np.sqrt(np.maximum(0, 1 - x**2))) / np.i0(beta)
    
    return h_ideal * window

def _fir_to_heights(fir_coeffs, base_val=0.0):
    """
    Convert FIR filter coefficients to step heights.
    
    The idea: apply the FIR filter to a square wave (or other test signal) to generate
    the step heights. The FIR filter shapes the frequency content to suppress sidelobes.
    
    Parameters:
    -----------
    fir_coeffs : ndarray
        FIR filter coefficients
    base_val : float
        Base offset to add to all heights
    
    Returns:
    --------
    h : ndarray
        Step heights of length N
    """
    n_fir = len(fir_coeffs)
    
    # Create a square wave test signal
    i = np.arange(N, dtype=float)
    t = i / N
    square_wave = np.sign(np.sin(2 * np.pi * t))
    square_wave[square_wave == 0] = 1.0
    
    # Apply FIR filter to the square wave
    # Use 'same' mode to keep output length N
    h_filtered = np.convolve(square_wave, fir_coeffs, mode='same')
    
    # Scale to have reasonable magnitude
    max_val = np.max(np.abs(h_filtered))
    if max_val > 0:
        h_filtered = h_filtered / max_val * 1.0
    
    return base_val + h_filtered

def _fir_smooth_obj(params, beta=50.0):
    """
    Smooth objective for FIR filter parameters.
    
    Parameters:
    -----------
    params : ndarray
        FIR filter coefficients (length _FIR_ORDER) plus base value
    beta : float
        Smoothness parameter for max approximation
    
    Returns:
    --------
    float
        Smooth objective value
    """
    fir_coeffs = params[:-1]
    base_val = params[-1]
    h = _fir_to_heights(fir_coeffs, base_val)
    return _smooth_max_value(h, beta)

def _fir_value(params):
    """Exact objective for FIR filter parameters."""
    fir_coeffs = params[:-1]
    base_val = params[-1]
    h = _fir_to_heights(fir_coeffs, base_val)
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
    
    # Phase 1: Kaiser windowed sinc FIR design with multiple cutoffs and betas
    if deadline_ok():
        try:
            cutoffs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
            betas = [3.0, 5.0, 7.0, 9.0, 11.0]
            
            for cutoff in cutoffs:
                if not deadline_ok():
                    break
                for beta_k in betas:
                    if not deadline_ok():
                        break
                    fir_init = _design_kaiser_sinc_fir(_FIR_ORDER, cutoff, beta_k)
                    params_init = np.concatenate([fir_init, [0.0]])
                    
                    # Optimize with L-BFGS-B
                    res = minimize(
                        lambda p: _fir_smooth_obj(p, beta=30.0),
                        params_init,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0}
                    )
                    if res.fun < np.inf:
                        h_candidate = _fir_to_heights(res.x[:-1], res.x[-1])
                        update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 2: Random FIR initializations
    if deadline_ok():
        try:
            rng = np.random.default_rng(42)
            for restart in range(10):
                if not deadline_ok():
                    break
                # Random FIR coefficients with symmetric structure
                half_len = _FIR_ORDER // 2
                fir_half = rng.normal(0, 0.5, size=half_len)
                # Make symmetric
                fir_init = np.concatenate([fir_half, fir_half[::-1]])
                base_init = rng.normal(0, 0.5)
                params_init = np.concatenate([fir_init, [base_init]])
                
                res = minimize(
                    lambda p: _fir_smooth_obj(p, beta=30.0),
                    params_init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0}
                )
                if res.fun < np.inf:
                    h_candidate = _fir_to_heights(res.x[:-1], res.x[-1])
                    update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 3: Direct high-dimensional polish on the best h found
    if deadline_ok():
        try:
            res_coarse = minimize(
                lambda hv: _smooth_max_value(hv, beta=20.0),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            if res_coarse.fun < np.inf and np.all(np.isfinite(res_coarse.x)):
                h_coarse = res_coarse.x
                v_coarse = _value(h_coarse)
                if v_coarse < best_v and abs(np.sum(h_coarse)) > 1e-12:
                    best_h = h_coarse
                    best_v = v_coarse
        except Exception:
            pass
    
    if deadline_ok():
        try:
            res_fine = minimize(
                lambda hv: _smooth_max_value(hv, beta=100.0),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14, 'iprint': 0}
            )
            if res_fine.fun < np.inf and np.all(np.isfinite(res_fine.x)):
                h_fine = res_fine.x
                v_fine = _value(h_fine)
                if v_fine < best_v and abs(np.sum(h_fine)) > 1e-12:
                    best_h = h_fine
                    best_v = v_fine
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
