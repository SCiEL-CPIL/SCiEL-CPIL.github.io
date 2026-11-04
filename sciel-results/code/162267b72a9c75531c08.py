"""
Strategy: Spectral phase optimization with subgradient polishing.
We parameterize the signal by its Fourier phase spectrum (keeping magnitudes fixed or optimized).
The autoconvolution is the inverse FFT of |X|^2. To minimize the peak of the autoconvolution,
we want the energy to be spread out as evenly as possible in the time domain (for the AC part).
This is equivalent to minimizing the peak of the inverse FFT of a power spectrum.
We optimize the phases using a subgradient method on the soft-max of the autoconvolution peaks,
combined with a scalar search for the DC component (integral) to balance the normalization.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _metric(x):
    """max |autoconv| / (integral)^2. Lower is better."""
    s = float(np.sum(x))
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _soft_max_abs(conv, beta=100.0):
    """Smooth approximation of max(abs(conv)) using log-sum-exp."""
    # We want to minimize the peak. 
    # Using log-sum-exp on squares approximates the L2 norm, not max.
    # A better approximation for max |x| is via log-sum-exp of x^2 with large beta?
    # Actually, max |x| = limit as beta->inf of (1/beta) log(sum exp(beta |x|)).
    # This is numerically unstable. 
    # Alternative: use the fact that for a signal with fixed power, minimizing max|conv| 
    # is related to minimizing the peak-to-average power ratio.
    # Let's stick to a subgradient approach on the actual max, which is non-smooth but 
    # we can use the subgradient (sign of the peak element in the gradient).
    # However, for a continuous optimization, a soft objective is better.
    # Let's use a quadratic penalty on the peaks.
    # Objective: sum( (max(0, |conv| - T))^2 ) for a threshold T?
    # Or simply minimize the L4 norm of conv? 
    # min ||conv||_4^4 / sum(x)^4. This is smooth and differentiable.
    # Let's try minimizing the 4-norm of the autoconvolution.
    pass

def _metric_l4(x):
    """L4 norm of autoconvolution / (integral)^2. Lower is better."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    conv = conv[:2 * N - 1]
    s = float(np.sum(x))
    if abs(s) < 1e-12:
        return float('inf')
    return float(np.sum(conv**4)) / (s**4)

def _gradient_l4(x):
    """Gradient of _metric_l4."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    conv = conv[:2 * N - 1]
    s = float(np.sum(x))
    
    # d/dx (sum(conv^4) / s^4)
    # Let J = sum(conv^4), S = s^4
    # dJ/dx = 4 * sum(conv^3 * dconv/dx)
    # dconv/dx is the autoconvolution kernel.
    # This is getting complicated to compute analytically efficiently.
    # Let's use a finite difference or a simpler subgradient approach.
    # Or use L-BFGS-B with numerical gradients.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(42)
    
    # Strategy: 
    # 1. Generate random phase spectra.
    # 2. For each phase spectrum, optimize the DC component (and possibly a few low-freq magnitudes) 
    #    to minimize the peak.
    # 3. Use L-BFGS-B to refine the time-domain signal starting from the phase-optimized signal.
    
    # We will try to minimize the L4 norm of the autoconvolution, which is a smooth surrogate 
    # for the max absolute value.
    
    def objective_l4(x):
        return _metric_l4(x)
    
    # Initial candidates
    # 1. Random phases with flat magnitude
    # 2. Random phases with random magnitude
    
    num_restarts = 20
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Generate a random phase spectrum
        # We want the signal to be real, so we use irfft.
        # The DC component (index 0) is real.
        # The other components need Hermitian symmetry if we were doing full FFT, 
        # but irfft handles this if we just provide the positive frequencies.
        
        # Let's generate random phases for the positive frequencies.
        phases = rng.uniform(0, 2 * np.pi, _FFT_SIZE // 2 + 1)
        
        # Magnitudes: start with flat (white noise spectrum)
        mags = np.ones(_FFT_SIZE // 2 + 1)
        
        # DC component: we will optimize this. Start with 1.
        # x = irfft(mags * exp(i * phases))
        # Note: irfft expects the DC component to be real. 
        # So phases[0] should be 0 or pi.
        phases[0] = 0.0
        
        x_init = np.fft.irfft(mags * np.exp(1j * phases), _FFT_SIZE)
        x_init = x_init[:N]
        
        # Normalize to have a reasonable sum
        s = np.sum(x_init)
        if abs(s) < 1e-6:
            # If sum is close to 0, the metric is huge. 
            # We want a non-zero sum. 
            # We can add a constant DC offset.
            # Let's set the DC component explicitly.
            # The DC component of x is related to x[0] of the FFT? 
            # No, sum(x) = N * X[0] for DFT.
            # For irfft, the DC bin is X[0].
            # sum(x) = X[0] * N? No.
            # Parseval: sum |x|^2 = sum |X|^2.
            # sum(x) is the DC component of the *autocorrelation*? No.
            # sum(x) = integral of x.
            # Let's just optimize the DC offset.
            
            # We can parameterize x = x_ac + c, where x_ac has zero sum.
            # Then sum(x) = N*c.
            # Autoconv(x) = Autoconv(x_ac) + 2*c*sum(x_ac)*delta + c^2*N*delta?
            # Actually, autoconv(x+c) = autoconv(x) + 2*c*sum(x)*delta + c^2*N*delta?
            # Let's just use L-BFGS-B to optimize the whole vector x, 
            # but start from a good spectral initialization.
            
            pass
        
        # Use L-BFGS-B to minimize the L4 metric
        # L-BFGS-B uses numerical gradients by default if not provided.
        # For N=400, this might be slow. 
        # Let's try a few iterations.
        
        try:
            res = minimize(objective_l4, x_init, method='L-BFGS-B', 
                           options={'maxiter': 100, 'gtol': 1e-5})
            
            if res.fun < best_val:
                # Check the actual metric (max abs) to see if it's better
                actual_val = _metric(res.x)
                if actual_val < best_val:
                    best_val = actual_val
                    best_x = res.x.copy()
        except Exception:
            pass
        
        # Also try a quick Nelder-Mead on the L4 objective
        try:
            res_nm = minimize(objective_l4, x_init, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            if res_nm.fun < best_val:
                actual_val = _metric(res_nm.x)
                if actual_val < best_val:
                    best_val = actual_val
                    best_x = res_nm.x.copy()
        except Exception:
            pass

    # Final polish with Nelder-Mead on the actual max-abs metric
    remaining = deadline - time.time()
    if remaining > 10:
        try:
            res = minimize(_metric, best_x, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
