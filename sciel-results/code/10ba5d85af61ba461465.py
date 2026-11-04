"""
Strategy: Nelder-Mead optimization with random restarts and frequency-domain initialization.
Instead of heuristic patterns, we treat the step heights as a continuous vector to be minimized
using a derivative-free method. We use multiple restarts from different random initializations,
including some derived from random Fourier phases to ensure diverse spectral content.
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


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(42)
    
    # We will perform multiple restarts of Nelder-Mead
    # Each restart gets a portion of the available time
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    # Initial candidates
    # 1. All ones (baseline)
    # 2. Random +/- 1
    # 3. Random Fourier phases (creates smooth but varied functions)
    
    candidates = []
    
    # Candidate 1: Random +/- 1
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Candidate 2: Random Fourier
    # Generate random phases for non-DC frequencies
    k = np.arange(_FFT_SIZE // 2 + 1)
    # Set DC component to 0 to ensure sum is close to 0? 
    # Actually, we want non-zero sum. 
    # Let's just generate random real values in frequency domain and inverse FFT.
    # To ensure real output, we need Hermitian symmetry or use irfft.
    # irfft takes N/2+1 bins.
    freq_bins = rng.standard_normal(_FFT_SIZE // 2 + 1)
    # Ensure the DC component (index 0) is non-zero to have a non-zero sum
    if abs(freq_bins[0]) < 1e-6:
        freq_bins[0] = 1.0
    x_fourier = np.fft.irfft(freq_bins, _FFT_SIZE)
    x_fourier = x_fourier[:N]
    # Normalize to have similar magnitude
    rms = np.sqrt(np.mean(x_fourier**2))
    if rms > 1e-12:
        x_fourier = x_fourier / rms
    candidates.append(x_fourier)
    
    # Candidate 3: Low-pass filtered random noise
    x_noise = rng.standard_normal(N)
    # Simple box filter
    kernel_size = 10
    kernel = np.ones(kernel_size) / kernel_size
    x_lowpass = np.convolve(x_noise, kernel, mode='same')
    rms = np.sqrt(np.mean(x_lowpass**2))
    if rms > 1e-12:
        x_lowpass = x_lowpass / rms
    candidates.append(x_lowpass)
    
    # Add the all-ones vector as a candidate
    candidates.append(np.ones(N))
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Select a starting point
        if i < len(candidates):
            x0 = candidates[i].copy()
        else:
            # Perturb the best found so far
            x0 = best_x + rng.normal(0, 0.1, N)
            
        # Run Nelder-Mead
        # We need to be careful with the time limit for a single optimization run
        # Nelder-Mead can take many iterations. We'll rely on the time check in the objective?
        # No, scipy's minimize doesn't easily support time-based stopping without wrapping.
        # We can limit maxiter based on time budget.
        # A typical Nelder-Mead step is fast, but let's estimate.
        # Let's set a maxiter that is likely to finish within the allocated time.
        # If we have ~10s per restart, and each eval is ~1ms (FFT of 512 is fast), 
        # we can afford ~5000-10000 evals. Nelder-Mead for N=400 might need more, 
        # but we are doing restarts.
        
        # To be safe, we'll use a wrapper that checks time?
        # Actually, it's better to just let it run but catch exceptions or timeouts?
        # The instruction says "keep an explicit deadline guard".
        # If minimize runs too long, it will be killed by the evaluator.
        # We should probably reduce N or use a faster method if N=400 is too slow for NM.
        # However, FFT is O(N log N). 400 is small.
        # Let's try maxiter=2000 for each restart.
        
        def objective(x):
            return _metric(x)
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline - 10:
            break
            
        # If we have time, do a local polish on the best found so far
        # using a smaller step size or different method?
        # Let's just continue with restarts.
        
    # Final polish with a very tight tolerance if time permits
    remaining = deadline - time.time()
    if remaining > 5:
        try:
            res = minimize(_metric, best_x, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
