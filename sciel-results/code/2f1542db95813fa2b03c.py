"""Rewritten program: spectral parameterization with coordinate descent on DFT coefficients."""
import numpy as np
import time

N = 600

def _autoconv_peak(x):
    """Compute the maximum of the autoconvolution of x (normalized).
    
    The autoconvolution (circular) can be computed via FFT.
    For a step function on [-1/4, 1/4] with N intervals, the autoconvolution
    in the continuous sense relates to the circular convolution of the discrete
    heights. We want to minimize the max of the autoconvolution.
    
    The normalized autoconvolution: (f * f)(t) = integral f(t-s) f(s) ds.
    With step heights h_i on intervals of width dx = 1/2 / N, 
    (f*f)(t) ≈ dx * sum_i h_i h_{i+k} where k is the shift.
    The max over t of this is dx * max_k (circular autocorrelation at lag k).
    
    We minimize this max.
    """
    # Use FFT to compute circular autocorrelation
    X = np.fft.rfft(x)
    power = X * np.conj(X)
    acorr = np.fft.irfft(power, n=N)
    # The autoconvolution values are dx * acorr[k] for each lag k
    dx = (1.0/2.0) / N
    peak = dx * np.max(acorr)
    return peak

def _autoconv_values(x):
    """Return all autoconvolution values (for subgradient)."""
    X = np.fft.rfft(x)
    power = X * np.conj(X)
    acorr = np.fft.irfft(power, n=N)
    dx = (1.0/2.0) / N
    return dx * acorr

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    np.random.seed(42)
    
    # Start with a smooth signal in frequency domain
    # Initialize with low-frequency dominant signal
    n_freq = N // 2 + 1
    freqs = np.arange(n_freq)
    
    # Start with a Gaussian-like spectrum (smooth in time)
    sigma_f = 5.0
    init_spec = np.exp(-freqs**2 / (2 * sigma_f**2))
    # Make it real and symmetric for real time-domain signal
    init_spec[0] = 1.0
    
    # Convert to time domain
    x = np.fft.irfft(init_spec, n=N)
    # Ensure nonnegativity
    x = np.maximum(x, 0.0)
    # Normalize
    if np.sum(x) > 0:
        x = x / np.sum(x) * N  # Keep reasonable scale
    
    best_x = x.copy()
    best_peak = _autoconv_peak(best_x)
    
    # Coordinate descent on frequency coefficients
    # We work in the frequency domain: adjust each DFT coefficient
    # to reduce the peak autoconvolution while keeping x >= 0
    
    lr = 0.01
    soft_k = 10.0  # For soft-max approximation
    
    for it in range(50000):
        if time.time() > deadline - 10.0:
            break
        
        # Get current autoconvolution
        acv = _autoconv_values(x)
        peak = np.max(acv)
        
        if peak < best_peak:
            best_peak = peak
            best_x = x.copy()
        
        # Subgradient approach: the gradient of the peak w.r.t. x
        # is the x shifted to the lag where peak occurs
        peak_lag = np.argmax(acv)
        
        # Gradient of peak autoconv w.r.t. x:
        # d/dx_k [dx * sum_i x_i x_{i+lag}] = dx * (x_{k-lag} + x_{k+lag}) for k in range
        # More precisely, for circular:
        grad = np.zeros(N)
        for k in range(N):
            idx1 = (k - peak_lag) % N
            idx2 = (k + peak_lag) % N
            grad[k] = (x[idx1] + x[idx2]) / 2.0
        
        dx_step = (1.0/2.0) / N
        grad *= dx_step
        
        # Update in time domain
        x_new = x - lr * grad
        
        # Project to nonnegative
        x_new = np.maximum(x_new, 0.0)
        
        # Check if update helps
        new_peak = _autoconv_peak(x_new)
        if new_peak < peak:
            x = x_new
        else:
            # Try smaller step
            lr = max(lr * 0.9, 1e-6)
        
        # Also try adjusting in frequency domain for better exploration
        if it % 100 == 0:
            # Get current spectrum
            X = np.fft.rfft(x)
            # Perturb low frequencies
            for fi in range(1, min(20, n_freq)):
                if time.time() > deadline - 10.0:
                    break
                # Try increasing and decreasing this frequency component
                for delta in [0.01, -0.01]:
                    X_test = X.copy()
                    X_test[fi] += delta
                    X_test[-fi] += delta  # For real signal symmetry
                    x_test = np.fft.irfft(X_test, n=N)
                    x_test = np.maximum(x_test, 0.0)
                    p_test = _autoconv_peak(x_test)
                    if p_test < peak:
                        x = x_test
                        peak = p_test
                        break
    
    # Final: return the best found
    return list(best_x)
