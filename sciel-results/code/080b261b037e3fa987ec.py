import numpy as np
import time
from scipy.optimize import linprog
from scipy.linalg import toeplitz

N = 400

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    # The objective is value = abs(2 * N * max(z) / sum(h)^2).
    # To minimize this, we can fix the sum of h (normalization) and minimize max(z).
    # Let sum(h) = 1. Then value = 2 * N * max(z).
    # We want to minimize max(z) subject to sum(h) = 1.
    # However, z = h * h (full convolution).
    # z[k] = sum_i h[i] h[i+k-1] (indices adjusted for 0-based).
    # The maximum is likely at k=0 (center) or near it.
    # If we constrain all z[k] <= t, and sum(h) = 1, we can minimize t.
    # This is a Linear Programming problem if we treat z[k] as variables?
    # No, z[k] are quadratic in h.
    # However, if we fix h, z is fixed.
    # Wait, the problem is non-convex because of the product h[i]h[j].
    
    # Let's rethink.
    # The current GA approach is slow and stuck.
    # Inspiration 1 & 2 suggest low-dimensional parametric families.
    # Let's try a specific parametric family that is known to have low autocorrelation peaks:
    # Barker sequences or similar "flat" autocorrelation codes, adapted to real values.
    # Or, use a window function that minimizes the sidelobes of its Fourier transform?
    # No, autocorrelation in time domain relates to power spectrum.
    # By Wiener-Khinchin, the autocorrelation is the inverse Fourier transform of the power spectrum |H(f)|^2.
    # We want the time-domain autocorrelation peak to be small.
    # This means we want the power spectrum to be as flat as possible (white noise).
    # A signal with a flat power spectrum is one where the magnitude of its DFT is constant.
    # Such signals are called "flat spectra" or have minimal peak-to-sidelobe ratio.
    # Random Gaussian noise has a roughly flat spectrum, but the peak can still be high due to randomness.
    # Deterministic sequences with flat spectra exist (e.g., Zadoff-Chu, though complex).
    # For real signals, we can use a window that approximates a flat spectrum.
    
    # Alternative Strategy: 
    # Use a "box" function with optimized width?
    # Or, use a sine wave?
    # Let's try a simple parametric family: h[i] = A * sin(2*pi*f*i + phi) + B.
    # We can optimize A, f, phi, B.
    # But the sum must be non-zero.
    
    # Let's look at the seed: np.ones(400) -> value 2.0.
    # The GA found 1.12.
    # Let's try a "bipolar" signal: +1 and -1.
    # If h is +/- 1, sum(h) is small, which might blow up the denominator?
    # No, we can scale h. The value is scale-invariant!
    # value(h) = value(c*h) for any c != 0.
    # So we can fix sum(h) = 1.
    
    # Let's try to construct a signal with a very flat power spectrum.
    # One way to get a flat power spectrum is to use a random phase in frequency domain, then inverse FFT.
    # But we need real values.
    # Let's generate a random vector in frequency domain, enforce Hermitian symmetry, IFFT, and take real part.
    # This gives a real signal with a flat power spectrum (in the ideal sense, if magnitudes are constant).
    # Let's try this:
    # 1. Generate random magnitudes (constant) and random phases.
    # 2. IFFT.
    # 3. Take real part.
    # 4. Subtract mean to make it zero-mean? No, sum(h) must not be zero.
    #    Actually, if we subtract the mean, the DC component (lag 0 of autocorrelation) changes.
    #    Wait, the autocorrelation z[0] = sum(h^2).
    #    If we remove the mean, sum(h^2) decreases?
    #    Let h = h_mean + h_fluct.
    #    sum(h^2) = N*h_mean^2 + sum(h_fluct^2).
    #    The peak of autocorrelation is usually at lag 0.
    #    If we make h zero-mean, z[0] = sum(h^2).
    #    But the denominator is (sum(h))^2. If sum(h) is small, the value is large.
    #    So we want a large sum(h) relative to the autocorrelation peak.
    #    This suggests h should be mostly positive, with small negative fluctuations.
    #    This is consistent with the GA's "center bump, negative sides" strategy.
    
    # Let's try a specific analytic form:
    # h[i] = 1 + a * cos(2*pi*k*i/N)
    # This is a "chirp" or modulated signal.
    # Let's optimize a and k.
    
    def score(h):
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))
    
    best_h = np.ones(N)
    best_score = score(best_h)
    
    # Parametric family: h[i] = 1 + a * cos(2*pi*k*i/N)
    # We can also add a linear ramp or other shapes.
    # Let's try a grid search over a and k.
    
    # a in [-0.9, 0.9] (to keep h mostly positive)
    # k in [1, N/2]
    
    # To speed up, we can use a coarse grid first, then refine.
    
    start_time = time.time()
    
    # Coarse grid
    a_vals = np.linspace(-0.9, 0.9, 100)
    k_vals = np.arange(1, N//2, 1)
    
    # We can vectorize the computation for a fixed k?
    # z depends on a and k.
    # It's easier to just loop, but 100 * 200 = 20000 evaluations.
    # Each evaluation is O(N^2) = 160,000 ops.
    # 20000 * 160000 = 3.2e9 ops. This might be too slow in Python.
    # We need to be smarter.
    
    # Use FFT to compute autocorrelation?
    # np.convolve is O(N^2).
    # FFT-based convolution is O(N log N).
    # But we are in a loop.
    
    # Let's reduce the grid.
    # a_vals = np.linspace(-0.5, 0.5, 20)
    # k_vals = np.arange(1, 50, 5)
    
    # Actually, let's try a different parametric family that is known to be good.
    # "Barker-like" sequences are binary.
    # Let's try a Gaussian window with optimized width.
    # h[i] = exp(-((i - N/2)^2) / (2*sigma^2))
    # This has a high peak at center.
    # We want to reduce the peak.
    
    # How about h[i] = 1 - a * exp(-((i - N/2)^2) / (2*sigma^2))?
    # This is a dip in the center.
    
    # Let's try a few specific shapes and optimize their parameters with a simple local search.
    
    shapes = [
        lambda i, p: 1 + p[0] * np.cos(2 * np.pi * p[1] * i / N),
        lambda i, p: 1 - p[0] * np.exp(-((i - N/2)**2) / (2 * p[1]**2)),
        lambda i, p: 1 + p[0] * np.sin(2 * np.pi * p[1] * i / N)
    ]
    
    i = np.arange(N)
    
    # For each shape, do a local search.
    for shape_func in shapes:
        # Initial guess
        if "cos" in str(shape_func):
            p0 = [0.5, 5.0]
        elif "exp" in str(shape_func):
            p0 = [0.5, 50.0]
        else:
            p0 = [0.5, 5.0]
            
        # Simple hill climbing
        p = p0[:]
        h = shape_func(i, p)
        s = score(h)
        
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
        # Local search
        step_size = 0.1
        for _ in range(1000):
            if time.time() > deadline:
                break
            improved = False
            for j in range(len(p)):
                for delta in [-step_size, step_size]:
                    p_new = p[:]
                    p_new[j] += delta
                    # Ensure parameters are in valid range
                    if p_new[j] < 0.01 and j > 0: # sigma or freq must be positive
                        continue
                    h_new = shape_func(i, p_new)
                    s_new = score(h_new)
                    if s_new < s:
                        p = p_new
                        s = s_new
                        h = h_new
                        improved = True
                        if s < best_score:
                            best_score = s
                            best_h = h.copy()
            if not improved:
                step_size *= 0.5
                if step_size < 1e-4:
                    break
                    
    return best_h
