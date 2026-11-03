"""Optimize a nonnegative step function on [-1/4, 1/4] to minimize the peak of its autoconvolution."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 600
H = 1.0 / (2 * N)

def _fft_autoconvolve(x):
    """Compute the autoconvolution of x using FFT."""
    n = len(x)
    # Zero-pad to avoid circular convolution
    f = np.fft.rfft(x, n=2 * n)
    conv = np.fft.irfft(f * f, n=2 * n)
    return conv

def _score(x):
    """Compute the upper bound on C1 from the step function x."""
    conv = _fft_autoconvolve(x)
    peak = np.max(conv) * H
    return peak

def _smooth_max(values, sigma=0.01):
    """Log-sum-exp based smooth maximum."""
    m = np.max(values)
    return m + sigma * np.log(np.sum(np.exp((values - m) / sigma)))

def _objective(weights, basis, target_sum):
    """Objective function for optimization."""
    x_raw = weights @ basis
    sum_x = np.sum(x_raw)
    if sum_x < 1e-12:
        return 1e9
    x = x_raw * (target_sum / sum_x)
    x = np.maximum(x, 0)
    conv = _fft_autoconvolve(x)
    peak = np.max(conv) * H
    return peak

def _smooth_objective(weights, basis, target_sum, sigma=0.005):
    """Smooth objective function for optimization."""
    x_raw = weights @ basis
    sum_x = np.sum(x_raw)
    if sum_x < 1e-12:
        return 1e9
    x = x_raw * (target_sum / sum_x)
    x = np.maximum(x, 0)
    conv = _fft_autoconvolve(x)
    conv_scaled = conv * H
    m = np.max(conv_scaled)
    return m + sigma * np.log(np.sum(np.exp((conv_scaled - m) / sigma)))

def construct():
    """Construct a nonnegative step function with 600 intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    
    Returns: list of 600 nonnegative step heights.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Create basis vectors: Gaussian functions with different centers and widths
    t = np.linspace(-1/4, 1/4, N, endpoint=False)
    M = 30  # Number of basis functions
    centers = np.linspace(-1/4, 1/4, M)
    widths = np.linspace(0.02, 0.1, M)
    
    basis = np.zeros((M, N))
    for k in range(M):
        basis[k] = np.exp(-0.5 * ((t - centers[k]) / widths[k]) ** 2)
        basis[k] = np.maximum(basis[k], 0)
        # Normalize each basis vector to have unit sum
        s = np.sum(basis[k])
        if s > 0:
            basis[k] /= s
    
    target_sum = N / 2.0  # Target sum for normalization
    
    # Initial guess: uniform weights
    w0 = np.ones(M) / M
    bounds = [(0, 10) for _ in range(M)]
    
    # First, try a few simple shapes to get a good starting point
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Raised cosine
    x_rc = 0.5 * (1 + np.cos(2 * np.pi * t))
    x_rc = np.maximum(x_rc, 0)
    norm_rc = np.sum(x_rc)
    if norm_rc > 0:
        x_rc = x_rc * (target_sum / norm_rc)
    score_rc = _score(x_rc)
    if score_rc < best_score:
        best_score = score_rc
        best_x = x_rc.copy()
    
    # Triangle window
    x_tri = 1.0 - 2 * np.abs(t)
    x_tri = np.maximum(x_tri, 0)
    norm_tri = np.sum(x_tri)
    if norm_tri > 0:
        x_tri = x_tri * (target_sum / norm_tri)
    score_tri = _score(x_tri)
    if score_tri < best_score:
        best_score = score_tri
        best_x = x_tri.copy()
    
    # Use the best simple shape to initialize the weights
    # Find the best combination of basis functions that approximates best_x
    # For simplicity, just use uniform weights as initial guess
    w_best = w0.copy()
    
    # Run Differential Evolution
    try:
        if time.time() < deadline - 30.0:
            result_de = differential_evolution(
                _objective, bounds, args=(basis, target_sum),
                maxiter=50, popsize=15, seed=42, tol=0.01, polish=False
            )
            w_best = result_de.x
            score_de = _objective(w_best, basis, target_sum)
            if score_de < best_score:
                best_score = score_de
                x_raw = w_best @ basis
                sum_x = np.sum(x_raw)
                if sum_x > 0:
                    best_x = x_raw * (target_sum / sum_x)
                    best_x = np.maximum(best_x, 0)
    except Exception as e:
        pass
    
    # Polish with L-BFGS-B using smooth objective
    try:
        if time.time() < deadline - 15.0:
            result_min = minimize(
                _smooth_objective, w_best, args=(basis, target_sum, 0.005),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            w_polish = result_min.x
            score_polish = _objective(w_polish, basis, target_sum)
            if score_polish < best_score:
                best_score = score_polish
                x_raw = w_polish @ basis
                sum_x = np.sum(x_raw)
                if sum_x > 0:
                    best_x = x_raw * (target_sum / sum_x)
                    best_x = np.maximum(best_x, 0)
    except Exception as e:
        pass
    
    # Final high-precision polish: coordinate descent on the full x vector
    try:
        if time.time() < deadline - 5.0:
            x = best_x.copy()
            lr = 0.001
            for it in range(500):
                if time.time() > deadline - 2.0:
                    break
                
                conv = _fft_autoconvolve(x)
                conv_scaled = conv * H
                peak_idx = np.argmax(conv_scaled)
                
                # Subgradient update
                k = peak_idx
                grad = np.zeros(N)
                for i in range(N):
                    j = k - i
                    if 0 <= j < N:
                        grad[i] = x[j]
                
                grad = H * grad
                x_new = x - lr * grad
                x_new = np.maximum(x_new, 0)
                
                # Renormalize
                integral = np.sum(x_new)
                if integral > 0:
                    x_new = x_new * (target_sum / integral)
                
                score_new = _score(x_new)
                if score_new < best_score:
                    best_score = score_new
                    best_x = x_new.copy()
                    x = x_new
                    lr *= 0.999
                else:
                    lr *= 0.9
                    x = best_x.copy()
    except Exception as e:
        pass
    
    return [float(v) for v in best_x]
