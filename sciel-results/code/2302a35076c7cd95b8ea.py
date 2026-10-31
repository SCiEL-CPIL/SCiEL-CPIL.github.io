"""
Second autocorrelation inequality: maximize ||f||_2^2 / (||f||_1 * ||f*f||_inf)
for nonnegative step heights on 50 equal intervals.

Strategy:
  - The ratio is scale-invariant, so we can fix ||f||_1 = 1 (or any positive constant).
  - To maximize the ratio, we want to minimize ||f*f||_inf (the peak of the autoconvolution)
    given a fixed L1 norm, while also maximizing ||f||_2^2 (spreading mass evenly).
  - A uniform distribution gives ratio = N (since ||f||_2^2 = 1/N, ||f||_1 = 1, ||f*f||_inf = 1/N).
    Actually let's compute: f = [1/N]*N, ||f||_2^2 = N*(1/N^2) = 1/N, ||f||_1 = 1,
    autoconvolution peak: (f*f)[k] = sum_i f[i]f[i+k] for k=0 is sum f[i]^2 = 1/N.
    So ratio = (1/N) / (1 * 1/N) = 1. Wait, that gives 1.
    
  Let me recompute. If f = [1]*N (all ones):
    ||f||_2^2 = N
    ||f||_1 = N
    autoconvolution: (f*f)[k] = number of i such that f[i]*f[i+k] = 1, which is N-|k| for |k|<N.
    Peak at k=0: N.
    Ratio = N / (N * N) = 1/N.
    
  If f = [1/N]*N:
    ||f||_2^2 = N * (1/N)^2 = 1/N
    ||f||_1 = 1
    autoconvolution peak: N * (1/N)^2 = 1/N
    Ratio = (1/N) / (1 * 1/N) = 1.
    
  So uniform gives ratio 1. The seed returns [1.0]*N which gives ratio 1/N? No wait,
  the score is value/seed. If seed is [1.0]*N and scores 1.0, then the actual value for [1.0]*N must be the baseline.
  
  Let me think about what gives ratio > 1. We want large ||f||_2^2 relative to ||f||_1 * ||f*f||_inf.
  
  If we concentrate mass: f = [1, 0, 0, ..., 0]:
    ||f||_2^2 = 1
    ||f||_1 = 1
    autoconvolution peak = 1
    Ratio = 1.
  
  If f = [1, 1, 0, ..., 0] (two adjacent ones):
    ||f||_2^2 = 2
    ||f||_1 = 2
    autoconvolution: [2, 1, 1, 0, ...] -> peak = 2
    Ratio = 2 / (2*2) = 0.5.
  
  If f = [1, 0, 1, 0, ...] alternating:
    ||f||_2^2 = 25
    ||f||_1 = 25
    autoconvolution peak: at k=0, 25; at k=2, 24; etc. Peak = 25.
    Ratio = 25 / (25*25) = 0.1.
    
  So uniform (all equal) gives ratio 1, which is better. The key insight: we want to MINIMIZE the peak of autoconvolution.
  
  For a nonnegative vector, the autoconvolution peak is at least ||f||_2^2 (by Cauchy-Schwarz or just the k=0 term).
  Actually (f*f)[0] = sum f[i]^2 = ||f||_2^2. And for other k, (f*f)[k] can be larger.
  
  So ratio = ||f||_2^2 / (||f||_1 * ||f*f||_inf) <= ||f||_2^2 / (||f||_1 * ||f||_2^2) = 1/||f||_1.
  
  To maximize, we want small ||f||_1 and small ||f*f||_inf / ||f||_2^2 ratio.
  
  Since it's scale invariant, fix ||f||_1 = 1. Then ratio = ||f||_2^2 / ||f*f||_inf.
  We want to maximize ||f||_2^2 (spread evenly) and minimize ||f*f||_inf (avoid peaks in convolution).
  
  For uniform: ||f||_2^2 = 1/N, ||f*f||_inf = 1/N, ratio = 1.
  
  Can we do better? If we use a "flat" spectrum signal... Actually, if f is such that its Fourier transform is flat, then the autoconvolution (which is inverse FT of |F|^2) would be a delta, giving peak = ||f||_2^2, so ratio = 1.
  
  To get ratio > 1, we'd need ||f*f||_inf < ||f||_2^2, but (f*f)[0] = ||f||_2^2, so ||f*f||_inf >= ||f||_2^2 always. Thus ratio <= 1/||f||_1 <= 1 (with ||f||_1 >= 1 if... no, ||f||_1 can be anything).
  
  Wait, I think I'm confusing myself. Let me just implement a local search that tries to improve on the uniform baseline.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_ratio(f):
    """Compute the ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf)"""
    f = np.asarray(f, dtype=float)
    if np.any(f < -1e-12):
        return -1e9
    f = np.maximum(f, 0.0)
    l1 = np.sum(f)
    if l1 < 1e-12:
        return -1e9
    l2_sq = np.sum(f**2)
    # Autoconvolution
    conv = np.convolve(f, f, mode='full')
    conv_max = np.max(conv)
    if conv_max < 1e-12:
        return -1e9
    return l2_sq / (l1 * conv_max)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_f = np.ones(N)
    best_ratio = _compute_ratio(best_f)
    
    # Try different initial patterns
    candidates = []
    # Uniform
    candidates.append(np.ones(N))
    # Alternating
    alt = np.ones(N)
    alt[1::2] = 0.0
    candidates.append(alt)
    # Two alternating
    alt2 = np.ones(N)
    alt2[2::4] = 0.0
    alt2[3::4] = 0.0
    candidates.append(alt2)
    
    for c in candidates:
        r = _compute_ratio(c)
        if r > best_ratio:
            best_ratio = r
            best_f = c.copy()
    
    # Local search: perturb and accept improvements
    rng = np.random.default_rng(42)
    current_f = best_f.copy()
    current_ratio = best_ratio
    
    # Normalize for optimization
    current_f = current_f / np.sum(current_f)
    
    # Use scipy minimize on negative ratio with L1 constraint
    def neg_ratio(x):
        return -_compute_ratio(x)
    
    # Try Nelder-Mead on the ratio
    from scipy.optimize import minimize
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    bounds = [(0, 1)] * N
    
    # Multiple restarts with different initial points
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = current_f.copy()
        else:
            # Random perturbation of best
            x0 = best_f.copy()
            x0 = x0 / np.sum(x0)
            # Add noise
            noise = rng.normal(0, 0.1, N)
            x0 = np.maximum(x0 + noise, 0)
            x0 = x0 / np.sum(x0)
        
        try:
            result = minimize(
                neg_ratio,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if result.success or result.fun < neg_ratio(x0):
                r = -result.fun
                if r > best_ratio:
                    best_ratio = r
                    best_f = result.x.copy()
                    current_f = result.x.copy()
                    current_ratio = r
        except Exception:
            pass
    
    # Simple local search with random swaps
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        i = rng.integers(0, N)
        j = rng.integers(0, N)
        if i == j:
            continue
        
        # Swap values
        new_f = current_f.copy()
        new_f[i], new_f[j] = new_f[j], new_f[i]
        
        r = _compute_ratio(new_f)
        if r > current_ratio:
            current_f = new_f
            current_ratio = r
            if r > best_ratio:
                best_ratio = r
                best_f = new_f.copy()
    
    # Return as list of floats
    return [float(x) for x in best_f]
