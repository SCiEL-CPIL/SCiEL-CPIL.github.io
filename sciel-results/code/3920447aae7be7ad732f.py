"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600

# EVOLVE-BLOCK-START

def _compute_ratio_fast(h):
    """Fast computation of normalized max autoconvolution ratio."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    padded = np.zeros(2 * n, dtype=np.float64)
    padded[:n] = h
    F = np.fft.rfft(padded)
    conv = np.fft.irfft(F * np.conj(F), n=2*n).real
    max_conv = np.max(conv[:2*n-1])
    total = np.sum(h)
    if total == 0:
        return 0.0
    dx = 1.0 / (2 * n)
    return max_conv / (total * total * dx)

def _local_refine(h0, deadline):
    """Local refinement using coordinate descent and L-BFGS-B."""
    best = h0.copy()
    best_score = _compute_ratio_fast(best)
    
    # Coordinate descent: iterate over blocks of indices
    block_size = 10
    n_blocks = N // block_size
    rng = np.random.RandomState(42)
    
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        improved = False
        for b in range(n_blocks):
            if time.time() > deadline - 5:
                break
            idx = slice(b * block_size, (b + 1) * block_size)
            # Try perturbing this block
            for _ in range(5):
                h_trial = best.copy()
                h_trial[idx] += rng.normal(0, 0.1, block_size)
                h_trial = np.maximum(h_trial, 0.0)
                if np.sum(h_trial) == 0:
                    continue
                score = _compute_ratio_fast(h_trial)
                if score < best_score:
                    best = h_trial
                    best_score = score
                    improved = True
        if not improved:
            break
    
    # L-BFGS-B polish
    if time.time() < deadline - 3:
        try:
            def _obj(h):
                h_pos = np.maximum(h, 0.0)
                if np.sum(h_pos) == 0:
                    return 1e6
                return _compute_ratio_fast(h_pos)
            
            result = minimize(
                _obj, best,
                method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-4}
            )
            if result.fun < best_score:
                best = np.maximum(result.x, 0.0)
                best_score = _compute_ratio_fast(best)
        except Exception:
            pass
    
    return best, best_score

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Baseline: uniform
    best_heights = np.ones(N, dtype=np.float64)
    best_score = _compute_ratio_fast(best_heights)
    
    rng = np.random.RandomState(123)
    
    # Random restarts with local refinement
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Generate random nonnegative heights
        # Mix of different distributions
        if i % 4 == 0:
            # Uniform random
            h = rng.random(N)
        elif i % 4 == 1:
            # Exponential (sparse)
            h = rng.exponential(1.0, N)
        elif i % 4 == 2:
            # Sparse: mostly zero with some spikes
            h = np.zeros(N)
            n_spikes = rng.randint(10, 100)
            spike_idx = rng.choice(N, n_spikes, replace=False)
            h[spike_idx] = rng.random(n_spikes) + 0.1
        else:
            # Low-frequency random (smooth)
            freqs = np.arange(N)
            coeffs = rng.normal(0, 1, N)
            coeffs[1:] *= 1.0 / np.sqrt(1 + freqs[1:]**2)  # low-pass
            h = np.fft.irfft(coeffs * 2 * N, n=N)
            h = np.abs(h)
        
        h = np.maximum(h, 0.0)
        if np.sum(h) == 0:
            continue
        
        # Local refinement
        h_refined, score = _local_refine(h, deadline)
        if score < best_score:
            best_score = score
            best_heights = h_refined
    
    # Try structured candidates
    t = np.arange(N, dtype=np.float64)
    
    # Raised cosine
    h_cos = 0.5 * (1.0 + np.cos(np.pi * t / (N - 1)))
    h_cos = np.maximum(h_cos, 0.0)
    if np.sum(h_cos) > 0:
        score = _compute_ratio_fast(h_cos)
        if score < best_score:
            best_score = score
            best_heights = h_cos.copy()
    
    # Various Gaussians
    center = N / 2.0
    for sigma_frac in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        if time.time() > deadline - 5:
            break
        sigma = sigma_frac * N
        h_gauss = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        h_gauss = np.maximum(h_gauss, 0.0)
        if np.sum(h_gauss) > 0:
            score = _compute_ratio_fast(h_gauss)
            if score < best_score:
                best_score = score
                best_heights = h_gauss.copy()
    
    # Final polish
    if time.time() < deadline - 3:
        best_heights, best_score = _local_refine(best_heights, deadline)
    
    return best_heights.tolist()

# EVOLVE-BLOCK-END
