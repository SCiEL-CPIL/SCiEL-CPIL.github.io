"""
Erdos minimum overlap: minimize max overlap of h with shifted (1-h).

Strategy:
1. Use a structured initial guess (e.g., a smoothed step function or a sinusoidal wave)
2. Apply gradient-based optimization (L-BFGS-B) to minimize the maximum overlap
3. The objective is the maximum over all shifts of the overlap integral
4. Use a time budget to ensure we finish within 150s
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 100
TIME_LIMIT = 120  # seconds, with safety margin

def _compute_overlap_max(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    
    The overlap at shift k is sum_i h[i] * (1-h[(i+k) % N])
    We want to minimize the maximum over all shifts k.
    """
    N = len(h)
    # (1-h) reversed for cross-correlation
    one_minus_h = 1.0 - h
    
    # For each shift k, overlap_k = sum_i h[i] * (1-h[(i+k) % N])
    # This is a circular cross-correlation
    
    # Using FFT for efficiency
    # overlap[k] = sum_i h[i] * one_minus_h[(i+k) % N]
    # = circular correlation of h and one_minus_h
    
    # Let's compute all shifts using FFT
    # h_fft = fft(h), o_fft = fft(one_minus_h)
    # correlation = ifft(h_fft * conj(o_fft))
    
    h_fft = np.fft.fft(h)
    o_fft = np.fft.fft(one_minus_h)
    corr = np.real(np.fft.ifft(h_fft * np.conj(o_fft)))
    
    # corr[k] = sum_i h[i] * one_minus_h[(i-k) % N]
    # We want sum_i h[i] * one_minus_h[(i+k) % N] = corr[-k] or corr[N-k]
    # Actually, let's just take the max over all shifts
    return np.max(corr)

def _objective(h):
    """Objective function: maximum overlap of h with shifted (1-h)."""
    return _compute_overlap_max(h)

def _project(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h[i] <= 1."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to ensure sum = N/2
    current_sum = np.sum(h)
    target_sum = N / 2.0
    if current_sum != target_sum:
        # Add uniform offset, then re-clip and iterate
        for _ in range(50):
            offset = (target_sum - current_sum) / N
            h_new = h + offset
            h_new = np.clip(h_new, 0.0, 1.0)
            if np.allclose(h_new, h):
                break
            h = h_new
            current_sum = np.sum(h)
    return h

def construct():
    start_time = time.time()
    
    # Initial guess: try a few structured initial points
    # 1. Uniform (baseline)
    # 2. Sinusoidal: 0.5 + 0.5*sin(2*pi*i/N)
    # 3. Step-like: first half 1, second half 0 (but sum must be N/2, so this works)
    
    best_h = np.full(N, 0.5)
    best_score = _objective(best_h)
    
    # Try sinusoidal initial guess
    i = np.arange(N)
    h_sin = 0.5 + 0.4 * np.sin(2 * np.pi * i / N)
    h_sin = _project(h_sin)
    score_sin = _objective(h_sin)
    if score_sin < best_score:
        best_h = h_sin.copy()
        best_score = score_sin
    
    # Try a smoother wave
    h_sin2 = 0.5 + 0.3 * np.sin(4 * np.pi * i / N)
    h_sin2 = _project(h_sin2)
    score_sin2 = _objective(h_sin2)
    if score_sin2 < best_score:
        best_h = h_sin2.copy()
        best_score = score_sin2
    
    # Run L-BFGS-B optimization with a few restarts
    elapsed = time.time() - start_time
    time_remaining = TIME_LIMIT - elapsed
    
    if time_remaining > 10:
        # Try gradient-based optimization
        try:
            res = minimize(
                _objective,
                best_h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_score:
                h_opt = _project(res.x)
                score_opt = _objective(h_opt)
                if score_opt < best_score:
                    best_h = h_opt
                    best_score = score_opt
        except Exception:
            pass
    
    # Check time and do a second round if we have time
    elapsed = time.time() - start_time
    time_remaining = TIME_LIMIT - elapsed
    
    if time_remaining > 30:
        # Try a different initial point
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project(h_rand)
        score_rand = _objective(h_rand)
        
        if score_rand < best_score:
            best_h = h_rand.copy()
            best_score = score_rand
        
        # Optimize from this point
        try:
            res = minimize(
                _objective,
                best_h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_score:
                h_opt = _project(res.x)
                score_opt = _objective(h_opt)
                if score_opt < best_score:
                    best_h = h_opt
                    best_score = score_opt
        except Exception:
            pass
    
    # Final projection to ensure feasibility
    best_h = _project(best_h)
    
    # Verify sum is exactly N/2
    # Due to floating point, do a final adjustment
    current_sum = np.sum(best_h)
    if not np.isclose(current_sum, N / 2.0, atol=1e-10):
        # Distribute the difference evenly
        diff = (N / 2.0 - current_sum) / N
        best_h = best_h + diff
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
