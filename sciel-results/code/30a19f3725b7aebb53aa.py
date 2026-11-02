"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _overlap_at_shifts(h, shifts):
    """Compute overlap at given shifts for h with 1-h."""
    N = len(h)
    one_minus_h = 1.0 - h
    overlaps = np.zeros(len(shifts))
    for idx, k in enumerate(shifts):
        shifted = np.roll(one_minus_h, -k)
        overlaps[idx] = np.dot(h, shifted)
    return overlaps

def _max_overlap(h):
    """Compute maximum overlap over all shifts."""
    N = len(h)
    one_minus_h = 1.0 - h
    # Use FFT for efficient computation of all shifts
    # overlap(k) = sum_i h[i] * (1-h[i+k]) = sum_i h[i] - sum_i h[i]*h[i+k]
    # The autocorrelation of h gives sum_i h[i]*h[i+k]
    H = np.fft.rfft(h, n=2*N)
    autocorr = np.fft.irfft(H * np.conj(H), n=2*N)[:N]
    # autocorr[k] = sum_i h[i]*h[i+k] (circular)
    overlaps = N * 0.5 - autocorr  # since sum h = N/2
    return np.max(overlaps)

def _soft_max_overlap(h, beta=50.0):
    """Smooth approximation of max overlap for gradient-based optimization."""
    N = len(h)
    one_minus_h = 1.0 - h
    H = np.fft.rfft(h, n=2*N)
    autocorr = np.fft.irfft(H * np.conj(H), n=2*N)[:N]
    overlaps = N * 0.5 - autocorr
    # Soft max using log-sum-exp
    m = np.max(overlaps)
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initial guess: use a structured pattern
    # Try alternating pattern perturbed
    h = np.zeros(N)
    h[0::2] = 0.9
    h[1::2] = 0.1
    # Normalize to sum = N/2
    h = h - np.mean(h) + 0.5
    
    best_h = h.copy()
    best_score = _max_overlap(best_h)
    
    # Try multiple random initializations and local optimization
    np.random.seed(42)
    n_attempts = 20
    
    for attempt in range(n_attempts):
        if time.time() > deadline - 10.0:
            break
        
        if attempt == 0:
            h_current = best_h.copy()
        else:
            # Random perturbation of a structured pattern
            if attempt % 3 == 1:
                # Near-alternating
                h_current = np.zeros(N)
                h_current[0::2] = 0.8 + 0.2 * np.random.rand(N//2)
                h_current[1::2] = 0.2 - 0.2 * np.random.rand(N//2)
            elif attempt % 3 == 2:
                # Smooth random
                h_current = 0.5 + 0.3 * np.random.randn(N)
            else:
                # Perturb best
                h_current = best_h + 0.1 * np.random.randn(N)
            
            # Project to [0,1] and normalize
            h_current = np.clip(h_current, 0, 1)
            h_current = h_current - np.mean(h_current) + 0.5
            h_current = np.clip(h_current, 0, 1)
            # Re-normalize sum to N/2
            if np.sum(h_current) > 0:
                h_current = h_current * (N / 2.0) / np.sum(h_current)
                h_current = np.clip(h_current, 0, 1)
                # Final adjustment to ensure sum is exactly N/2
                diff = N / 2.0 - np.sum(h_current)
                h_current += diff / N
        
        # Local optimization using Nelder-Mead on a subset of parameters
        # Or use scipy minimize with soft max
        def objective(x):
            x = np.clip(x, 0, 1)
            x = x - np.mean(x) + 0.5
            x = np.clip(x, 0, 1)
            return _soft_max_overlap(x, beta=30.0)
        
        def gradient(x):
            # Numerical gradient
            eps = 1e-6
            grad = np.zeros_like(x)
            for i in range(N):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
            return grad
        
        try:
            result = minimize(
                objective, 
                h_current, 
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            h_opt = np.clip(result.x, 0, 1)
            h_opt = h_opt - np.mean(h_opt) + 0.5
            h_opt = np.clip(h_opt, 0, 1)
            # Ensure sum is exactly N/2
            diff = N / 2.0 - np.sum(h_opt)
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
            # Final correction
            diff = N / 2.0 - np.sum(h_opt)
            if abs(diff) > 1e-10:
                h_opt += diff / N
            
            score = _max_overlap(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception:
            pass
        
        # Also try Nelder-Mead for a few steps
        if time.time() < deadline - 5.0:
            try:
                result2 = minimize(
                    lambda x: _soft_max_overlap(np.clip(x, 0, 1), beta=20.0),
                    best_h,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                h_nm = np.clip(result2.x, 0, 1)
                h_nm = h_nm - np.mean(h_nm) + 0.5
                h_nm = np.clip(h_nm, 0, 1)
                diff = N / 2.0 - np.sum(h_nm)
                h_nm += diff / N
                h_nm = np.clip(h_nm, 0, 1)
                diff = N / 2.0 - np.sum(h_nm)
                if abs(diff) > 1e-10:
                    h_nm += diff / N
                    
                score_nm = _max_overlap(h_nm)
                if score_nm < best_score:
                    best_score = score_nm
                    best_h = h_nm.copy()
            except Exception:
                pass
    
    # Final cleanup to ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    diff = N / 2.0 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-10:
        best_h += diff / N
    
    return best_h
# EVOLVE-BLOCK-END
