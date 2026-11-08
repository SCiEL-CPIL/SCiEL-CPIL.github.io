"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import fft, ifft


def _overlap_value(h):
    """Compute the maximum circular overlap of h with shifted (1-h).
    
    For shift k, overlap(k) = sum_i h_i * (1 - h_{i+k mod N})
    = sum_i h_i - sum_i h_i * h_{i+k mod N}
    = N/2 - (h*h)_k  where (h*h)_k is circular cross-correlation at lag k.
    
    So max overlap = N/2 - min_k (h*h)_k.
    We want to minimize max overlap, i.e., minimize (N/2 - min_k (h*h)_k).
    Equivalently, maximize min_k (h*h)_k.
    """
    # Circular cross-correlation of h with itself = circular autocorrelation
    # Using FFT: corr = ifft(fft(h) * conj(fft(h)))
    H = fft(h)
    corr = np.real(ifft(H * np.conj(H)))
    min_corr = np.min(corr)
    return N / 2.0 - min_corr


def _soft_min(x, temperature=0.05):
    """Smooth approximation of min(x) using log-sum-exp."""
    min_val = np.min(x)
    return min_val + temperature * np.log(np.sum(np.exp(-(x - min_val) / temperature)))


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Initial guess: constant 0.5
    h0 = np.full(N, 0.5)
    
    # We want to maximize min_k autocorr_k subject to:
    # 0 <= h_i <= 1, sum(h_i) = N/2
    # Objective: maximize t where t <= autocorr_k for all k
    # This is a convex problem (maximize t subject to linear + quadratic constraints)
    # But let's use a smooth surrogate: maximize soft_min(autocorr, temp)
    
    # Use Fourier domain approach:
    # h_i in [0,1], sum h_i = N/2
    # autocorr_k = sum_i h_i h_{i+k}
    # 
    # Alternative: work in real domain with scipy.optimize
    # Variables: h (N values), t (scalar)
    # Maximize t
    # Subject to: autocorr_k >= t for all k
    #             0 <= h_i <= 1
    #             sum(h_i) = N/2
    
    best_h = h0.copy()
    best_score = _overlap_value(h0)
    
    def objective(x):
        h = x[:N]
        t = x[N]
        # Compute autocorrelation
        H = fft(h)
        corr = np.real(ifft(H * np.conj(H)))
        # Penalty for corr_k < t (we want corr_k >= t)
        violations = np.maximum(0, t - corr)
        # Soft constraint: penalize violations
        penalty = np.sum(violations ** 2)
        # Also ensure h in [0,1] with soft penalty
        h_below = np.maximum(0, -h)
        h_above = np.maximum(0, h - 1)
        bound_penalty = np.sum(h_below ** 2) + np.sum(h_above ** 2)
        # Sum constraint: sum(h) = N/2
        sum_penalty = (np.sum(h) - N / 2.0) ** 2
        
        # Maximize t, so minimize -t + penalties
        return -t + penalty + bound_penalty + sum_penalty
    
    # Start with h=0.5, t slightly below min autocorr
    h_init = np.full(N, 0.5)
    H_init = fft(h_init)
    corr_init = np.real(ifft(H_init * np.conj(H_init)))
    t_init = np.min(corr_init) - 0.01
    
    x0 = np.concatenate([h_init, [t_init]])
    
    # Try multiple starting points and temperatures
    starts = [
        ('constant', np.full(N, 0.5)),
        ('sine', 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N)),
        ('cosine', 0.5 + 0.3 * np.cos(2 * np.pi * np.arange(N) / N)),
        ('alternating', np.where(np.arange(N) % 2 == 0, 0.7, 0.3)),
        ('random1', np.clip(np.random.default_rng(42).uniform(0.2, 0.8, N), 0, 1)),
    ]
    
    # Normalize starts to have sum = N/2
    for name, h_start in starts:
        if time.time() > deadline - 10.0:
            break
        
        h_s = h_start.copy()
        h_s = np.clip(h_s, 0.01, 0.99)
        h_s = h_s / np.sum(h_s) * (N / 2.0)
        h_s = np.clip(h_s, 0.0, 1.0)
        # Re-normalize
        diff = N / 2.0 - np.sum(h_s)
        if np.abs(diff) > 1e-10:
            # Distribute difference evenly where possible
            for i in range(N):
                if np.abs(diff) < 1e-10:
                    break
                if diff > 0 and h_s[i] < 1.0:
                    adj = min(diff, 1.0 - h_s[i])
                    h_s[i] += adj
                    diff -= adj
                elif diff < 0 and h_s[i] > 0.0:
                    adj = max(diff, -h_s[i])
                    h_s[i] += adj
                    diff -= adj
        
        H_s = fft(h_s)
        corr_s = np.real(ifft(H_s * np.conj(H_s)))
        t_s = np.min(corr_s) - 0.01
        
        x0_s = np.concatenate([h_s, [t_s]])
        
        try:
            res = minimize(
                objective, x0_s, method='L-BFGS-B',
                bounds=[(0, 1)] * N + [(0, N / 2.0)],
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_res = np.clip(res.x[:N], 0, 1)
            # Enforce sum constraint
            h_res = h_res / np.sum(h_res) * (N / 2.0)
            h_res = np.clip(h_res, 0, 1)
            
            score_res = _overlap_value(h_res)
            if score_res < best_score:
                best_score = score_res
                best_h = h_res.copy()
                
        except Exception:
            pass
    
    # Polish with tighter tolerance if time remains
    if time.time() < deadline - 5.0 and len(best_h) == N:
        try:
            H_b = fft(best_h)
            corr_b = np.real(ifft(H_b * np.conj(H_b)))
            t_b = np.min(corr_b)
            x_polish = np.concatenate([best_h, [t_b]])
            
            res_polish = minimize(
                objective, x_polish, method='L-BFGS-B',
                bounds=[(0, 1)] * N + [(0, N / 2.0)],
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_polish = np.clip(res_polish.x[:N], 0, 1)
            h_polish = h_polish / np.sum(h_polish) * (N / 2.0)
            h_polish = np.clip(h_polish, 0, 1)
            
            score_polish = _overlap_value(h_polish)
            if score_polish < best_score:
                best_score = score_polish
                best_h = h_polish.copy()
                
        except Exception:
            pass
    
    # Final enforcement of constraints
    best_h = np.clip(best_h, 0, 1)
    best_h = best_h / np.sum(best_h) * (N / 2.0)
    best_h = np.clip(best_h, 0, 1)
    
    return best_h

# EVOLVE-BLOCK-END
