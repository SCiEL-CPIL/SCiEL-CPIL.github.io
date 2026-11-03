import numpy as np
from scipy.signal import fftconvolve
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    # Initialize with constant function
    f = np.ones(N)
    
    # Precompute window for convolution
    # f is defined on [-1/4, 1/4]. Length 0.5.
    # Convolution support is [-1/2, 1/2]. Length 1.0.
    # We need to align the convolution result to the original grid or just find the max.
    # Using fftconvolve gives full convolution.
    
    def get_score(f_vec):
        # Ensure non-negative
        f_vec = np.maximum(f_vec, 0)
        if np.sum(f_vec) == 0:
            return float('inf'), f_vec
        conv = fftconvolve(f_vec, f_vec)
        # The peak of the convolution of a non-negative function is at the center (index N-1 in full conv of length 2N-1)
        # Actually, fftconvolve output length is 2N-1.
        # Index N-1 corresponds to lag 0.
        max_val = np.max(conv)
        l1 = np.sum(f_vec)
        return max_val / (l1 * l1), f_vec

    current_score, f = get_score(f)
    best_score = current_score
    best_f = f.copy()
    
    # Simple gradient descent on the bins
    # Objective: minimize J = max(conv) / (sum(f))^2
    # Let M = max(conv). J = M / S^2.
    # dJ/df = (dM/df * S^2 - M * 2S * dS/df) / S^4
    # dS/df = 1.
    # dM/df: If max is at index k, dM/df_i = f_{k-i} (with appropriate shifts/boundaries).
    
    step_size = 0.1
    for i in range(500):
        if time.time() > deadline:
            break
            
        conv = fftconvolve(f, f)
        max_idx = np.argmax(conv)
        M = conv[max_idx]
        S = np.sum(f)
        
        # Gradient of M w.r.t f
        # conv[k] = sum f[j] f[k-j]
        # d conv[k] / d f[i] = f[k-i] (if valid)
        # We need to construct the gradient vector.
        grad_M = np.zeros(N)
        # For each i in 0..N-1, the contribution is f[max_idx - i] if 0 <= max_idx - i < N
        # Note: fftconvolve indices 0..2N-2.
        # f indices 0..N-1.
        # lag = max_idx.
        # term is f[j] * f[lag - j].
        # d/d f[i] -> f[lag - i] (if lag-i in range) + f[i] (if lag in range, from the other side? No, symmetric)
        # Actually, conv is symmetric.
        # Let's just compute the gradient numerically or analytically for the peak.
        
        # Analytical gradient for the peak at lag L:
        # g_L = sum_{j} f_j f_{L-j}
        # dg_L / df_i = f_{L-i} + f_{L-i} (if L-i is valid and i is valid)
        # Wait, f is the same in both slots.
        # If L-i is in [0, N-1], then df_i appears in term f_i f_{L-i} and f_{L-i} f_i.
        # So derivative is 2 * f_{L-i}.
        
        valid = (max_idx - np.arange(N) >= 0) & (max_idx - np.arange(N) < N)
        idx = np.arange(N)
        grad_M[valid] = 2.0 * f[max_idx - idx[valid]]
        
        # Gradient of J
        # J = M / S^2
        # dJ = (S^2 dM - M 2S dS) / S^4 = (dM - 2 (M/S) dS) / S^2
        # dS/df = 1
        grad_J = (grad_M - 2.0 * (M / S) * np.ones(N)) / (S * S)
        
        # Update
        f = f - step_size * grad_J
        f = np.maximum(f, 0)
        
        # Renormalize? No, the ratio is scale invariant?
        # J(cf) = c^2 M / (c S)^2 = M/S^2. Yes, scale invariant.
        # So we don't need to normalize.
        
        new_score, f = get_score(f)
        if new_score < best_score:
            best_score = new_score
            best_f = f.copy()
            step_size *= 1.05
        else:
            step_size *= 0.8
            f = best_f.copy() # Revert
            
    return best_f.tolist()
# EVOLVE-BLOCK-END
