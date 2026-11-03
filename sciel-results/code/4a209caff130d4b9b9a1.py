import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds margin
    
    # Initial guess: Gaussian window
    x = np.linspace(-1, 1, N)
    sigma = 0.2
    f = np.exp(-x**2 / (2 * sigma**2))
    
    # Normalize to sum 1
    f = f / np.sum(f)
    
    # Helper to compute autoconvolution
    def get_autoconv(f):
        # Use FFT for speed
        n_fft = 2 * N
        F = np.fft.fft(f, n=n_fft)
        G = F * np.conj(F)
        g = np.fft.ifft(G).real
        # The autoconvolution is centered at N/2 in the FFT output?
        # Standard linear convolution via FFT:
        # ifft(fft(f) * conj(fft(f))) gives circular correlation?
        # Let's use np.convolve for simplicity and correctness, N=600 is small.
        g = np.convolve(f, f, mode='full')
        return g

    # Objective: Minimize the maximum of the autoconvolution.
    # Hypothesis: The "normalized maximum" might exclude the peak at 0 or normalize by it.
    # We will try to minimize the peak side-lobe.
    
    def objective(f):
        g = get_autoconv(f)
        # g is length 2N-1. Center is at index N-1.
        center = N - 1
        # Exclude the center bin (main lobe peak) to target side lobes
        # Or take the global max if that's the true metric.
        # Given the "1.266" hint, side-lobe suppression is likely the key to improvement over Uniform.
        # Uniform side lobes decay linearly. Gaussian decays exponentially.
        side_lobes = np.concatenate([g[:center], g[center+1:]])
        return np.max(side_lobes)
    
    # Simple Projected Gradient Descent
    # We want to minimize max(g_k) for k != 0.
    # Gradient of max is tricky, but we can use a soft-max approximation or just gradient of the max bin.
    
    lr = 1e-3
    best_f = f.copy()
    best_score = objective(best_f)
    
    for i in range(5000):
        if time.time() > deadline:
            break
            
        g = get_autoconv(f)
        center = N - 1
        # Find the index of the max side lobe
        side_indices = np.concatenate([np.arange(0, center), np.arange(center+1, len(g))])
        side_vals = g[side_indices]
        max_idx_local = np.argmax(side_vals)
        max_idx_global = side_indices[max_idx_local]
        
        # Gradient of g_k with respect to f:
        # g_k = sum f_i f_{i+k}
        # d(g_k)/d(f_j) = f_{j+k} + f_{j-k} (with boundary handling)
        # This is equivalent to the cross-correlation of f with a delta at k.
        # Effectively, the gradient step to reduce g_k is to subtract a shifted version of f.
        
        # Approximation: 
        # grad = 2 * (shifted_f + reversed_shifted_f)
        # But we must project back to sum=1 and f>=0.
        
        # Simpler: Random perturbation / Simulated Annealing style
        # Or use the gradient of the soft-max.
        
        # Let's use a simple heuristic: 
        # Reduce the values of f that contribute most to the peak side lobe.
        # The peak is at lag max_idx_global.
        # The contribution to g_k comes from pairs (i, i+k).
        # We can reduce f_i where f_{i+k} is large.
        
        # For speed and simplicity, we will just stick with the Gaussian if optimization is complex.
        # The Gaussian is already a strong candidate for low side lobes.
        
        # Let's try a few steps of gradient descent on the max side lobe.
        # grad_f = 2 * (np.roll(f, max_idx_global) + np.roll(f, -max_idx_global))
        # Note: np.roll is circular, need to handle linear convolution boundaries.
        # Since N=600 is small, we can compute the exact gradient for the specific lag.
        
        k = max_idx_global
        # g_k = sum_{i} f_i f_{i+k}
        # d(g_k)/d(f_j) = f_{j+k} + f_{j-k}
        # Valid indices for j: 
        # j+k < N and j-k >= 0 for the terms to exist in linear conv?
        # Actually, f is defined on [0, N-1].
        # g_k = sum_{i=max(0, -k)}^{min(N-1, N-1-k)} f_i f_{i+k}
        
        grad = np.zeros(N)
        if k > 0:
            # i from 0 to N-1-k
            # f_i contributes to f_i * f_{i+k}
            # f_{i+k} contributes to f_{i+k} * f_i
            # So d/d(f_i) = f_{i+k} for i in [0, N-1-k]
            # and d/d(f_i) = f_{i-k} for i in [k, N-1]
            grad[:N-k] += f[k:]
            grad[k:] += f[:N-k]
        else:
            # k=0 is main lobe, we excluded it.
            pass
        
        # Update
        f_new = f - lr * grad
        # Project to sum=1 and f>=0
        f_new[f_new < 0] = 0
        s = np.sum(f_new)
        if s > 0:
            f_new = f_new / s
        else:
            f_new = f
        
        score_new = objective(f_new)
        if score_new < best_score:
            best_score = score_new
            best_f = f_new
            f = f_new
        else:
            # Simple restart or step reduction
            lr *= 0.9
            if lr < 1e-6:
                break

    return list(best_f)

# EVOLVE-BLOCK-END
