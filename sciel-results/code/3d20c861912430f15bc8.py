import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, 20s margin
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Try a few random initializations
    rng = np.random.default_rng(42)
    
    for _ in range(5):
        h = rng.standard_normal(N)
        h = h / np.sum(h)
        
        # Gradient descent
        for step in range(500):
            if time.time() > deadline:
                break
            
            c = np.convolve(h, h)
            S = np.sum(h)
            # Value calculation
            val = abs(2 * N * np.max(c) / S**2)
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            # Softmax gradient
            beta = 10.0
            max_c = np.max(c)
            exp_c = np.exp(beta * (c - max_c))
            sum_exp = np.sum(exp_c)
            g = exp_c / sum_exp
            
            # Gradient of L w.r.t h
            # L = (2N/S^2) * (1/beta) * log(sum_exp)
            # dL/dh = (2N/S^2) * (1/beta) * (sum_exp)^-1 * sum_k exp(beta c_k) * dc_k/dh
            # dL/dh = (2N/S^2) * sum_k g_k dc_k/dh
            # Note: S is fixed during gradient step?
            # If we normalize h after step, S changes.
            # Better to treat S as constant for gradient, then renormalize.
            # Or use log-sum-exp trick carefully.
            
            # dc_k/dh_i = h_{k-i} + h_{i-k}
            grad = np.zeros(N)
            # Efficient calculation:
            # grad = 2 * (g * h) but truncated?
            # Let's use the explicit loop for safety, N=400 is small.
            # Or use np.convolve with padding.
            
            # g is length 2N-1. h is length N.
            # conv = np.convolve(g, h, mode='full')
            # The derivative is related to this.
            # Let's stick to the loop if it's fast enough.
            # 500 steps * 400*400 = 80M ops. Might be slow in Python.
            
            # Vectorized gradient:
            # dc_k/dh_i is non-zero if |k-i| < N.
            # grad[i] = sum_k g[k] * (h[k-i] + h[i-k])
            # Term 1: sum_k g[k] h[k-i]. Let j = k-i. sum_j g[j+i] h[j].
            # This is cross-correlation of g and h.
            # Term 2: sum_k g[k] h[i-k]. Let j = i-k. sum_j g[i-j] h[j].
            # This is also cross-correlation?
            
            # We can compute this using FFT or just numpy slicing.
            # For N=400, a simple loop over k might be okay if optimized?
            # No, 400*400 = 160k per step. 500 steps = 80M.
            # Python loops are slow.
            
            # Let's use the convolution property.
            # The gradient of the sum of squares of convolution is 2 * convolve(h, grad_c).
            # Here we have weighted sum.
            # grad_h = 2 * np.convolve(h, g, mode='valid')? No.
            
            # Let's just use a simple random search + local polish if gradient is too complex.
            # Or use a simpler update: h = h - alpha * (c - mean_c) * h?
            
            # Actually, let's just use the loop but break early if time is up.
            # Or reduce steps.
            
            # Let's try a different approach: 
            # Minimize max(c) is hard.
            # Minimize sum(c^2) (L2 norm) is easier.
            # If we minimize L2, the max usually decreases.
            # Objective: sum(c^2).
            # grad_h = 4 * (c * h)?
            # Let's try minimizing sum(c^2) / S^4.
            
            c_sq_sum = np.sum(c**2)
            # grad of c_sq_sum w.r.t h
            # d/dc (c^2) = 2c.
            # grad_h = 2 * (2c * h) = 4 * (c * h) ?
            # Let's check N=2.
            # c = [h0^2, 2h0h1, h1^2].
            # sum c^2 = h0^4 + 4h0^2h1^2 + h1^4.
            # d/dh0 = 4h0^3 + 8h0h1^2.
            # 4 * (c * h):
            # (c*h)_0 = c0 h0 + c1 h1 = h0^3 + 2h0h1^2.
            # 4 * (h0^3 + 2h0h1^2) = 4h0^3 + 8h0h1^2. Matches.
            
            # So grad = 4 * np.convolve(h, c, mode='valid')?
            # No, mode='full' then slice?
            # np.convolve(h, c, mode='full') has length 3N-1.
            # We need the first N elements?
            # Let's verify indices.
            # (h*c)_i = sum_j h_j c_{i-j}.
            # This matches the derivation.
            
            grad = 4 * np.convolve(h, c, mode='full')[:N]
            
            # Normalize gradient?
            h = h - 0.01 * grad
            h = h / np.sum(h)
            
    return best_h
