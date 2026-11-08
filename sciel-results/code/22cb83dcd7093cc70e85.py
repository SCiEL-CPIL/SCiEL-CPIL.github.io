import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute C3 upper bound.
    """
    f = signed_step.astype(np.float64)
    h = 1.0 / N
    n_fft = 2 * N
    F = np.fft.fft(f, n=n_fft)
    conv = np.fft.ifft(F * F).real
    autoconv_values = conv * h
    return np.max(np.abs(autoconv_values))

def construct():
    start_time = time.time()
    
    # Continuous relaxation approach
    # Variables x in [-1, 1]^N
    # Objective: min max_k |(x*x)_k|
    
    # Initialization: Use a smooth waveform to start, e.g., cosine
    # Or random
    np.random.seed(42)
    x = np.random.uniform(-1, 1, N)
    
    # Hyperparameters
    lr = 0.1
    decay = 0.999
    iterations = 5000
    
    for it in range(iterations):
        if time.time() - start_time > DEADLINE - 10.0:
            break
            
        # Compute convolution
        n_fft = 2 * N
        F = np.fft.fft(x, n=n_fft)
        conv = np.fft.ifft(F * F).real
        
        # Find peak index
        k_star = np.argmax(np.abs(conv))
        val = conv[k_star]
        
        # Gradient of |conv[k_star]| wrt x
        # d/dx_i (conv[k_star]) = 2 * x[(k_star - i) % N] 
        # Wait, the convolution is linear, not circular.
        # The gradient is sparse.
        # conv[k] = sum_j x[j] x[k-j]
        # d conv[k] / d x[i] = x[k-i] + x[k-i] ? No.
        # If we treat x as a vector, the gradient is the cross-correlation.
        # Specifically, grad[i] = 2 * x[k_star - i] if 0 <= k_star - i < N, else 0?
        # Actually, let's just use FFT to get the gradient efficiently?
        # The gradient of the max norm is in the subdifferential.
        # Approximation: gradient is proportional to the "back-projection" of the peak.
        
        # Let's construct the gradient vector g
        g = np.zeros(N)
        # For each i, contribution is x[k_star - i]
        # Indices j = k_star - i. So i = k_star - j.
        # If 0 <= j < N, then i = k_star - j.
        # If 0 <= i < N, then j = k_star - i.
        # So g[i] += x[k_star - i] if valid.
        
        # Vectorized:
        # We want g[i] = 2 * sign(val) * x[k_star - i]
        # Let's create an array of indices
        idx = np.arange(N)
        j_idx = k_star - idx
        # Valid j_idx are those in [0, N-1]
        mask = (j_idx >= 0) & (j_idx < N)
        g[mask] = 2.0 * np.sign(val) * x[j_idx[mask]]
        
        # Update
        x = x - lr * g
        
        # Project to [-1, 1]
        x = np.clip(x, -1, 1)
        
        # Decay learning rate
        lr *= decay
        
    # Round to signs
    signs = np.sign(x)
    # Handle zeros
    signs[signs == 0] = 1
    
    # Local search polish
    best_step = signs.astype(np.int8)
    best_c3 = _compute_c3(best_step)
    
    improved = True
    while improved and time.time() - start_time < DEADLINE - 2.0:
        improved = False
        for i in range(N):
            if time.time() - start_time > DEADLINE - 2.0:
                break
            best_step[i] = -best_step[i]
            c3 = _compute_c3(best_step)
            if c3 < best_c3:
                best_c3 = c3
                improved = True
            else:
                best_step[i] = -best_step[i]
                
    return best_step
