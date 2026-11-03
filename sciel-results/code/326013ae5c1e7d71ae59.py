"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_value = np.inf
    best_h = np.ones(N) / N  # Initial guess
    
    # Parameters for optimization
    beta = 10.0  # Sharpness of soft-max
    eta = 0.01   # Learning rate
    max_steps = 1000
    
    # Helper: Compute linear autoconvolution and value
    def get_value(h):
        # Ensure sum is 1 for scale invariance check, but value is scale invariant
        S = np.sum(h)
        if np.abs(S) < 1e-9:
            return np.inf
        c = np.convolve(h, h, mode='full')
        max_abs_c = np.max(np.abs(c))
        return max_abs_c / (S * S)
    
    # Helper: Gradient descent step
    def grad_step(h):
        S = np.sum(h)
        if np.abs(S) < 1e-9:
            return h
        # Normalize h to sum 1 for gradient calculation stability?
        # The objective is scale invariant, so gradient on normalized h is easier
        h_norm = h / S
        
        c = np.convolve(h_norm, h_norm, mode='full')
        abs_c = np.abs(c)
        
        # Soft-max weights
        # Prevent overflow
        max_abs = np.max(abs_c)
        exp_vals = np.exp(beta * (abs_c - max_abs))
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient of soft-max of |c|
        # d/dh (1/beta * log(sum(exp(beta*|c|))))
        # = (1/beta) * (sum(exp(beta*|c|) * beta * sign(c) * dc/dh) / sum(exp(beta*|c|)))
        # = sum(weights * sign(c) * dc/dh)
        sign_c = np.sign(c)
        w = weights * sign_c
        
        # Compute gradient of c wrt h
        # c = convolve(h, h)
        # dc/dh_j = sum_k w_k (h_{k-j} + h_{j-k})
        # We use FFT for efficiency
        L = 2048
        h_pad = np.zeros(L)
        h_pad[:N] = h_norm
        w_pad = np.zeros(L)
        w_pad[:len(w)] = w
        
        H = np.fft.fft(h_pad)
        W = np.fft.fft(w_pad)
        
        # B = convolve(w, h)
        B = np.fft.ifft(W * H).real
        # A = corr(w, h) -> IFFT(conj(W) * H)
        A = np.fft.ifft(np.conj(W) * H).real
        
        grad = (A + B)[:N]
        
        # Update
        h_new = h_norm - eta * grad
        
        # Project to sum 1
        h_new = h_new + (1.0 - np.sum(h_new)) / N
        
        return h_new

    # Restart 1: All ones
    h = np.ones(N)
    val = get_value(h)
    if val < best_value:
        best_value = val
        best_h = h.copy()
    
    # Restart 2: Random
    np.random.seed(42)
    for i in range(5):
        if time.time() > deadline:
            break
        
        h = np.random.randn(N)
        h = h / np.sum(h) if np.sum(h) != 0 else h + 1e-9
        
        current_val = get_value(h)
        if current_val < best_value:
            best_value = current_val
            best_h = h.copy()
            
        # Run gradient descent
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            h = grad_step(h)
            current_val = get_value(h)
            if current_val < best_value:
                best_value = current_val
                best_h = h.copy()
                
            # Adaptive learning rate?
            if step % 10 == 0:
                eta = max(0.001, eta * 0.99)

    return best_h
# EVOLVE-BLOCK-END
