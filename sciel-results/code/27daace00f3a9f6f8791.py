import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # 120s limit, leave margin
    
    # Initial guess: uniform distribution (seed)
    a = np.ones(N) / N
    
    # Helper to compute score
    def get_score(x):
        c = np.convolve(x, x, mode='full')
        return np.max(np.abs(c))
    
    current_score = get_score(a)
    best_a = a.copy()
    best_score = current_score
    
    # Gradient descent with soft-max
    # Objective: J = (1/beta) * log(sum(exp(beta * c_k^2)))
    # This approximates max(c_k^2) = (max|c_k|)^2
    # We minimize J, which minimizes max|c_k|
    
    beta = 10.0  # Sharpness of max approximation
    lr = 1e-4
    steps = 1000
    
    for i in range(steps):
        if time.time() > deadline - 5:
            break
            
        c = np.convolve(a, a, mode='full')
        c_abs_sq = c**2
        
        # Soft max log-sum-exp trick for stability
        max_val = np.max(c_abs_sq)
        if max_val == 0:
            break
        log_sum_exp = max_val + np.log(np.sum(np.exp(c_abs_sq - max_val)))
        J = log_sum_exp / beta
        
        # Gradient of J w.r.t a
        # dJ/da = (1/beta) * sum_k (2 c_k * exp(beta c_k^2) / sum(exp(beta c_j^2))) * dc_k/da
        # weights_k = exp(beta c_k^2) / sum(exp(beta c_j^2))
        weights = np.exp(beta * c_abs_sq - log_sum_exp) # Normalized weights
        grad_c = 2 * c * weights # Gradient of sum w_k c_k^2
        
        # Gradient of c w.r.t a is [a, a, ..., a, 0, ..., 0] convolved?
        # c_k = sum a_i a_{k-i}
        # dc_k/da_j = a_{k-j} + a_{j-k} (if in range)
        # This is equivalent to convolving a with the gradient of c?
        # Actually, grad_a = 2 * convolve(a, grad_c, mode='same')?
        # Let's derive:
        # c = conv(a, a)
        # grad_a = conv(a, grad_c, mode='same') * 2? 
        # No, chain rule: dL/da = 2 * conv(a, dL/dc, mode='same')
        # Note: conv(a, g, 'same') aligns centers.
        
        grad_a = 2 * np.convolve(a, grad_c, mode='same')
        
        # Project gradient to maintain sum(a) = 1
        # Subtract mean of gradient
        grad_a -= np.mean(grad_a)
        
        # Update
        a = a - lr * grad_a
        
        # Enforce sum = 1 (numerical drift correction)
        a = a / np.sum(a)
        
        new_score = get_score(a)
        if new_score < best_score:
            best_score = new_score
            best_a = a.copy()
            
        # Adaptive learning rate
        if new_score > current_score * 1.1:
            lr *= 0.5
        else:
            lr *= 1.05
            if lr > 1e-2:
                lr = 1e-2
                
        current_score = new_score
        
    return best_a
# EVOLVE-BLOCK-END
