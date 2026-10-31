import numpy as np
import time

N = 600
DEADLINE = 115

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def compute_autoconv(h):
        """Compute full circular autoconvolution of h."""
        H = np.fft.fft(h, n=N)
        conv = np.fft.ifft(H * H).real
        return conv

    def compute_lse_grad(h, beta):
        """
        Compute gradient of the Log-Sum-Exp approximation of max(autoconv(h)).
        F(h) = (1/beta) * log(sum(exp(beta * C_k)))
        grad F = (2/beta) * (w * h) where w = softmax(beta * C)
        """
        C = compute_autoconv(h)
        # Subtract max for numerical stability in exp
        C_max = np.max(C)
        exp_C = np.exp(beta * (C - C_max))
        sum_exp = np.sum(exp_C)
        w = exp_C / sum_exp
        
        # Gradient is (2/beta) * (w * h)
        # w and h are length N
        W = np.fft.fft(w, n=N)
        H = np.fft.fft(h, n=N)
        grad = np.fft.ifft(W * H).real
        grad *= (2.0 / beta)
        
        # The value of the smooth objective (for monitoring)
        lse_val = (C_max + np.log(sum_exp)) / beta
        return grad, lse_val

    # Initialize with a smooth non-negative distribution
    # A raised cosine or Gaussian is better than uniform for faster convergence
    i = np.arange(N)
    center = N / 2.0
    # Gaussian with moderate width
    sigma = N / 6.0
    h = np.exp(-((i - center) ** 2) / (2 * sigma ** 2))
    h = h / np.sum(h)
    
    best_h = h.copy()
    best_val = np.max(compute_autoconv(h))
    
    # Mirror Descent / Multiplicative Weights Update
    # h_new = h * exp(-eta * grad)
    
    # Parameters
    beta = 20.0  # Initial sharpness
    eta = 0.05   # Initial step size
    n_steps = 2000
    
    for k in range(n_steps):
        if time_left() < 10:
            break
        
        # Compute gradient of smooth max
        grad, lse_val = compute_lse_grad(h, beta)
        
        # Update using Mirror Descent (Multiplicative Weights)
        # We want to minimize, so we move in direction -grad
        # h <- h * exp(-eta * grad)
        # To avoid underflow/overflow, we can normalize the exponent
        # Or just use a small eta
        
        # Clip grad to avoid extreme updates
        grad = np.clip(grad, -10, 10)
        
        h_new = h * np.exp(-eta * grad)
        
        # Normalize to sum 1
        s = np.sum(h_new)
        if s < 1e-10:
            break
        h_new = h_new / s
        
        # Ensure non-negativity (should be preserved by exp, but check for numerical issues)
        if np.any(h_new < 0):
            h_new = np.maximum(h_new, 0)
            h_new = h_new / np.sum(h_new)
        
        # Evaluate actual max autoconvolution
        C = compute_autoconv(h_new)
        curr_val = np.max(C)
        
        if curr_val < best_val:
            best_val = curr_val
            best_h = h_new.copy()
        
        h = h_new
        
        # Adaptive beta: increase sharpness over time
        # Start loose, end sharp
        if k < 100:
            beta = 10.0
        elif k < 500:
            beta = 50.0
        elif k < 1000:
            beta = 100.0
        else:
            beta = 200.0
            
        # Decay eta slightly to allow convergence
        if k % 100 == 0 and k > 0:
            eta *= 0.9

    # Final polish: Local coordinate descent or small gradient steps
    if time_left() > 5:
        h = best_h.copy()
        eta = 0.01
        for _ in range(500):
            if time_left() < 2:
                break
            grad, _ = compute_lse_grad(h, 200.0)
            h_new = h * np.exp(-eta * np.clip(grad, -10, 10))
            h_new = h_new / np.sum(h_new)
            C = compute_autoconv(h_new)
            if np.max(C) < np.max(compute_autoconv(h)):
                h = h_new
            else:
                eta *= 0.9
        if np.max(compute_autoconv(h)) < best_val:
            best_val = np.max(compute_autoconv(h))
            best_h = h

    return best_h.tolist()
