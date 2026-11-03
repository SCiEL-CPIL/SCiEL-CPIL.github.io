import numpy as np
import time

N = 100

def project(h, target_sum):
    """
    Project h onto the set {x : sum(x) = target_sum, 0 <= x_i <= 1}.
    """
    # Binary search for lambda such that sum(clip(h - lambda, 0, 1)) = target_sum
    lo = -1.0
    hi = 2.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    h_proj = np.clip(h - mid, 0.0, 1.0)
    return h_proj

def compute_score(h):
    """
    Compute the maximum overlap C.
    C_k = N/2 - S_k.
    Max C = N/2 - Min S.
    """
    H = np.fft.fft(h)
    S = np.real(np.fft.ifft(H * np.conj(H)))
    min_S = np.min(S)
    return N / 2.0 - min_S

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = compute_score(best_h)
    
    # Parameters
    beta = 10.0  # Sharpness of softmax
    eta = 0.01   # Learning rate
    n_iters = 500
    n_restarts = 10
    
    for r in range(n_restarts):
        if time.time() - start_time > deadline:
            break
            
        # Initialize
        if r == 0:
            h = np.full(N, 0.5)
        else:
            # Random initialization with sum N/2
            h = np.random.rand(N)
            h = project(h, N / 2.0)
            # Add some structure? Maybe alternating or random blocks
            # Just random for now
            
        for i in range(n_iters):
            if time.time() - start_time > deadline:
                break
                
            # Compute S
            H = np.fft.fft(h)
            S = np.real(np.fft.ifft(H * np.conj(H)))
            
            # Compute C
            C = N / 2.0 - S
            
            # Compute weights
            # Avoid overflow
            C_max = np.max(C)
            exp_C = np.exp(beta * (C - C_max))
            w = exp_C / np.sum(exp_C)
            
            # Compute Gradient
            # G = 2 * circconv(w, h)
            W = np.fft.fft(w)
            G = 2.0 * np.real(np.fft.ifft(W * H))
            
            # Update
            h_new = h - eta * G
            
            # Project
            h = project(h_new, N / 2.0)
            
            # Optional: Check score periodically
            if i % 10 == 0:
                curr_score = compute_score(h)
                if curr_score < best_score:
                    best_score = curr_score
                    best_h = h.copy()
                    
        # End of restart
        curr_score = compute_score(h)
        if curr_score < best_score:
            best_score = curr_score
            best_h = h.copy()

    return best_h
