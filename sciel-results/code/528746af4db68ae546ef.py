"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    # Binary chirp: x_n = sign(cos(2*pi*alpha*n^2/N))
    # Optimize alpha to minimize max |autoconv(k)| for k != 0, normalized by (sum)^2
    
    def score_for_alpha(alpha):
        if time.time() > deadline - 5:
            return np.inf
        n = np.arange(N)
        x = np.sign(np.cos(2 * np.pi * alpha * n**2 / N))
        # Ensure no zeros (sign of 0 is 0, replace with 1)
        x[x == 0] = 1
        
        # Compute autoconvolution
        # Full autoconvolution has length 2N-1
        autoconv = np.correlate(x, x, mode='full')
        # autoconv[0] corresponds to lag -(N-1), autoconv[N-1] is lag 0
        center = N - 1
        
        # Sum of x
        s = np.sum(x)
        if abs(s) < 1e-10:
            return np.inf
        
        # Max absolute autoconvolution, excluding lag 0
        # We want max over all lags of |autoconv|, but the task says "maximum absolute autoconvolution"
        # Let's include all lags to be safe
        max_abs = np.max(np.abs(autoconv))
        
        return max_abs / (s * s)
    
    # Try a range of alpha values
    best_score = np.inf
    best_alpha = 0
    
    # Coarse search
    alphas = np.linspace(0, 1, 50)
    for a in alphas:
        if time.time() > deadline - 10:
            break
        s = score_for_alpha(a)
        if s < best_score:
            best_score = s
            best_alpha = a
    
    # Fine search around best
    if time.time() < deadline - 15:
        local_alphas = np.linspace(best_alpha - 0.02, best_alpha + 0.02, 20)
        local_alphas = local_alphas[(local_alphas >= 0) & (local_alphas <= 1)]
        for a in local_alphas:
            if time.time() > deadline - 5:
                break
            s = score_for_alpha(a)
            if s < best_score:
                best_score = s
                best_alpha = a
    
    # Generate final sequence
    n = np.arange(N)
    x = np.sign(np.cos(2 * np.pi * best_alpha * n**2 / N))
    x[x == 0] = 1
    
    return x
# EVOLVE-BLOCK-END
