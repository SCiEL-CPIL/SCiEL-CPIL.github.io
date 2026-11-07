import numpy as np

def heuristics(prizes, weights):
    # Handle potential shape issues
    # If prizes is (5, 100), sum to (100,)
    if prizes.ndim == 2:
        P = prizes.sum(axis=0)
    else:
        P = prizes
        
    if weights.ndim == 2:
        W = weights
    else:
        W = weights.reshape(1, -1) # unlikely, but safe
        
    # W is (5, 100)
    # Calculate Soft-Max denominator
    beta = 5.0
    # Use log-sum-exp trick for stability
    # W_max = np.max(W, axis=0)
    # log_sum_exp = W_max + np.log(np.sum(np.exp(W - W_max), axis=0))
    # W_den = log_sum_exp / beta
    
    # Direct calculation is fine for small numbers in [0,1]
    # e^(5 * 1) = e^5 ~ 148. Sum of 5 is ~740. Log is ~6.6. No overflow.
    W_den = np.log(np.sum(np.exp(beta * W), axis=0)) / beta
    
    # Heuristic
    H = P / (W_den + 1e-9)
    
    # Ensure positive
    H = np.maximum(H, 1e-9)
    
    return H
