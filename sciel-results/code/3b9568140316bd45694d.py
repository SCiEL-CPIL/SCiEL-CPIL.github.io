import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def compute_p(s, w):
    # s: support, w: weights (normalized)
    # P = sum_{k,l} w_k w_l * (sum_{i,j: s_i+s_j < 2s_l-s_k} w_i w_j)
    n = len(s)
    # Precompute all pair sums s_i + s_j and their weights w_i w_j
    # There are n^2 pairs
    S = s[:, None] + s[None, :] # shape (n, n)
    W = w[:, None] * w[None, :] # shape (n, n)
    
    # We need for each pair (k, l) the sum of W[i, j] where S[i, j] < 2*s[l] - s[k]
    # Let A = 2*s[l] - s[k] for all k, l
    A = 2 * s[None, :] - s[:, None] # shape (n, n) where A[k, l] = 2*s_l - s_k
    
    # For efficiency, flatten S and W
    S_flat = S.flatten()
    W_flat = W.flatten()
    
    # Sort S_flat and compute cumulative sum of W_flat
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_cumsum = np.cumsum(W_flat[sort_idx])
    
    # For each A[k, l], find the index where S_sorted < A[k, l]
    # np.searchsorted returns the index where the value would be inserted to maintain order
    # We want strict inequality: S < A
    # So we use side='left'
    indices = np.searchsorted(S_sorted, A.flatten(), side='left')
    
    # Get the cumulative sum at these indices
    # If index is 0, sum is 0. If index is len, sum is total.
    # We need to handle the case where index == len(S_sorted)
    # W_cumsum has length n^2.
    # We can pad W_cumsum with 0 at the beginning to make indexing easier?
    # Or just use np.take with clip.
    # Let's create an array of cumulative sums with a 0 prepended.
    W_cumsum_padded = np.concatenate(([0.0], W_cumsum))
    
    prob_terms = W_cumsum_padded[indices]
    
    # Now sum over k, l with weights w_k w_l
    W_kl = w[:, None] * w[None, :]
    P = np.sum(W_kl * prob_terms.reshape(n, n))
    return P

def beat_the_average_measure():
    start_time = time.time()
    deadline = start_time + 600 # 600s limit, keep 30s margin
    
    # Parameters
    n = 10 # Number of support points
    
    # Initial guess
    # Let's try a geometric distribution or uniform on [0, 10]
    u0 = np.zeros(n) # s_i = softplus(0) = ln(2) ~ 0.69
    v0 = np.zeros(n) # w_i = exp(0) = 1, normalized to 1/n
    
    def softplus(x):
        return np.log1p(np.exp(x))
        
    def objective(params):
        u = params[:n]
        v = params[n:2*n]
        s = softplus(u)
        w_raw = np.exp(v)
        w = w_raw / np.sum(w_raw)
        return -compute_p(s, w)
    
    best_score = -1.0
    best_params = None
    
    # Try a few restarts with different initializations
    for restart in range(5):
        if time.time() > deadline - 30:
            break
            
        if restart == 0:
            params0 = np.concatenate([u0, v0])
        else:
            # Random initialization
            u_rand = np.random.uniform(-2, 5, n)
            v_rand = np.random.uniform(-2, 2, n)
            params0 = np.concatenate([u_rand, v_rand])
            
        res = minimize(objective, params0, method='Powell', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
        # Check time
        if time.time() > deadline - 30:
            break
            
    # Also try differential evolution if time permits?
    # Powell is usually faster for this size.
    
    if best_params is None:
        # Fallback to uniform
        s = np.arange(n, dtype=np.float64)
        w = np.ones(n, dtype=np.float64) / n
        return s, w
        
    u = best_params[:n]
    v = best_params[n:2*n]
    s = softplus(u)
    w_raw = np.exp(v)
    w = w_raw / np.sum(w_raw)
    
    return s, w
