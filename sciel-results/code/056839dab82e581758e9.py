import numpy as np
import time
from scipy.optimize import differential_evolution

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for given support and weights."""
    n = len(support)
    w = weights / weights.sum()
    s = support
    
    # P = sum_{k,l} w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j)
    # Vectorized approach:
    # For each (k, l) pair, compute the inner sum
    
    # Build pairwise sum matrix: S[i,j] = s[i] + s[j]
    S = s[:, None] + s[None, :]
    
    # For each (k,l), threshold = 2*s[l] - s[k]
    # We need sum over (i,j) where S[i,j] < threshold of w[i]*w[j]
    
    # Build threshold matrix: T[k,l] = 2*s[l] - s[k]
    T = 2.0 * s[None, :] - s[:, None]  # T[k,l] = 2*s[l] - s[k]
    
    # For each (k,l), compute inner = sum_{i,j: S[i,j] < T[k,l]} w[i]*w[j]
    # This is O(n^4) naively, but we can vectorize
    
    # inner[k,l] = sum_{i,j} w[i]*w[j] * (S[i,j] < T[k,l])
    # = sum_i w[i] * (sum_j w[j] * (S[i,j] < T[k,l]))
    
    # Let's compute this as: for each k,l, inner = (w * (S < T[k,l]) @ w).T? No...
    # inner[k,l] = sum_i sum_j w[i]*w[j]*1[S[i,j] < T[k,l]]
    #            = w^T * M[k,l] * w where M[k,l][i,j] = 1[S[i,j] < T[k,l]]
    
    # Vectorized: 
    # Compare S (n×n) with T (n×n) broadcast: S[None, :, None, :] < T[:, None, None, :]
    # This gives a (n, n, n, n) boolean array - too big for large n
    
    # For n <= 16, n^4 = 65536, which is fine
    # For n > 16, use a different approach
    
    if n <= 20:
        # S[i,j] shape (n,n), T[k,l] shape (n,n)
        # We want: for each (k,l), sum_{i,j} w[i]*w[j]*1[S[i,j] < T[k,l]]
        # Broadcast: S[None, :, None, :] has shape (1, n, 1, n) -> (n, n, n, n)
        # T[:, None, None, :] has shape (n, 1, 1, n) -> (n, n, n, n)
        # mask = S[None, :, None, :] < T[:, None, None, :]  # shape (n, n, n, n)
        # inner = np.einsum('i,ijkl,j->kl', w, mask, w)
        
        # Actually: inner[k,l] = sum_{i,j} w[i]*w[j]*mask[k,i,l,j]
        # = np.einsum('i,ijkl,j->kl', w, mask, w)
        
        mask = (S[None, :, None, :] < T[:, None, None, :])  # (n_k, n_i, n_l, n_j)
        inner = np.einsum('i,ijkl,j->kl', w, mask, w)
        P = np.sum(w[:, None] * w[None, :] * inner)
        return P
    else:
        # Slower but memory-efficient: loop over k,l
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * s[l] - s[k]
                inner = np.sum(w[:, None] * w[None, :] * (S < threshold))
                P += w[k] * w[l] * inner
        return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep safety margin
    
    # Start with the known optimal 2-point solution
    p_opt = (1.0 + np.sqrt(33.0)) / 16.0
    support0 = np.array([0.0, 1.0])
    weights0 = np.array([p_opt, 1.0 - p_opt])
    
    best_P = _compute_P(support0, weights0)
    best_support = support0.copy()
    best_weights = weights0.copy()
    
    # Try to improve with more points using differential evolution
    # Keep n small for speed
    
    def neg_P(params):
        n = len(params) // 2
        s = params[:n]
        w = params[n:]
        
        # Ensure support is non-negative
        s = np.maximum(s, 0.0)
        # Ensure weights are non-negative
        w = np.maximum(w, 0.0)
        
        if np.sum(w) < 1e-9:
            return 0.0
        
        # Sort by support for consistency
        idx = np.argsort(s)
        s = s[idx]
        w = w[idx]
        
        return -_compute_P(s, w)
    
    # Try n=3, 4, 5, 6 with DE
    for n in [3, 4, 5, 6, 8, 12, 16]:
        if time.time() > deadline - 120.0:
            break
        
        # Bounds: support in [0, 10], weights in [0, 1]
        bounds = [(0.0, 10.0)] * n + [(0.0, 1.0)] * n
        
        # Use multiple restarts
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 60.0:
                break
            
            try:
                result = differential_evolution(
                    neg_P,
                    bounds=bounds,
                    maxiter=200,
                    popsize=15,
                    tol=1e-6,
                    seed=restart,
                    polish=True,
                    init='sobol'
                )
                
                params = result.x
                m = len(params) // 2
                s = np.maximum(params[:m], 0.0)
                w = np.maximum(params[m:], 0.0)
                
                if np.sum(w) < 1e-9:
                    continue
                
                idx = np.argsort(s)
                s = s[idx]
                w = w[idx]
                
                P_val = _compute_P(s, w)
                
                if P_val > best_P:
                    best_P = P_val
                    best_support = s.copy()
                    best_weights = w.copy()
                    
            except Exception:
                continue
    
    # Final validation
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    if np.sum(best_weights) < 1e-9:
        # Fallback to 2-point solution
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([p_opt, 1.0 - p_opt])
    
    return best_support, best_weights
