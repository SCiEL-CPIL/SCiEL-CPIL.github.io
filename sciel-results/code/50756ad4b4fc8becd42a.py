import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for a discrete measure.
    support: 1D array of shape (n,)
    weights: 1D array of shape (n,)
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Ensure normalized
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0
    w = w / total_w
    
    # 1. Pair sums S_ij = s_i + s_j
    S = s[:, None] + s[None, :]
    # 2. Thresholds T_kl = 2*s_l - s_k
    T = 2.0 * s[None, :] - s[:, None]
    
    # Weights matrix W = w w^T
    W = w[:, None] * w[None, :]
    
    # --- Compute F_kl ---
    # F_kl = sum_{i,j: S_ij < T_kl} W_ij
    S_flat = S.flatten()
    T_flat = T.flatten()
    W_flat = W.flatten()
    
    order_S = np.argsort(S_flat, kind='stable')
    S_sorted = S_flat[order_S]
    W_sorted_S = W_flat[order_S]
    cum_W_S = np.cumsum(W_sorted_S)
    
    idxs_F = np.searchsorted(S_sorted, T_flat, side='left')
    padded_cum_W_S = np.concatenate(([0.0], cum_W_S))
    F_flat = padded_cum_W_S[idxs_F]
    F = F_flat.reshape(n, n)
    
    # --- Compute P ---
    # P = sum_{k,l} w_k w_l F_kl
    P = np.sum(W * F)
    
    return float(P)

def _objective(x, n):
    """
    Objective function for minimization (we maximize P, so minimize -P).
    x is flattened [s_0, ..., s_{n-1}, w_0, ..., w_{n-1}]
    """
    s = x[:n]
    w = x[n:]
    
    # Project w to simplex to ensure validity during iterations if needed,
    # but SLSQP constraints should handle it. 
    # However, if w goes slightly negative, P calculation might be weird or invalid.
    # Let's clip w to 0 and renormalize for safety in the objective evaluation.
    w = np.maximum(w, 0)
    if w.sum() <= 1e-12:
        return 1.0 # Large penalty
    w = w / w.sum()
    
    # Ensure s is in [0, 1] (bounds handle this, but just in case)
    s = np.clip(s, 0, 1)
    
    P = _compute_P(s, w)
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # We will optimize a small number of points, e.g., 10 or 20.
    # The score improves significantly when support is optimized.
    # Let's try n=10 first, then n=20 if time permits.
    
    for n in [10, 20, 30]:
        if time.time() > deadline - 60.0:
            break
            
        # Random restarts
        n_restarts = 5 if n <= 10 else 3
        
        for r in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Initial guess
            # s: sorted random in [0, 1]
            # w: random dirichlet
            s_init = np.sort(np.random.uniform(0, 1, n))
            w_init = np.random.dirichlet(np.ones(n))
            
            x0 = np.concatenate([s_init, w_init])
            
            # Bounds for s: [0, 1], for w: [0, 1] (sum=1 constraint handles rest)
            bounds = [(0, 1)] * n + [(0, 1)] * n
            
            # Constraints: sum(w) = 1
            constraints = [{'type': 'eq', 'fun': lambda x, n=n: x[n:].sum() - 1.0}]
            
            # Objective
            res = minimize(
                _objective, 
                x0, 
                args=(n,), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success:
                s_opt = res.x[:n]
                w_opt = res.x[n:]
                
                # Clean up weights
                w_opt = np.maximum(w_opt, 0)
                if w_opt.sum() > 0:
                    w_opt = w_opt / w_opt.sum()
                else:
                    continue
                
                # Clean up support (ensure sorted for consistency, though not strictly required)
                idx = np.argsort(s_opt)
                s_opt = s_opt[idx]
                w_opt = w_opt[idx]
                
                # Calculate score
                score = _compute_P(s_opt, w_opt)
                
                if score > best_score:
                    best_score = score
                    best_support = s_opt.copy()
                    best_weights = w_opt.copy()
                    
    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    # Prune small weights
    mask = best_weights > 1e-6
    best_support = best_support[mask]
    best_weights = best_weights[mask]
    
    if len(best_support) == 0:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    else:
        best_weights = best_weights / best_weights.sum()
        
    return best_support.astype(np.float64), best_weights.astype(np.float64)
