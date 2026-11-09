import numpy as np
import time
from scipy.optimize import minimize

def calculate_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4] for discrete distribution.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Normalize weights
    w_sum = np.sum(w)
    if w_sum <= 1e-12:
        return 0.0
    w = w / w_sum
    
    n = len(s)
    if n == 0:
        return 0.0
        
    # Precompute pairwise sums and weights
    # s_i + s_j
    S = s[:, None] + s[None, :]
    W = w[:, None] * w[None, :]
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Sum W_ij where S_ij < threshold
            mask = S < threshold
            P += w[k] * w[l] * np.sum(W[mask])
            
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    # Try different n
    for n in [2, 3, 4, 5, 6]:
        if time.time() > deadline:
            break
            
        # Objective function for optimization
        def objective(params):
            # params: [s_0, ..., s_{n-1}, logit_w_0, ..., logit_w_{n-1}]
            s_params = params[:n]
            w_logits = params[n:]
            
            # Support must be non-negative
            s = np.abs(s_params) # Simple projection to non-negative
            # To avoid all zeros which is degenerate, add small epsilon or constrain min
            # Actually, if all s are 0, P=0. Optimizer should move away.
            
            # Weights via softmax
            w_logits = w_logits - np.max(w_logits) # for stability
            w = np.exp(w_logits)
            w_sum = np.sum(w)
            w = w / w_sum
            
            return -calculate_P(s, w)
        
        # Initial guesses
        # Guess 1: Uniform
        s_init = np.linspace(0, 1, n)
        w_init = np.log(np.ones(n))
        params_init = np.concatenate([s_init, w_init])
        
        # Guess 2: Two-point like (for n>2, some weights ~0)
        if n > 2:
            s_init2 = np.zeros(n)
            s_init2[-1] = 1.0
            w_init2 = np.zeros(n)
            w_init2[0] = 0.5
            w_init2[-1] = 0.5
            # Add small noise to break symmetry
            s_init2 += np.random.uniform(0, 0.1, n)
            w_init2 += np.random.uniform(0, 0.1, n)
            params_init2 = np.concatenate([s_init2, w_init2])
        else:
            params_init2 = params_init
            
        guesses = [params_init]
        if n > 2:
            guesses.append(params_init2)
            
        # Random restarts
        for _ in range(10):
            if time.time() > deadline:
                break
            s_rand = np.sort(np.random.uniform(0, 1, n))
            w_rand = np.random.uniform(0.1, 1.0, n)
            w_rand = np.log(w_rand)
            params_rand = np.concatenate([s_rand, w_rand])
            guesses.append(params_rand)
            
        for p_init in guesses:
            if time.time() > deadline:
                break
            try:
                res = minimize(objective, p_init, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.success or res.fun < best_score: # Note: best_score is negative P
                    # Extract best params
                    s_opt = np.abs(res.x[:n])
                    w_logits = res.x[n:]
                    w_opt = np.exp(w_logits - np.max(w_logits))
                    w_opt = w_opt / np.sum(w_opt)
                    
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_support = s_opt
                        best_weights = w_opt
            except Exception:
                pass
                
    # If optimization failed or found nothing, fall back to 2-point analytic solution
    if best_support is None or best_score < 0.2:
        p = (15 - np.sqrt(33)) / 16.0
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([p, 1-p])
        
    # Ensure validity
    best_support = np.asarray(best_support, dtype=np.float64)
    best_weights = np.asarray(best_weights, dtype=np.float64)
    
    # Flatten
    best_support = best_support.flatten()
    best_weights = best_weights.flatten()
    
    return best_support, best_weights
