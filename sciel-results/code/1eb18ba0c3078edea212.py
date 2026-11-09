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
    
    # Analytic 2-point baseline
    # From analysis: maximize p^2(1-p) + p(1-p)^2 * (something)
    # Actually, let's just use the known good value or compute it.
    # For {0, 1} with weights (1-p, p):
    # P = Prob[X1+X2+X3 < 2*X4]
    # X4=0 (prob 1-p): 2*X4=0. X1+X2+X3 < 0 is impossible (sum >= 0).
    # X4=1 (prob p): 2*X4=2. X1+X2+X3 < 2.
    #   X1+X2+X3 can be 0, 1, 2, 3.
    #   P(sum < 2 | X4=1) = P(sum=0) + P(sum=1)
    #   P(sum=0) = (1-p)^3
    #   P(sum=1) = 3*p*(1-p)^2
    #   So P = p * [ (1-p)^3 + 3*p*(1-p)^2 ]
    #   Let f(p) = p(1-p)^2(1-p + 3p) = p(1-p)^2(1+2p)
    #   f'(p) = (1-p)^2(1+2p) + p[ -2(1-p)(1+2p) + (1-p)^2*2 ]
    #   = (1-p)[ (1-p)(1+2p) + p(-2-4p+2p) ]
    #   = (1-p)[ 1-p+2p-2p^2 + p(-2-2p) ]
    #   = (1-p)[ 1+p-2p^2 -2p-2p^2 ]
    #   = (1-p)[ 1-p-4p^2 ]
    #   Roots: 1-p-4p^2=0 => 4p^2+p-1=0 => p = (-1+sqrt(17))/8
    #   Wait, the current code uses (15-sqrt(33))/16. Let me check.
    #   sqrt(33) ~ 5.744. (15-5.744)/16 = 9.256/16 = 0.578.
    #   (-1+sqrt(17))/8 = (-1+4.123)/8 = 3.123/8 = 0.390.
    #   Let's evaluate f(0.39) vs f(0.578).
    #   f(0.39) = 0.39 * (0.61)^2 * (1+0.78) = 0.39 * 0.3721 * 1.78 = 0.258
    #   f(0.578) = 0.578 * (0.422)^2 * (1+1.156) = 0.578 * 0.178 * 2.156 = 0.220
    #   So p ~ 0.39 is better.
    #   Why did the current code use that formula? Maybe I derived it wrong or the code is suboptimal.
    #   Let's just run optimization to find the best 2-point.
    
    def opt2(p):
        w = np.array([1-p, p])
        s = np.array([0.0, 1.0])
        return -calculate_P(s, w)
    
    res2 = minimize(opt2, 0.4, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
    if res2.fun < best_score:
        best_score = -res2.fun
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([1-res2.x[0], res2.x[0]])
        
    # Try n=3, 4, 5 with fixed support structure {0, ..., 1}
    # This reduces the search space significantly and is more likely to find good local optima quickly.
    
    for n in [3, 4, 5]:
        if time.time() > deadline:
            break
            
        # Variables: internal supports (n-2 of them) + weights (n-1 of them)
        # s = [0, t1, ..., t_{n-2}, 1]
        # w = [w0, ..., w_{n-2}, w_{n-1}] with sum=1
        
        num_t = n - 2
        num_w = n - 1
        total_vars = num_t + num_w
        
        def objective(params):
            t = params[:num_t]
            w_vars = params[num_t:]
            
            # Construct support
            s = np.zeros(n)
            s[0] = 0.0
            s[n-1] = 1.0
            if num_t > 0:
                # Ensure ordering: sort t
                t_sorted = np.sort(t)
                s[1:n-1] = t_sorted
            
            # Construct weights
            w = np.zeros(n)
            w[:num_w] = w_vars
            w[n-1] = 1.0 - np.sum(w_vars)
            
            # Validity check
            if np.any(w < -1e-9) or np.any(s < -1e-9):
                return 1.0
            
            # Clip small negatives
            w = np.clip(w, 0, None)
            w_sum = np.sum(w)
            if w_sum <= 1e-12:
                return 1.0
            w = w / w_sum
            
            return -calculate_P(s, w)
        
        # Initial guesses
        guesses = []
        
        # Uniform-ish
        t_init = np.linspace(0.1, 0.9, num_t)
        w_init = np.ones(num_w) / n
        guesses.append(np.concatenate([t_init, w_init]))
        
        # 2-point like (concentrate weight on 0 and 1)
        t_init2 = np.linspace(0.1, 0.9, num_t)
        w_init2 = np.zeros(num_w)
        w_init2[0] = 0.5
        w_init2[-1] = 0.5
        # Add small noise
        t_init2 += np.random.uniform(0, 0.05, num_t)
        w_init2 += np.random.uniform(0, 0.05, num_w)
        guesses.append(np.concatenate([t_init2, w_init2]))
        
        # Random
        for _ in range(5):
            t_rand = np.sort(np.random.uniform(0, 1, num_t))
            w_rand = np.random.uniform(0.1, 1.0, num_w)
            w_rand = w_rand / np.sum(w_rand)
            guesses.append(np.concatenate([t_rand, w_rand]))
            
        for p_init in guesses:
            if time.time() > deadline:
                break
            try:
                res = minimize(objective, p_init, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9})
                if -res.fun > best_score:
                    # Extract best params
                    t_opt = np.sort(res.x[:num_t])
                    w_vars_opt = np.clip(res.x[num_t:], 0, None)
                    w_opt = np.zeros(n)
                    w_opt[:num_w] = w_vars_opt
                    w_opt[n-1] = 1.0 - np.sum(w_vars_opt)
                    w_opt = np.clip(w_opt, 0, None)
                    w_sum = np.sum(w_opt)
                    if w_sum > 1e-12:
                        w_opt = w_opt / w_sum
                        
                        s_opt = np.zeros(n)
                        s_opt[0] = 0.0
                        s_opt[n-1] = 1.0
                        if num_t > 0:
                            s_opt[1:n-1] = t_opt
                            
                        # Verify score
                        score = calculate_P(s_opt, w_opt)
                        if score > best_score:
                            best_score = score
                            best_support = s_opt
                            best_weights = w_opt
            except Exception:
                pass
                
    # If optimization failed or found nothing, fall back to best 2-point
    if best_support is None or best_score < 0.2:
        # Use the optimized 2-point result
        if best_score < 0.2:
             # Re-evaluate the 2-point result if it was lost or not set
             pass
             
    # Ensure validity
    best_support = np.asarray(best_support, dtype=np.float64)
    best_weights = np.asarray(best_weights, dtype=np.float64)
    
    # Flatten
    best_support = best_support.flatten()
    best_weights = best_weights.flatten()
    
    # Final sanity checks
    if len(best_support) != len(best_weights) or len(best_support) < 1:
        # Fallback to simple valid distribution
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    return best_support, best_weights
