import numpy as np
import scipy.optimize as opt
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Candidate sets of indices
    # 1. Powers of 2 plus 21
    set1 = [0, 1, 2, 4, 8, 16, 32, 21]
    
    # 2. Uniform
    set2 = list(range(N))
    
    # 3. Random sparse sets
    np.random.seed(42)
    random_sets = []
    for _ in range(20):
        size = np.random.randint(6, 10)
        s = np.random.choice(N, size, replace=False)
        random_sets.append(s)
    
    all_sets = [set1, set2] + random_sets
    
    best_score = 0.0
    best_h = None
    
    for S in all_sets:
        if time.time() > deadline:
            break
            
        S = np.array(S)
        K = len(S)
        
        # Initial guess: uniform weights
        w0 = np.ones(K) / K
        
        def objective(w):
            # w is vector of weights on S
            h = np.zeros(N)
            h[S] = w
            
            l2_sq = np.sum(w**2)
            
            # Compute autoconvolution
            # c_k = sum_{i+j=k} h_i h_j
            # Using FFT for speed? N=50 is small, direct is fine.
            c = np.convolve(h, h, mode='full')
            l_inf = np.max(c)
            
            if l_inf == 0:
                return 0.0
            
            ratio = l2_sq / l_inf
            return -ratio
        
        # Constraint: sum(w) = 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1) for _ in range(K)]
        
        try:
            res = opt.minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
            w_opt = res.x
            score = -res.fun
            if score > best_score:
                best_score = score
                h_best = np.zeros(N)
                h_best[S] = w_opt
                best_h = h_best
        except Exception:
            pass

    if best_h is None:
        # Fallback to seed
        return [1.0] * N
    
    # Normalize to sum 1 to be safe
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    
    return list(best_h)
# EVOLVE-BLOCK-END
