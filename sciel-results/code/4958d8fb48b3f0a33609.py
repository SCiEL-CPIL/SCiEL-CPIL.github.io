import numpy as np
import random
import time
from scipy.optimize import minimize

def compute_score(h):
    """
    Computes the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    h = np.array(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution
    conv = np.convolve(h, h)
    l_inf = np.max(conv)
    if l_inf == 0:
        return 0.0
    return (l2_sq / (l1 * l_inf))

def optimize_amplitudes(support_indices, n=50, max_iter=500):
    """
    Optimizes the amplitudes for a given set of support indices to maximize the score.
    """
    k = len(support_indices)
    if k == 0:
        return [0.0]*n, 0.0
    
    def objective(x):
        # x is the vector of amplitudes
        # We want to maximize score, so minimize -score
        # Enforce non-negativity
        x_pos = np.maximum(x, 0)
        h = np.zeros(n)
        h[support_indices] = x_pos
        return -compute_score(h)
    
    # Initial guess: equal weights
    x0 = np.ones(k) / k
    
    # Bounds: [0, inf)
    bounds = [(0, None) for _ in range(k)]
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                       options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
        x_opt = np.maximum(res.x, 0)
    except:
        x_opt = x0
        
    h_opt = np.zeros(n)
    h_opt[support_indices] = x_opt
    
    score_opt = compute_score(h_opt)
    
    # Check initial point
    h_init = np.zeros(n)
    h_init[support_indices] = x0
    score_init = compute_score(h_init)
    
    if score_init > score_opt:
        h_opt = h_init
        score_opt = score_init
        
    return h_opt.tolist(), score_opt

def construct():
    start_time = time.time()
    n = 50
    best_h = [0.0]*n
    best_score = 0.0
    
    random.seed(42)
    
    # Try various support sizes
    for k in range(1, 12):
        num_restarts = 10 if k < 8 else 5
        
        for r in range(num_restarts):
            if time.time() - start_time > 100:
                break
                
            # Random support
            support = random.sample(range(n), k)
            
            # Optimize amplitudes
            h_opt, score_opt = optimize_amplitudes(support, n=n, max_iter=300)
            
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt
            
            # Local search on support: swap one element
            current_support = support[:]
            current_h = h_opt
            current_score = score_opt
            
            for _ in range(10):
                if time.time() - start_time > 100:
                    break
                
                if not current_support:
                    break
                i_remove = random.choice(current_support)
                
                outside = [i for i in range(n) if i not in current_support]
                if not outside:
                    break
                i_add = random.choice(outside)
                
                new_support = [x for x in current_support if x != i_remove]
                new_support.append(i_add)
                
                h_new, score_new = optimize_amplitudes(new_support, n=n, max_iter=200)
                
                if score_new > best_score:
                    best_score = score_new
                    best_h = h_new
                    current_support = new_support
                    current_h = h_new
                    current_score = score_new
                elif score_new > current_score:
                    current_support = new_support
                    current_h = h_new
                    current_score = score_new

    for i in range(n):
        if best_h[i] < 0:
            best_h[i] = 0.0
            
    return best_h
