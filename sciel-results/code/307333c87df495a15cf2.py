import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # We assume the evaluator normalizes the input to L1=1 or similar,
    # or the metric is scale-invariant in a way that shape matters.
    # We will optimize the shape h with sum(h)=1.
    
    def calc_score(h):
        h = np.clip(h, 0, None)
        s = np.sum(h)
        if s < 1e-12:
            return 0.0
        h_norm = h / s
        l2_sq = np.sum(h_norm**2)
        l1 = 1.0
        conv_h = np.convolve(h_norm, h_norm, mode='full')
        max_conv = np.max(conv_h)
        if max_conv < 1e-12:
            return 0.0
        return l2_sq / (l1 * max_conv)

    # Simulated Annealing
    # Start with a random sparse vector
    h = np.zeros(N)
    h[0] = 1.0
    
    best_h = h.copy()
    best_score = calc_score(best_h)
    
    T = 1.0
    T_min = 1e-6
    alpha = 0.999
    
    for i in range(5000):
        # Perturb: pick a random index and change its value
        idx = np.random.randint(N)
        # Also pick a target index to move mass
        idx2 = np.random.randint(N)
        
        h_new = h.copy()
        # Move a small amount of mass
        amount = np.random.uniform(0, 0.1)
        h_new[idx] -= amount
        h_new[idx2] += amount
        
        if np.any(h_new < -1e-9):
            continue
            
        h_new = np.clip(h_new, 0, None)
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            continue
        h_new = h_new / s_new
        
        score_new = calc_score(h_new)
        
        # Accept if better or with probability exp((new-old)/T)
        if score_new > best_score:
            best_score = score_new
            best_h = h_new.copy()
            
        delta = score_new - calc_score(h)
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            h = h_new
            
        T *= alpha
        if T < T_min:
            break

    # Final polish with Nelder-Mead on the best found
    from scipy.optimize import minimize
    
    def objective(x):
        h = np.clip(x, 0, None)
        s = np.sum(h)
        if s < 1e-12:
            return 0.0
        h = h / s
        return -calc_score(h)
        
    res = minimize(objective, best_h, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    h_opt = np.clip(res.x, 0, None)
    s_opt = np.sum(h_opt)
    if s_opt > 0:
        h_opt = h_opt / s_opt
        score_opt = calc_score(h_opt)
        if score_opt > best_score:
            best_h = h_opt
            best_score = score_opt

    return list(best_h)
# EVOLVE-BLOCK-END
