import numpy as np
from scipy.optimize import minimize

def construct():
    n = 50
    
    def calc_score(h):
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        c = np.convolve(h, h, mode='full')
        c_max = np.max(c)
        if c_max == 0:
            return 0.0
        return (l2_sq / (l1 * c_max))

    def soft_max(c, beta=100.0):
        # Stable soft max
        c_max = np.max(c)
        return c_max + np.log(np.sum(np.exp(beta * (c - c_max)))) / beta

    def objective_neg_log_ratio(weights, support_indices):
        # weights are the values on support_indices
        # Reconstruct h
        h = np.zeros(n)
        h[support_indices] = weights
        
        # Normalize to sum 1 for consistent optimization landscape?
        # The ratio is scale invariant. We can fix sum(h)=1.
        # But minimize doesn't handle constraints easily unless we use them.
        # Let's just optimize the raw weights and normalize in the score?
        # No, better to optimize on the simplex.
        # Let's use a transformation: weights = exp(z) / sum(exp(z))
        # Or just use L-BFGS-B with bounds [0, 1] and check constraint.
        
        # Simple approach: 
        # The function to minimize is -log(ratio)
        # ratio = ||h||_2^2 / (||h||_1 * ||c||_inf)
        # log(ratio) = 2 log(||h||_2) - log(||h||_1) - log(||c||_inf)
        
        l1 = np.sum(h)
        if l1 < 1e-9:
            return 1e9
        l2 = np.sqrt(np.sum(h**2))
        if l2 < 1e-9:
            return 1e9
        
        c = np.convolve(h, h, mode='full')
        c_inf = soft_max(c, beta=50.0)
        
        if c_inf < 1e-9:
            return 1e9
            
        # We want to maximize ratio, so minimize -log(ratio)
        # To keep numbers stable, we can work with normalized h (sum=1)
        # But L-BFGS-B optimizes the variables directly.
        # If we don't constrain sum(h)=1, the ratio is invariant to scaling.
        # The optimizer might drift.
        # Let's normalize h inside the objective to fix the scale.
        h_norm = h / l1
        l2_norm = np.sqrt(np.sum(h_norm**2))
        c_norm = np.convolve(h_norm, h_norm, mode='full')
        c_inf_norm = soft_max(c_norm, beta=50.0)
        
        if l2_norm < 1e-9 or c_inf_norm < 1e-9:
             return 1e9
             
        # Maximize log(l2_norm^2 / c_inf_norm) = 2 log(l2_norm) - log(c_inf_norm)
        # Minimize: - (2 log(l2_norm) - log(c_inf_norm))
        val = -(2 * np.log(l2_norm) - np.log(c_inf_norm))
        return val

    best_h = np.zeros(n)
    best_score = 0.0

    # Generate multiple random Sidon sets
    for _ in range(200):
        indices = list(range(n))
        np.random.shuffle(indices)
        
        current_set = []
        used_sums = set()
        
        for i in indices:
            valid = True
            new_sums = []
            for j in current_set:
                s = i + j
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
            
            if valid:
                s_ii = i + i
                if s_ii in used_sums:
                    valid = False
                else:
                    new_sums.append(s_ii)
            
            if valid:
                current_set.append(i)
                used_sums.update(new_sums)
        
        if len(current_set) < 2:
            continue
            
        # Initialize weights uniformly on this set
        k = len(current_set)
        x0 = np.ones(k) / k
        
        # Optimize weights
        # We allow weights to be >= 0
        bounds = [(0, 1) for _ in range(k)]
        
        try:
            res = minimize(objective_neg_log_ratio, x0, args=(current_set,), 
                           method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
            
            if res.success:
                h_opt = np.zeros(n)
                h_opt[current_set] = res.x
                # Normalize
                s = np.sum(h_opt)
                if s > 0:
                    h_opt = h_opt / s
                score = calc_score(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Fallback
    if best_score == 0.0:
        best_h[0] = 1.0
        
    return list(best_h)
