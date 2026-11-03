import numpy as np
from scipy.optimize import minimize

def construct():
    n = 50
    best_h = np.zeros(n)
    best_score = 0.0
    
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

    def optimize_weights_for_support(S):
        """
        Given a set of indices S, find weights h_i (i in S) that maximize
        sum(h_i^2) / max_k(c_k) subject to sum(h_i)=1, h_i>=0.
        """
        K = len(S)
        if K == 0:
            return np.zeros(n)
        
        S_arr = np.array(S)
        
        def objective(x):
            h_full = np.zeros(n)
            h_full[S_arr] = x
            c = np.convolve(h_full, h_full, mode='full')
            c_max = np.max(c)
            if c_max < 1e-12:
                return 0.0
            l2_sq = np.sum(x**2)
            return -(l2_sq / c_max)

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        bounds = [(0.0, 1.0) for _ in range(K)]
        
        x0 = np.ones(K) / K
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-6})
            if res.success:
                h_opt = np.zeros(n)
                h_opt[S_arr] = res.x
                h_opt = np.maximum(h_opt, 0)
                s = np.sum(h_opt)
                if s > 0:
                    h_opt /= s
                return h_opt
        except Exception:
            pass
            
        h_unif = np.zeros(n)
        h_unif[S_arr] = 1.0 / K
        return h_unif

    def generate_sidon_support():
        S = []
        used_sums = set()
        indices = list(range(n))
        np.random.shuffle(indices)
        
        for i in indices:
            valid = True
            new_sums = []
            for j in S:
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
                S.append(i)
                used_sums.update(new_sums)
                
        return S

    num_trials = 200
    for _ in range(num_trials):
        S = generate_sidon_support()
        if len(S) < 2:
            continue
            
        h_opt = optimize_weights_for_support(S)
        score = calc_score(h_opt)
        
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()

    if best_score == 0.0:
        best_h[0] = 1.0
        
    return list(best_h)
