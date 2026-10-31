import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    # Step 1: Find a good Sidon-like set
    def find_sidon_set_greedy(n, target_size):
        # Greedy selection to minimize max sum multiplicity
        best_set = []
        for restart in range(20):
            current = []
            sums_count = np.zeros(2*n, dtype=int)
            # Random start point or fixed 0?
            # Let's try to fill up to target_size
            # To make it robust, we can shuffle the order of candidates
            candidates = list(range(n))
            np.random.shuffle(candidates)
            
            # Always include 0 for symmetry/simplicity? Or not.
            # Let's just pick greedily.
            for i in candidates:
                if len(current) >= target_size:
                    break
                # Check impact of adding i
                # New sums: i+i, i+j for j in current
                new_sums = [2*i]
                for j in current:
                    new_sums.append(i+j)
                
                # Calculate max count if we added i
                # We need to know current max
                # Actually, we can just simulate
                temp_counts = sums_count.copy()
                for s in new_sums:
                    temp_counts[s] += 1
                max_count = np.max(temp_counts)
                
                # Heuristic: Accept if max_count is small (e.g. <= 2)
                # Or accept if it improves the "score"
                # For Sidon, we want max_count <= 2.
                if max_count <= 2:
                    current.append(i)
                    sums_count = temp_counts
                    if len(current) == target_size:
                        break
            
            if len(current) > len(best_set):
                best_set = current
            if len(best_set) >= target_size:
                break
        
        return best_set

    # Try to find a set of size 8, then 7, etc.
    best_sidon = []
    for k in range(10, 3, -1):
        s = find_sidon_set_greedy(N, k)
        if len(s) == k:
            best_sidon = s
            break
    
    if not best_sidon:
        best_sidon = [0, 1, 4, 10]
        
    k = len(best_sidon)
    
    # Step 2: Optimize weights
    # h is supported on best_sidon
    # w are the weights
    def objective(w):
        # w is array of length k
        # Construct h
        h = np.zeros(N)
        h[best_sidon] = w
        # Calculate Q
        Q = np.sum(h**2)
        if Q == 0:
            return 1e10
        # Calculate M
        conv = np.convolve(h, h)
        M = np.max(conv)
        # We want to minimize M / Q
        # Note: S = sum(w) = 1 is enforced by constraint
        return M / Q

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: w} # w >= 0
    ]
    
    bounds = [(1e-9, 1.0)] * k
    
    # Initial guess: uniform
    w0 = np.ones(k) / k
    
    # Optimize
    # Nelder-Mead is robust for non-smooth (max)
    res = minimize(objective, w0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    w_opt = res.x
    
    # Ensure non-negative and sum to 1
    w_opt = np.clip(w_opt, 0, None)
    w_opt = w_opt / np.sum(w_opt)
    
    # Construct final h
    h = np.zeros(N)
    h[best_sidon] = w_opt
    
    return list(h)
