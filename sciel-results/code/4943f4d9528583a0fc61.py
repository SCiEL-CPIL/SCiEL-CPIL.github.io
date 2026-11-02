import numpy as np
from scipy.optimize import minimize

def construct():
    n = 50
    
    # Greedy construction of a Sidon set
    # We want to maximize the size k, and for a given k, minimize the max autoconvolution overlap.
    # A perfect Sidon set has max overlap 1 (in discrete sense).
    
    # Start with a known good set or build one
    # Powers of 2: 1, 2, 4, 8, 16, 32 (size 6)
    # Let's try to add more.
    
    def is_sidon(set_list):
        sums = set()
        for i in range(len(set_list)):
            for j in range(i, len(set_list)):
                s = set_list[i] + set_list[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Greedy: add smallest possible element that maintains Sidon property
    # Or try to maximize size.
    
    # We will search for the largest Sidon set in 0..49
    # Using a simple backtracking with pruning (similar to seed but potentially optimized)
    # Or use a heuristic.
    
    # Let's use a heuristic: Start with empty set, add elements one by one.
    # To maximize size, we should pick elements that "spread out".
    
    best_sidon = []
    
    # Try to build a set of size 7, then 8, etc.
    # Max possible is ~sqrt(50) = 7.
    
    def backtrack(start, current, used_sums):
        nonlocal best_sidon
        if len(current) > len(best_sidon):
            best_sidon = list(current)
            
        # Pruning: if remaining elements + current size <= best size, return
        if len(current) + (n - start) <= len(best_sidon):
            return
            
        for i in range(start, n):
            # Check if i can be added
            valid = True
            new_sums = []
            for j in current:
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
                current.append(i)
                used_sums.update(new_sums)
                backtrack(i + 1, current, used_sums)
                current.pop()
                for s in new_sums:
                    used_sums.discard(s)

    backtrack(0, [], set())
    
    if not best_sidon:
        return [1.0] + [0.0]*49
        
    k = len(best_sidon)
    
    # Initialize heights
    h = np.zeros(n)
    for idx in best_sidon:
        h[idx] = 1.0 / k
        
    # Optional: Optimize amplitudes
    # Define the score function
    def score_func(amplitudes):
        h_opt = np.zeros(n)
        for i, idx in enumerate(best_sidon):
            h_opt[idx] = amplitudes[i]
        
        # Normalize L1 to 1
        l1 = np.sum(h_opt)
        if l1 == 0:
            return 0.0
        h_opt = h_opt / l1
        
        l2_sq = np.sum(h_opt**2)
        
        # Continuous autoconvolution approximation
        # Discrete convolution
        conv = np.convolve(h_opt, h_opt, mode='full')
        # The grid spacing is 1/100? 
        # The task says "intervals of [-1/4, 1/4]".
        # If we assume the discrete sum approximates the integral:
        # ||h||_1 = sum h_i * dx
        # ||h||_2^2 = sum h_i^2 * dx
        # ||h*h||_inf = max (conv * dx)
        
        # The ratio is scale invariant, so dx cancels out.
        # Ratio = (sum h_i^2) / (sum h_i * max(conv))
        
        max_conv = np.max(conv)
        if max_conv == 0:
            return 0.0
        return l2_sq / (l1 * max_conv)
        
    # We want to maximize score.
    # Minimize -score.
    
    # Initial guess: uniform
    x0 = np.ones(k) / k
    
    # Bounds: non-negative
    bounds = [(0, None)] * k
    
    # Use Nelder-Mead
    try:
        res = minimize(lambda x: -score_func(x), x0, method='Nelder-Mead', bounds=bounds)
        if res.success:
            h_opt = np.zeros(n)
            for i, idx in enumerate(best_sidon):
                h_opt[idx] = res.x[i]
            # Normalize
            l1 = np.sum(h_opt)
            if l1 > 0:
                h_opt = h_opt / l1
            # Check if it's better
            if score_func(res.x) > score_func(x0):
                return list(h_opt)
    except Exception:
        pass
        
    return list(h)
