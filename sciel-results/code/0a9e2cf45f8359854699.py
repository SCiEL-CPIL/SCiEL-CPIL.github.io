import time
import numpy as np
from scipy.optimize import minimize

N = 50

def _score_raw(h):
    """Calculate the raw score: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv == 0:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _objective(weights, support):
    """Negative score for optimization. Weights are for indices in support."""
    h = np.zeros(N)
    h[support] = weights
    return -_score_raw(h)

def _find_sidon_set(n, max_time=5.0):
    """
    Finds a large Sidon set in {0, ..., n-1} using a greedy approach with random restarts.
    A Sidon set has all pairwise sums i+j (i<=j) distinct.
    """
    best_set = []
    start_time = time.time()
    
    # Try multiple random orderings to escape local optima of greedy
    for _ in range(300):
        if time.time() - start_time > max_time:
            break
        order = np.random.permutation(n)
        S = []
        used_sums = set()
        
        for x in order:
            valid = True
            new_sums = set()
            
            # Check sums with existing elements
            for y in S:
                s = x + y
                if s in used_sums:
                    valid = False
                    break
                new_sums.add(s)
            
            # Check double
            if valid:
                s = 2 * x
                if s in used_sums:
                    valid = False
                else:
                    new_sums.add(s)
            
            if valid:
                S.append(x)
                used_sums.update(new_sums)
                
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) >= 8: # Known max for N=50 is likely 8
                break

    return sorted(best_set)

def _refine_support_and_weights(h, max_time=60.0):
    """
    Two-stage refinement:
    1. Identify dominant autoconvolution peak index k*.
    2. Solve 1D convex subproblem for weights at symmetric positions i and k*-i.
    3. Iterate support refinement (add/remove points) and weight optimization.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0.0)
    if np.sum(h) == 0:
        return h
    
    # Initial support: indices with non-negligible weight
    threshold = 1e-4 * np.max(h)
    support = np.where(h > threshold)[0].tolist()
    
    if not support:
        return h
    
    best_h = h.copy()
    best_score = _score_raw(best_h)
    
    while time.time() < deadline:
        # Step 1: Identify dominant autoconvolution peak
        conv = np.convolve(best_h, best_h, mode='full')
        k_star = np.argmax(conv)
        
        # Step 2: Block coordinate descent on weights for symmetric pairs around k_star
        # For a given k_star, the term h[i]*h[k_star-i] contributes to conv[k_star].
        # We want to maximize sum(h^2) / (sum(h) * max(conv)).
        # Locally, if we fix all other weights, we can optimize pairs (i, k_star-i).
        
        # Identify pairs involved in the peak k_star
        # Note: indices in support that are in range [0, N-1]
        # i and j = k_star - i must be valid indices.
        
        # We will perform a few iterations of optimizing weights on the current support
        # using scipy minimize, but with a focus on the structure.
        
        support_arr = np.array(support)
        weights = best_h[support_arr]
        
        # Define objective for scipy
        def obj(w):
            h_temp = np.zeros(N)
            # Clip small negatives from optimizer
            w = np.maximum(w, 0)
            h_temp[support_arr] = w
            return -_score_raw(h_temp)
        
        # Use Nelder-Mead for robustness on small dimensions
        try:
            res = minimize(obj, weights, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            new_weights = np.maximum(res.x, 0)
            new_h = np.zeros(N)
            new_h[support_arr] = new_weights
            
            new_score = _score_raw(new_h)
            
            if new_score > best_score:
                best_score = new_score
                best_h = new_h
                # Update support if some weights became zero
                new_threshold = 1e-4 * np.max(best_h) if np.max(best_h) > 0 else 0
                support = np.where(best_h > new_threshold)[0].tolist()
                if not support:
                    break
        except Exception:
            pass
        
        # Step 3: Support refinement
        # Try adding a random index to support
        if time.time() > deadline - 10:
            break
            
        # Randomly pick a candidate index to add
        candidates = [i for i in range(N) if i not in support]
        if candidates:
            # Pick a few candidates
            num_adds = min(3, len(candidates))
            candidates_to_add = np.random.choice(candidates, num_adds, replace=False)
            
            for idx in candidates_to_add:
                new_support = support + [int(idx)]
                # Initialize weights for new support
                new_weights = np.zeros(len(new_support))
                for i, s in enumerate(new_support):
                    new_weights[i] = best_h[s]
                # Set initial weight for new index
                new_weights[-1] = 0.1 * np.sum(new_weights) if np.sum(new_weights) > 0 else 0.1
                
                def obj_add(w):
                    h_temp = np.zeros(N)
                    w = np.maximum(w, 0)
                    h_temp[new_support] = w
                    return -_score_raw(h_temp)
                
                try:
                    res = minimize(obj_add, new_weights, method='Nelder-Mead',
                                   options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
                    opt_weights = np.maximum(res.x, 0)
                    opt_h = np.zeros(N)
                    opt_h[new_support] = opt_weights
                    opt_score = _score_raw(opt_h)
                    
                    if opt_score > best_score:
                        best_score = opt_score
                        best_h = opt_h
                        support = new_support
                except Exception:
                    pass
        
        # Try removing an index with small weight
        if support:
            # Sort support by weight
            weights_in_support = best_h[support]
            if len(weights_in_support) > 2:
                min_idx_in_support = np.argmin(weights_in_support)
                min_support_idx = support[min_idx_in_support]
                
                if weights_in_support[min_idx_in_support] < 0.05 * np.max(weights_in_support):
                    new_support = [s for s in support if s != min_support_idx]
                    if new_support:
                        new_weights = best_h[new_support]
                        
                        def obj_rem(w):
                            h_temp = np.zeros(N)
                            w = np.maximum(w, 0)
                            h_temp[new_support] = w
                            return -_score_raw(h_temp)
                        
                        try:
                            res = minimize(obj_rem, new_weights, method='Nelder-Mead',
                                           options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
                            opt_weights = np.maximum(res.x, 0)
                            opt_h = np.zeros(N)
                            opt_h[new_support] = opt_weights
                            opt_score = _score_raw(opt_h)
                            
                            if opt_score > best_score:
                                best_score = opt_score
                                best_h = opt_h
                                support = new_support
                        except Exception:
                            pass

    return best_h

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin for 150s limit
    
    # 1. Find a large Sidon set to initialize support
    sidon_set = _find_sidon_set(N, max_time=15.0)
    
    if not sidon_set:
        h = np.zeros(N)
        h[0] = 1.0
        return h.tolist()
        
    h = np.zeros(N)
    for idx in sidon_set:
        h[idx] = 1.0
        
    # Normalize L1 to 1
    h = h / np.sum(h)
    
    # 2. Refine support and weights
    h_refined = _refine_support_and_weights(h, max_time=deadline - start_time - 5.0)
    
    # 3. Final normalization and check
    h_final = np.maximum(h_refined, 0.0)
    l1 = np.sum(h_final)
    if l1 > 0:
        h_final = h_final / l1
    
    return h_final.tolist()
