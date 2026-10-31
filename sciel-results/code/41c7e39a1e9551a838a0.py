import numpy as np
import time
import itertools

N = 50

def _exact_ratio(h):
    """
    Computes the score: ||h||_2^2 / (||h||_1 * ||h * h||_inf)
    """
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _evaluate_support(support):
    """
    Evaluates the ratio for a support set with equal weights.
    """
    if len(support) == 0:
        return 0.0
    k = len(support)
    h = np.zeros(N)
    h[support] = 1.0 / k
    return _exact_ratio(h)

def _greedy_support_search(deadline):
    """
    Greedily builds a support set to maximize the ratio.
    """
    best_support = []
    best_score = 0.0
    
    # We will try to build the set incrementally.
    # At each step, we try adding each candidate index and keep the one that improves the score.
    # To speed up, we can limit the number of candidates or use a heuristic.
    
    current_support = []
    current_score = 0.0
    
    # Precompute potential pairs?
    # For N=50, checking all candidates is 50 * 50 * conv_cost.
    # Conv cost is ~50. Total ~125k ops per step. 50 steps -> 6M ops. Very fast.
    
    while time.time() < deadline:
        best_candidate = -1
        best_candidate_score = current_score
        
        # Candidates: all indices not in current_support
        # To be smarter, we can prioritize indices that are "far" from existing ones
        # But let's just check all for now, it's fast enough.
        
        candidates = [i for i in range(N) if i not in current_support]
        
        for i in candidates:
            new_support = current_support + [i]
            # Quick check: if len is small, just compute.
            score = _evaluate_support(new_support)
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = i
        
        if best_candidate == -1:
            break
        
        current_support.append(best_candidate)
        current_score = best_candidate_score
        
        if current_score > best_score:
            best_score = current_score
            best_support = current_support.copy()
            
        # If we hit a local optimum where adding any node decreases score, we stop.
        # Note: Adding a node increases K (good) but might increase C(S) (bad).
        # Usually, adding a node to a sparse set is beneficial until collisions become dense.
        
    return best_support

def _optimize_weights_on_support(support_indices, deadline):
    """
    Refines weights on the fixed support to maximize the ratio.
    """
    if not support_indices:
        return np.zeros(N)
        
    k = len(support_indices)
    h_init = np.zeros(N)
    h_init[support_indices] = 1.0 / k
    
    best_h = h_init.copy()
    best_score = _exact_ratio(best_h)
    
    # Simple local search: perturb weights
    # Since k is likely small (< 20), we can afford a few iterations.
    
    # Use Nelder-Mead on the log-weights?
    # Or just random restarts of L-BFGS-B?
    
    def objective(params):
        w = np.exp(params)
        w = w / np.sum(w)
        h = np.zeros(N)
        h[support_indices] = w
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-15:
            return 0.0
        return -l2_sq / max_conv

    params_init = np.log(1.0/k) * np.ones(k)
    
    try:
        from scipy.optimize import minimize
        res = minimize(objective, params_init, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
        w_opt = np.exp(res.x)
        w_opt = w_opt / np.sum(w_opt)
        h_opt = np.zeros(N)
        h_opt[support_indices] = w_opt
        score_opt = _exact_ratio(h_opt)
        
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt
    except Exception:
        pass

    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin

    # Step 1: Find a good sparse support using greedy search
    support = _greedy_support_search(deadline)
    
    # Step 2: Optimize weights on this support
    if time.time() < deadline - 5:
        h_final = _optimize_weights_on_support(support, deadline)
    else:
        h_final = np.zeros(N)
        if support:
            h_final[support] = 1.0 / len(support)
        else:
            h_final[0] = 1.0

    return h_final.tolist()
