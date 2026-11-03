import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import block_diag

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _find_sidon_greedy(N, seed=None):
    """Find a large Sidon set (B_2 set) in {0, ..., N-1} using greedy with random restart."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    S = [0]
    candidates = list(range(1, N))
    rng.shuffle(candidates)
    
    existing_sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            existing_sums.add(S[i] + S[j])
    
    for x in candidates:
        new_sums = [x + s for s in S] + [2 * x]
        
        if any(ns in existing_sums for ns in new_sums):
            continue
        
        S.append(x)
        for ns in new_sums:
            existing_sums.add(ns)
            
    return S

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin
    
    # 1. Find a good initial support (Sidon set) to guide the SQP
    best_sidon = []
    for i in range(200):
        if time.time() > deadline - 10:
            break
        S = _find_sidon_greedy(N, seed=i)
        if len(S) > len(best_sidon):
            best_sidon = S
            if len(best_sidon) >= 8:
                break
    
    # If no good Sidon set found, use a sparse random set
    if not best_sidon:
        best_sidon = list(np.random.choice(N, 8, replace=False))
    
    support = np.array(sorted(best_sidon))
    k = len(support)
    
    # 2. Define the objective and gradients for SQP
    # We maximize f(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # Subject to: sum(h) = 1, h >= 0
    
    # Since the problem is scale invariant, we fix sum(h)=1.
    # Let M(h) = max_k (h*h)_k (discrete convolution, note: the DELTA factor in score_raw 
    # is constant for the optimization if we are consistent, but score_raw includes DELTA.
    # Wait, score_raw returns (l2_sq / (l1 * autoconv_max)) where autoconv_max includes DELTA.
    # So maximizing score_raw is equivalent to maximizing l2_sq / (l1 * max_conv) because DELTA is constant.
    # Let's optimize J(h) = ||h||_2^2 / (||h||_1 * max_k (h*h)_k)
    # With sum(h)=1, J(h) = ||h||_2^2 / max_k (h*h)_k.
    
    # We use a log-barrier for h >= 0.
    # Objective to minimize: -log(J(h)) + mu * sum(-log(h_i))
    # Or simply minimize -J(h) with barrier.
    # SQP handles non-smoothness of max via subgradients or active set.
    # SciPy's SLSQP can handle non-smooth objectives poorly. 
    # A better approach for non-smooth max: 
    # Introduce a variable t such that (h*h)_k <= t for all k.
    # Then maximize ||h||_2^2 / t subject to sum(h)=1, h>=0, (h*h)_k <= t.
    # This is a non-convex problem (maximize convex function ||h||^2 subject to non-convex constraints? No, (h*h)_k is quadratic, so <= t is convex constraint? 
    # (h*h)_k = sum_i h_i h_{k-i}. This is a quadratic form. 
    # h^T A_k h <= t. If A_k is positive semidefinite, this is a convex constraint.
    # A_k has 1s at (i, k-i). It is a permutation matrix (part of it). 
    # Actually, (h*h)_k = sum_{i+j=k} h_i h_j.
    # This is not a simple quadratic form h^T A h unless we expand.
    # h^T A h = sum_i,j h_i A_ij h_j.
    # If A_ij = 1 if i+j=k, then h^T A h = 2 * sum_{i<j, i+j=k} h_i h_j + sum_{i=j, 2i=k} h_i^2.
    # This is not the same as (h*h)_k which includes all pairs i,j with i+j=k.
    # Wait, (h*h)_k = sum_{i=0}^k h_i h_{k-i}.
    # h^T A_k h where A_k is the matrix with 1s at (i, k-i) gives exactly sum_i,j h_i A_ij h_j = sum_i h_i h_{k-i} = (h*h)_k.
    # So (h*h)_k = h^T A_k h.
    # Is A_k PSD? A_k is a symmetric matrix with 1s on the anti-diagonal.
    # Eigenvalues of such a matrix are +/- 1 (if N allows) or 0.
    # So A_k is NOT PSD. It is indefinite.
    # Therefore, the constraint h^T A_k h <= t is NOT convex.
    # This means the problem is non-convex.
    
    # Strategy: Use SLSQP directly on the original non-smooth objective but with a smooth approximation of the max?
    # Or use the active set method manually?
    # Given time constraints, let's try a different approach:
    # Since the max is usually achieved at a small number of points (often 1 or 2),
    # we can iterate:
    # 1. Start with h0 (Sidon set, equal weights).
    # 2. Compute h*h. Find index k* of max.
    # 3. Fix the constraint (h*h)_k* <= t.
    # 4. Maximize ||h||_2^2 / t subject to sum(h)=1, h>=0, (h*h)_k* <= t.
    #    This is still non-convex because (h*h)_k* is indefinite.
    
    # Alternative: Use the fact that for Sidon sets, the max convolution value is achieved at the "diagonal" or specific pairs.
    # For a Sidon set, all sums i+j are distinct.
    # So (h*h)_k is non-zero only for k in S+S.
    # If weights are equal, max is h_i^2 (for i=i) or h_i h_j.
    
    # Let's stick to the direct SLSQP optimization of the smooth part, 
    # but handle the max by penalizing values above the current max.
    # Objective: - ( ||h||_2^2 / (max_conv + eps) ) 
    # This is non-smooth at max_conv.
    
    # Better: Use a soft-max approximation.
    # max_k (h*h)_k approx sum_k exp((h*h)_k / tau)^(1/tau)
    # This is smooth and convex if (h*h)_k are convex? No, (h*h)_k is indefinite quadratic.
    # But the sum of exponentials of indefinite quadratics is still non-convex.
    
    # Let's try a simple coordinate ascent / local search hybrid.
    # 1. Get good support S.
    # 2. Optimize weights on S using SLSQP on the smooth objective 
    #    f(w) = sum(w^2) / max_k (conv_k(w))
    #    We can compute the gradient of the max term using the active set.
    
    # Let's implement a custom SLSQP-like loop or just use scipy SLSQP with a smoothed max.
    
    def smooth_max(x, tau=0.01):
        # Soft max
        m = np.max(x)
        return m + tau * np.log(np.sum(np.exp((x - m) / tau)))
    
    def neg_obj_smooth(w, tau=0.01):
        h = np.zeros(N)
        h[support] = w
        l1 = np.sum(h)
        if l1 == 0:
            return 0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        # h_conv values are sums of products. 
        # Note: h_conv can have negative values? No, h>=0 so h_conv >= 0.
        max_conv_smooth = smooth_max(h_conv, tau)
        if max_conv_smooth == 0:
            return 0
        return -(l2_sq / (l1 * max_conv_smooth))
    
    # Start with equal weights
    w0 = np.ones(k) / k
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0)] * k  # Small lower bound for log barrier if needed, or just 0
    
    best_h = None
    best_score = 0.0
    
    # Optimize with SLSQP
    if time.time() < deadline - 5:
        res = minimize(neg_obj_smooth, w0, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
        
        if res.success or res.fun < 0:
            h_opt = np.zeros(N)
            h_opt[support] = res.x
            # Clean up small negatives
            h_opt = np.maximum(h_opt, 0)
            # Renormalize to ensure sum=1 exactly if needed, though SLSQP should handle it
            if np.sum(h_opt) > 0:
                h_opt /= np.sum(h_opt)
            
            s_opt = _score_raw(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
    
    # If SLSQP failed or didn't improve, try a few random perturbations of the Sidon set
    if best_h is None or best_score < 1.1:
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            # Perturb support
            S_new = best_sidon.copy()
            # Remove one random element
            idx_rm = np.random.randint(len(S_new))
            S_new.pop(idx_rm)
            # Add a random new element
            candidates = [i for i in range(N) if i not in S_new]
            if candidates:
                S_new.append(np.random.choice(candidates))
            
            support_try = np.array(sorted(S_new))
            k_try = len(support_try)
            w0_try = np.ones(k_try) / k_try
            
            def neg_obj_try(w):
                h = np.zeros(N)
                h[support_try] = w
                l1 = np.sum(h)
                if l1 == 0:
                    return 0
                l2_sq = np.sum(h**2)
                h_conv = np.convolve(h, h, mode='full')
                max_conv_smooth = smooth_max(h_conv, 0.01)
                if max_conv_smooth == 0:
                    return 0
                return -(l2_sq / (l1 * max_conv_smooth))
            
            res_try = minimize(neg_obj_try, w0_try, method='SLSQP', 
                              bounds=[(1e-6, 1.0)]*k_try,
                              constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}],
                              options={'maxiter': 200, 'ftol': 1e-12})
            
            if res_try.success:
                h_try = np.zeros(N)
                h_try[support_try] = res_try.x
                h_try = np.maximum(h_try, 0)
                if np.sum(h_try) > 0:
                    h_try /= np.sum(h_try)
                s_try = _score_raw(h_try)
                if s_try > best_score:
                    best_score = s_try
                    best_h = h_try
    
    # Fallback
    if best_h is None or best_score <= 0:
        best_h = np.zeros(N)
        best_h[best_sidon] = 1.0 / len(best_sidon)
        best_score = _score_raw(best_h)
    
    return list(best_h)
