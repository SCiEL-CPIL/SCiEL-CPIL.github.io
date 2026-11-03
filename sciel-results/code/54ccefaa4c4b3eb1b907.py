import numpy as np
import time

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the score: L2^2 / (L1 * Sup(autoconv))
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def is_sidon(S):
        """Check if S is a Sidon set (all pairwise sums i+j with i<=j are distinct)."""
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def find_best_sidon_set(N, target_size):
        """Find a Sidon set of target_size elements in [0, N) that maximizes spread."""
        # Greedy: try to find a Sidon set with elements spread out
        # Start with small indices and greedily add
        S = [0]
        for i in range(1, N):
            candidate = S + [i]
            if is_sidon(candidate):
                S = candidate
                if len(S) >= target_size:
                    break
        return S

    # Find a Sidon set of size ~15-20 (enough to get good ratio)
    # For N=50, a Sidon set of size ~15-20 should be findable
    target_k = 18
    sidon_set = find_best_sidon_set(N, target_k)
    K = len(sidon_set)
    sorted_sidon = sorted(sidon_set)
    
    print(f"Found Sidon set of size {K}: {sorted_sidon}")

    # Initialize with uniform weights on Sidon set
    best_h = np.zeros(N)
    best_h[sorted_sidon] = 1.0 / K
    best_score = _score_raw(best_h)
    print(f"Initial uniform Sidon score: {best_score}")

    # Now try to optimize the weights on the Sidon set
    # Key insight: for a Sidon set, the autoconvolution peak is dominated by the largest weight squared
    # To maximize L2^2 / (L1 * peak), we want weights as equal as possible
    # But we can also try slight variations to see if we can do better
    
    # Phase 1: Try different weight distributions on the Sidon set
    # The ratio for Sidon set with weights w is:
    # L2^2 / (L1 * max_conv)
    # where max_conv = max over all (i,j) of w_i * w_j (with i-j being the lag)
    # For Sidon, most lags have at most 1-2 terms
    
    # Let's do a local search on the weights
    from scipy.optimize import minimize
    
    def neg_obj_weights(w):
        """Objective for weight optimization on Sidon set."""
        h = np.zeros(N)
        h[sorted_sidon] = w
        return -_score_raw(h)
    
    # Initial weights: uniform
    w0 = np.ones(K) / K
    
    # Try multiple random initializations and local optimization
    best_w = w0.copy()
    best_w_score = best_score
    
    # First, try uniform as baseline
    for trial in range(20):
        if time.time() > deadline - 10:
            break
            
        # Generate random initial weights (positive, sum to 1)
        if trial == 0:
            w_init = w0.copy()
        else:
            # Random Dirichlet-like initialization
            w_init = np.random.dirichlet(np.ones(K))
        
        # Optimize
        try:
            bounds = [(1e-8, 1.0)] * K
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            
            res = minimize(neg_obj_weights, w_init, method='SLSQP', 
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 300, 'ftol': 1e-12})
            
            if res.success:
                w_candidate = np.clip(res.x, 1e-8, None)
                s = np.sum(w_candidate)
                if s > 0:
                    w_candidate /= s
                h_candidate = np.zeros(N)
                h_candidate[sorted_sidon] = w_candidate
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_w_score:
                    best_w_score = s_candidate
                    best_w = w_candidate.copy()
                    best_h = h_candidate.copy()
                    best_score = s_candidate
                    print(f"Trial {trial}: improved to {best_score}")
        except Exception as e:
            pass

    # Phase 2: Try adding a few more points to the Sidon set with small weights
    # This might help if the optimal solution uses slightly more than the Sidon set
    if time.time() < deadline - 5:
        # Try all possible single additions to the Sidon set
        for extra_idx in range(N):
            if extra_idx in sorted_sidon:
                continue
            if time.time() > deadline - 2:
                break
            
            candidate_support = sorted_sidon + [extra_idx]
            candidate_support = sorted(candidate_support)
            K_new = len(candidate_support)
            
            # Check if still Sidon (or close to it)
            # For now, just try uniform weights
            h_candidate = np.zeros(N)
            h_candidate[candidate_support] = 1.0 / K_new
            s_candidate = _score_raw(h_candidate)
            
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
                print(f"Added point {extra_idx}: score = {best_score}")

    # Final: ensure we return the best
    # Normalize to ensure nonnegative
    best_h = np.clip(best_h, 0, None)
    
    return list(best_h)
