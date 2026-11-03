import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Calculate the raw score R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    Note: The autoconvolution of the step function f(t) = sum h_i delta(x_i - x_j)
    is approximated by DELTA * conv(h, h).
    The ratio is scale invariant, so we can assume ||h||_1 = 1.
    Then R(h) = ||h||_2^2 / ||h*h||_inf.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    
    # Normalize to L1 = 1 to remove scale dependency
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    
    # Autoconvolution
    # h*h is the discrete convolution. 
    # The peak of the continuous autoconvolution is approx by DELTA * max(conv(h, h)).
    # However, the constant DELTA cancels out if we are just comparing relative scores 
    # or if the evaluator uses a specific normalization. 
    # Looking at the seed score 1.0, let's assume the standard definition:
    # Score = (DELTA * sum(h^2)) / (DELTA * sum(h) * DELTA * max(conv(h,h))) ?
    # No, the task says "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution".
    # L2^2 = DELTA * sum(h_i^2)
    # L1 = DELTA * sum(h_i)
    # Supremum of autoconvolution = DELTA * max(conv(h, h))
    # Ratio = (DELTA * L2_sq_discrete) / (DELTA * L1_discrete * DELTA * MaxConv_discrete)
    #       = (1/DELTA) * (L2_sq_discrete / (L1_discrete * MaxConv_discrete))
    # Since 1/DELTA is constant, we maximize the discrete ratio.
    
    h_conv = np.convolve(h_norm, h_norm, mode='full')
    max_conv = np.max(h_conv)
    if max_conv == 0:
        return 0.0
        
    return l2_sq / max_conv

def _greedy_find_support(K, seed=42):
    """
    Greedily select K positions that minimize mutual overlap in autoconvolution.
    We start with the position that gives the best 'standalone' score (which is 1 for any single spike).
    Then we add positions that contribute least to the current peak.
    """
    rng = np.random.default_rng(seed)
    
    # Initial set: pick a random start
    support = [rng.integers(0, N)]
    
    current_h = np.zeros(N)
    current_h[support[0]] = 1.0
    
    while len(support) < K:
        best_pos = -1
        best_overlap = np.inf
        
        for pos in range(N):
            if pos in support:
                continue
            
            # Test adding this position
            test_h = current_h.copy()
            test_h[pos] = 1.0
            
            # Calculate the new peak
            conv = np.convolve(test_h, test_h, mode='full')
            peak = np.max(conv)
            
            # We want to minimize the increase in peak, or keep it low.
            # Actually, we want to maximize the final score. 
            # For equal weights, score = K / Peak. 
            # So minimizing Peak is the goal for equal weights.
            if peak < best_overlap:
                best_overlap = peak
                best_pos = pos
                
        if best_pos == -1:
            break
        support.append(best_pos)
        current_h = np.zeros(N)
        for p in support:
            current_h[p] = 1.0
            
    return support

def _optimize_weights_for_support(support):
    """
    Given a fixed support (indices), optimize the weights to maximize the score.
    Score(h) = sum(h_i^2) / max_k (sum_{i+j=k} h_i h_j)
    Subject to sum(h_i) = 1, h_i >= 0.
    """
    K = len(support)
    if K == 0:
        return np.zeros(N), 0.0
    
    # Variable vector is weights
    def objective(w):
        h = np.zeros(N)
        h[support] = w
        return -_score_raw(h)
    
    # Initial guess: uniform weights
    w0 = np.ones(K) / K
    
    bounds = [(0, 1)] * K
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    
    try:
        res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
        if res.success:
            w_opt = res.x
            h_opt = np.zeros(N)
            h_opt[support] = w_opt
            # Ensure non-negativity
            h_opt = np.maximum(h_opt, 0)
            if np.sum(h_opt) > 0:
                h_opt /= np.sum(h_opt)
            return h_opt, _score_raw(h_opt)
    except Exception:
        pass
        
    # Fallback: uniform weights
    h_uniform = np.zeros(N)
    h_uniform[support] = 1.0 / K
    return h_uniform, _score_raw(h_uniform)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (10s buffer)
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # 1. Try various sparsity levels K
    # K=1 is trivial (score 1.0). K=2,3,... might be better.
    # We search for K from 2 to 15.
    
    for K in range(2, 16):
        if time.time() > deadline:
            break
            
        # Generate multiple candidate supports for this K
        # Use different seeds to explore different "Sidon-like" structures
        num_trials = 20 if K < 8 else 5
        if time.time() > deadline - 20:
            num_trials = 2
            
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Greedy support selection
            support = _greedy_find_support(K, seed=trial)
            
            # Optimize weights for this support
            h_candidate, score_candidate = _optimize_weights_for_support(support)
            
            if score_candidate > best_score:
                best_score = score_candidate
                best_h = h_candidate.copy()
                
            # Also try random supports to escape greedy traps
            # Random support of size K
            random_support = np.random.choice(N, K, replace=False)
            h_rand, score_rand = _optimize_weights_for_support(random_support)
            if score_rand > best_score:
                best_score = score_rand
                best_h = h_rand.copy()

    # 2. Local Refinement / Perturbation of the Best Support
    # Take the best_h found, identify its support
    support_best = np.where(best_h > 1e-6)[0]
    
    # Try adding one more element to the support and re-optimizing
    if len(support_best) < N - 1:
        current_score = best_score
        for pos in range(N):
            if pos in support_best:
                continue
            new_support = list(support_best) + [pos]
            h_new, score_new = _optimize_weights_for_support(new_support)
            if score_new > best_score:
                best_score = score_new
                best_h = h_new.copy()
                support_best = np.where(best_h > 1e-6)[0]
                break # Restart search with new support
                
    # Try removing one element from the support and re-optimizing
    if len(support_best) > 2:
        for pos in support_best:
            new_support = [p for p in support_best if p != pos]
            h_new, score_new = _optimize_weights_for_support(new_support)
            if score_new > best_score:
                best_score = score_new
                best_h = h_new.copy()
                support_best = np.where(best_h > 1e-6)[0]
                break

    # 3. Final Polish with high-precision optimization on the best support
    if time.time() < deadline - 5:
        support_final = np.where(best_h > 1e-6)[0]
        if len(support_final) > 0:
            # Re-optimize with tighter tolerance
            def objective_final(w):
                h = np.zeros(N)
                h[support_final] = w
                return -_score_raw(h)
            
            K = len(support_final)
            w0 = best_h[support_final]
            bounds = [(0, 1)] * K
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            
            try:
                res = minimize(objective_final, w0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
                if res.success:
                    w_opt = res.x
                    h_opt = np.zeros(N)
                    h_opt[support_final] = w_opt
                    h_opt = np.maximum(h_opt, 0)
                    if np.sum(h_opt) > 0:
                        h_opt /= np.sum(h_opt)
                    score_opt = _score_raw(h_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_h = h_opt
            except Exception:
                pass

    return list(best_h)

# Ensure the function is available
if __name__ == "__main__":
    # Test
    h = construct()
    print(f"Score: {_score_raw(h)}")
