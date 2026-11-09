import numpy as np
import time

def _calc_prob_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for a discrete measure.
    Vectorized for speed.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights to probabilities
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # Pairwise sums matrix: S[i,j] = s_i + s_j
    S = support[:, None] + support[None, :]
    
    # Weights outer product: W[i,j] = p_i * p_j
    W = p[:, None] * p[None, :]
    
    S_flat = S.flatten()
    W_flat = W.flatten()
    
    total_prob = 0.0
    
    # Pre-sort S_flat and corresponding W_flat for fast prefix sums
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_flat[sort_idx]
    W_cumsum = np.cumsum(W_sorted)
    W_cumsum_padded = np.concatenate(([0.0], W_cumsum))
    
    for k in range(n):
        threshold = 2.0 * support[k]
        # We need sum_l p_l * (sum_{i,j} p_i p_j I(s_i+s_j < threshold - s_l))
        # Let T_l = threshold - s_l
        T = threshold - support
        
        # Find indices in S_sorted where S_sorted < T
        # np.searchsorted returns index of first element >= T
        # So elements [0:idx] are < T
        indices = np.searchsorted(S_sorted, T, side='left')
        
        # Sum of weights for indices < T
        # W_cumsum_padded[indices] gives sum of first 'indices' elements
        sums_for_l = W_cumsum_padded[indices]
        
        # prob3 = sum_l p_l * sums_for_l
        prob3 = np.dot(p, sums_for_l)
        
        total_prob += p[k] * prob3
        
    return total_prob

def _validate(support, weights):
    if len(support) != len(weights):
        return False
    if len(support) < 1:
        return False
    if np.any(np.isnan(support)) or np.any(np.isnan(weights)):
        return False
    if np.any(np.isinf(support)) or np.any(np.isinf(weights)):
        return False
    if np.any(support < -1e-9):
        return False
    if np.any(weights < -1e-9):
        return False
    if np.sum(weights) <= 1e-9:
        return False
    return True

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game using analytical reduction and grid search.
    Strategy:
    1. Restrict to a 3-point support {0, 1, M}.
    2. The probability P is a piecewise polynomial function of weights and M.
    3. The global maximum likely occurs at boundaries or specific transition points.
    4. Perform a dense grid search over M and the weight at 1 (p1), with p0, p2 determined by normalization.
    5. Since P is linear in each weight for fixed support (and vice versa in some regimes), 
       the optimum for fixed M is often at a vertex of the weight simplex or where derivatives vanish.
       However, due to the strict inequality and discrete nature, we simply scan a fine grid.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    best_prob = 0.0
    best_support = np.array([0.0, 1.0, 2.0])
    best_weights = np.array([1.0/3, 1.0/3, 1.0/3])

    # The support is fixed as {0, 1, M}.
    # Weights are (p0, p1, p2) with p0+p1+p2=1, p_i >= 0.
    # We can parameterize by p1 and p2, with p0 = 1 - p1 - p2.
    # Or more simply, scan M, and for each M, scan the weight simplex.
    
    # To be efficient, we note that for a fixed support, P is a multilinear function of weights?
    # P = sum_{k,l} w_k w_l (sum_{i,j} w_i w_j I(...))
    # This is actually a polynomial of degree 4 in weights.
    # However, the indicator depends on M.
    
    # Let's do a 2D grid search over M and p1 (weight at 1).
    # We assume the optimal weight distribution for a given M lies on the boundary of the simplex 
    # or at a critical point. Given the complexity, a dense grid is safer.
    
    # Range for M:
    # If M is too small, it's close to uniform on {0,1,2} or similar.
    # If M is large, the term 2*M dominates.
    # Let's explore M from 0.1 to 10.0.
    
    # Grid resolution:
    # 200 points for M, 200 points for p1.
    # For each (M, p1), we need to optimize p2? 
    # Actually, for fixed M and p1, p0 = 1 - p1 - p2.
    # P is a function of p2. We can just grid search p2 as well? 
    # That would be 200*200*200 = 8M evaluations. Too slow if each eval is 1ms.
    # But _calc_prob_fast for n=3 is very fast (n^2=9 ops). 
    # 8M * 1us = 8s. Feasible.
    
    # Let's refine:
    # M in [0.1, 10.0], log-spaced or linear?
    # Linear might miss sharp transitions if they happen at specific ratios.
    # But the function is smooth except at boundaries of the indicator.
    
    m_values = np.linspace(0.1, 10.0, 100)
    p1_values = np.linspace(0.0, 1.0, 50)
    p2_values = np.linspace(0.0, 1.0, 50)
    
    # We will iterate over M, and for each M, iterate over p1, p2.
    # Constraint: p1 + p2 <= 1.
    
    for M in m_values:
        if time.time() > deadline:
            break
            
        support = np.array([0.0, 1.0, M])
        
        for p1 in p1_values:
            max_p2 = 1.0 - p1
            if max_p2 < 0:
                continue
                
            # p2 ranges from 0 to max_p2
            # We can just sample p2 in this range
            # To save time, we can coarsen the p2 grid or use the fact that for fixed M, p1,
            # P(p2) is a polynomial. But grid is robust.
            
            # Let's use 20 points for p2 to keep it fast.
            p2_grid = np.linspace(0.0, max_p2, 20)
            
            for p2 in p2_grid:
                p0 = 1.0 - p1 - p2
                if p0 < -1e-9:
                    continue
                
                weights = np.array([p0, p1, p2])
                
                # Quick validity check
                if np.any(weights < -1e-9):
                    continue
                    
                prob = _calc_prob_fast(support, weights)
                
                if prob > best_prob:
                    best_prob = prob
                    best_support = support.copy()
                    best_weights = weights.copy()
    
    # Refine the best found region with a finer grid
    # If we found a good M and p1, p2, let's zoom in.
    if best_prob > 0.5: # Arbitrary threshold to ensure we found something non-trivial
        M_best = best_support[2]
        p1_best = best_weights[1]
        p2_best = best_weights[2]
        
        # Define a small box around the best
        M_range = np.linspace(max(0.01, M_best*0.9), min(100.0, M_best*1.1), 50)
        p1_range = np.linspace(max(0.0, p1_best*0.8), min(1.0, p1_best*1.2 + 0.1), 50)
        
        for M in M_range:
            if time.time() > deadline:
                break
            support = np.array([0.0, 1.0, M])
            for p1 in p1_range:
                max_p2 = 1.0 - p1
                if max_p2 < 0:
                    continue
                p2_range = np.linspace(max(0.0, p2_best*0.8), min(max_p2, p2_best*1.2 + 0.1), 50)
                for p2 in p2_range:
                    p0 = 1.0 - p1 - p2
                    if p0 < -1e-9:
                        continue
                    weights = np.array([p0, p1, p2])
                    if np.any(weights < -1e-9):
                        continue
                    prob = _calc_prob_fast(support, weights)
                    if prob > best_prob:
                        best_prob = prob
                        best_support = support.copy()
                        best_weights = weights.copy()

    # Final cleanup
    if not _validate(best_support, best_weights):
        best_support = np.array([0.0, 1.0, 2.0])
        best_weights = np.array([1.0/3, 1.0/3, 1.0/3])
        
    # Normalize weights just in case
    w_sum = np.sum(best_weights)
    if w_sum > 1e-9:
        best_weights = best_weights / w_sum
    else:
        best_weights = np.ones(len(best_support)) / len(best_support)
        
    return best_support.astype(np.float64), best_weights.astype(np.float64)

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print("Support:", s)
    print("Weights:", w)
    print("Probability:", _calc_prob_fast(s, w))
