# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_P_3pt(M, p1, p0, pM):
    """
    Compute P[X1+X2+X3 < 2*X4] for a 3-point support {0, 1, M} with weights (p0, p1, pM).
    Weights are assumed to sum to 1.
    """
    if M < 1.0:
        # Swap roles if needed, but we assume M >= 1 in search.
        # If M < 1, the order is {0, M, 1}. We handle M >= 1 only.
        return 0.0

    # Support: s = [0, 1, M]
    # Weights: w = [p0, p1, pM]
    # Indices: 0, 1, 2
    
    # P = sum_{k,l} w_k w_l * (sum_{i,j: s_i + s_j < 2*s_l - s_k} w_i w_j)
    
    # Let's iterate over k (X4 index) and l (X4 index in formula? No, formula is sum over k,l of w_k w_l ... )
    # Wait, the formula in the prompt is:
    # P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)
    # Here X4 corresponds to index k? Or l?
    # "X1 + X2 + X3 < 2*X4"
    # X4 is one variable. Let's say X4 takes value s_k with prob w_k.
    # Then X1, X2, X3 are i.i.d.
    # The term inside is Prob[X1 + X2 + X3 < 2*s_k | X4=s_k].
    # But X1, X2, X3 are 3 variables. The formula has w_i w_j. That's only 2 variables.
    # Ah, the prompt says: "P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)"
    # This looks like it might be for X1+X2 < 2*X3? No, the task says X1+X2+X3 < 2*X4.
    # Let's re-read carefully.
    # "Metric ... P = Prob[X1 + X2 + X3 < 2*X4]"
    # "P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)"
    # This formula in the prompt seems inconsistent with 3 variables on LHS.
    # Usually, if X1, X2, X3 are i.i.d., P[X1+X2+X3 < T] is a convolution of 3.
    # The formula provided has w_k w_l (2 weights) and inner sum w_i w_j (2 weights). Total 4 weights.
    # So it IS for 4 variables.
    # Let's map indices to variables.
    # Variables: X1, X2, X3, X4.
    # Condition: X1 + X2 + X3 < 2*X4.
    # Let X4 = s_k. Prob = w_k.
    # Then we need Prob[X1 + X2 + X3 < 2*s_k].
    # This is sum over values of X1, X2, X3.
    # The prompt's formula: sum_{k,l} w_k w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i w_j)
    # This looks like:
    # Outer sum over k, l.
    # Inner sum over i, j.
    # Condition: s_i + s_j < 2*s_l - s_k.
    # Rearranging: s_i + s_j + s_k < 2*s_l.
    # This corresponds to X1+X2+X3 < 2*X4 if we identify:
    # X4 -> s_l
    # X1, X2, X3 -> s_i, s_j, s_k
    # So the sum is over all quadruples (i,j,k,l) such that s_i + s_j + s_k < 2*s_l.
    # The weight is w_i w_j w_k w_l.
    # So P = sum_{i,j,k,l: s_i+s_j+s_k < 2*s_l} w_i w_j w_k w_l.
    
    # My loop structure:
    # for l in range(3):  # X4
    #   for k in range(3): # X3
    #     for j in range(3): # X2
    #       for i in range(3): # X1
    #         if s[i] + s[j] + s[k] < 2*s[l]:
    #            P += w[i]*w[j]*w[k]*w[l]
    
    # This is 3^4 = 81 operations. Very fast.
    
    s = np.array([0.0, 1.0, M], dtype=np.float64)
    w = np.array([p0, p1, pM], dtype=np.float64)
    
    P = 0.0
    for l in range(3):
        target = 2.0 * s[l]
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    if s[i] + s[j] + s[k] < target:
                        P += w[i] * w[j] * w[k] * w[l]
    return P


def _search_3pt():
    """
    Search over 3-point support {0, 1, M} with M >= 1.
    Weights: p0, p1, pM with p0+p1+pM=1.
    We can parameterize by M, p1, and p0 (pM = 1-p0-p1).
    Or M, p1, pM (p0 = 1-p1-pM).
    
    Let's do a grid search over M and p1, and optimize p0/pM analytically or with a 1D search?
    Actually, for fixed M and p1, P is a function of p0 (and pM=1-p1-p0).
    It's a polynomial in p0. We can just do a fine grid for p0 as well.
    """
    best_P = -1.0
    best_params = None
    
    # M range: 1 to 10?
    # p1 range: 0 to 1
    # p0 range: 0 to 1-p1
    
    # Let's use a coarse grid first, then refine.
    # Deadline guard: 600s total. This function should be fast.
    start_time = time.time()
    
    # Coarse grid
    M_vals = np.linspace(1.0, 10.0, 200)
    p1_vals = np.linspace(0.0, 1.0, 200)
    
    for M in M_vals:
        for p1 in p1_vals:
            # p0 + pM = 1 - p1
            # Let p0 vary from 0 to 1-p1
            # We can just check endpoints and a few points?
            # P is a polynomial in p0 of degree 3?
            # w0=p0, w1=p1, wM=1-p1-p0.
            # P = sum ...
            # It's a cubic in p0. A cubic on an interval has at most 2 extrema.
            # We can just sample a few points for p0.
            p0_vals = np.linspace(0.0, 1.0-p1, 10)
            for p0 in p0_vals:
                pM = 1.0 - p1 - p0
                if pM < 0: continue
                P = _compute_P_3pt(M, p1, p0, pM)
                if P > best_P:
                    best_P = P
                    best_params = (M, p1, p0, pM)
        
        if time.time() - start_time > 500:
            break
    
    return best_P, best_params


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized;
            the evaluator normalizes them to sum to 1).
    """
    # Try the 3-point analytical/grid search
    best_P, params = _search_3pt()
    
    if params is not None:
        M, p1, p0, pM = params
        # Filter out zero weights to keep n small?
        # The evaluator handles any n. But smaller n is faster for the evaluator's n^2 check?
        # The evaluator builds n^2 pairwise sums. n=3 is trivial.
        support = np.array([0.0, 1.0, M], dtype=np.float64)
        weights = np.array([p0, p1, pM], dtype=np.float64)
        
        # Clean up tiny negatives or zeros
        weights[weights < 1e-12] = 0.0
        
        # If all weights are zero (shouldn't happen), fallback
        if np.sum(weights) < 1e-9:
             # Fallback to baseline
             n = 32
             support = np.arange(n, dtype=np.float64)
             weights = np.ones(n, dtype=np.float64) / n
        else:
            # Remove points with zero weight to minimize n
            mask = weights > 1e-12
            if np.sum(mask) < len(support):
                support = support[mask]
                weights = weights[mask]
                
        return support, weights
    else:
        # Fallback to baseline
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights

# EVOLVE-BLOCK-END
