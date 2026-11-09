# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a 3-point support {0, 1, M} with large M.
    This allows for a "two-tier" structure where:
    1. Small values (0, 1) interact to sometimes satisfy the inequality.
    2. Large value M creates a clear separation.
    
    We optimize the weights (p0, p1, pM) using a grid search followed by local refinement.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    # Support points: 0, 1, M
    # M should be large enough so that 0+0+0 < 2*M is always true (which it is for M > 0)
    # and 1+1+1 < 2*M is true for M > 1.5
    # But more importantly, we want 0+0+M < 2*0 to be FALSE (M < 0, impossible)
    # and 0+1+M < 2*1 to be FALSE (M < 2)
    # Let's pick M = 10 to be safe.
    
    M = 10.0
    support = np.array([0.0, 1.0, M], dtype=np.float64)
    
    def calc_prob(weights):
        """
        Compute P[X1 + X2 + X3 < 2*X4] for i.i.d. X from discrete measure.
        weights: array of shape (n,)
        """
        n = len(weights)
        w = weights / np.sum(weights)
        
        # Precompute all pairwise sums and their probabilities
        # s[i] + s[j] < 2*s[l]
        # We need sum over k,l of w[k]*w[l] * (sum over i,j with s[i]+s[j] < 2*s[l]-s[k] of w[i]*w[j])
        
        # For efficiency, precompute the inner sum for each (k, l)
        # Let A[k, l] = sum_{i,j: s[i]+s[j] < 2*s[l]-s[k]} w[i]*w[j]
        # Then P = sum_{k,l} w[k]*w[l]*A[k,l]
        
        # n is small (3), so we can just do it directly
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[l] - support[k]
                # Sum w[i]*w[j] for i,j where support[i] + support[j] < threshold
                inner = 0.0
                for i in range(n):
                    for j in range(n):
                        if support[i] + support[j] < threshold:
                            inner += w[i] * w[j]
                P += w[k] * w[l] * inner
        return P
    
    # Grid search over p0, p1, pM where p0+p1+pM = 1
    best_score = 0.0
    best_weights = np.array([1.0/3, 1.0/3, 1.0/3])
    
    # Coarse grid
    step = 0.02
    for p0 in np.arange(0.0, 1.0 + step, step):
        if time.time() > deadline - 10:
            break
        for p1 in np.arange(0.0, 1.0 - p0 + step, step):
            pM = 1.0 - p0 - p1
            if pM < -1e-9:
                continue
            if pM < 0:
                pM = 0.0
            weights = np.array([p0, p1, pM])
            score = calc_prob(weights)
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
    
    # Local refinement around best found
    # Use a finer grid around the best point
    if time.time() < deadline - 5:
        p0_c, p1_c, pM_c = best_weights
        fine_step = 0.005
        for dp0 in np.arange(-0.1, 0.1 + fine_step, fine_step):
            if time.time() > deadline - 5:
                break
            for dp1 in np.arange(-0.1, 0.1 + fine_step, fine_step):
                p0_new = p0_c + dp0
                p1_new = p1_c + dp1
                pM_new = 1.0 - p0_new - p1_new
                if p0_new < 0 or p1_new < 0 or pM_new < 0:
                    continue
                weights = np.array([p0_new, p1_new, pM_new])
                score = calc_prob(weights)
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
    
    # Even finer refinement
    if time.time() < deadline - 2:
        p0_c, p1_c, pM_c = best_weights
        finer_step = 0.001
        for dp0 in np.arange(-0.05, 0.05 + finer_step, finer_step):
            if time.time() > deadline - 2:
                break
            for dp1 in np.arange(-0.05, 0.05 + finer_step, finer_step):
                p0_new = p0_c + dp0
                p1_new = p1_c + dp1
                pM_new = 1.0 - p0_new - p1_new
                if p0_new < 0 or p1_new < 0 or pM_new < 0:
                    continue
                weights = np.array([p0_new, p1_new, pM_new])
                score = calc_prob(weights)
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
    
    return support, best_weights


# EVOLVE-BLOCK-END
