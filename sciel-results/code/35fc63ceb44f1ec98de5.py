import numpy as np
import time

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
    # M should be large enough so that 0+0+0 < 2*1 is the only "small-small" win,
    # and 0+0+1 < 2*M is always a win, etc.
    # Specifically, we want 1+1+1 < 2*M to be true (3 < 2M => M > 1.5).
    # And 1+1+M < 2*M => 2 < M.
    # Let's pick M = 10.0 to be safe and distinct.
    
    M = 10.0
    support = np.array([0.0, 1.0, M], dtype=np.float64)
    
    # Weights: p0, p1, pM
    # p0 + p1 + pM = 1
    
    best_score = -np.inf
    best_weights = np.array([1/3, 1/3, 1/3])
    
    # Grid search over p0 and p1. pM = 1 - p0 - p1.
    # p0, p1, pM > 0
    
    # Initial coarse grid
    # p0 from 0.01 to 0.98
    # p1 from 0.01 to 0.98 - p0
    
    # To speed up, we can derive the probability analytically for fixed support.
    # P[X1+X2+X3 < 2*X4]
    # Let S = X1+X2+X3.
    # The condition is S < 2*X4.
    
    # Since support is {0, 1, 10}, the possible values for X4 are 0, 1, 10.
    # If X4=0, 2*X4=0. S must be < 0. Impossible since S >= 0.
    # If X4=1, 2*X4=2. S must be < 2. 
    #   S = X1+X2+X3.
    #   Possible sums < 2:
    #   0+0+0 = 0 (prob p0^3)
    #   0+0+1 = 1 (prob 3*p0^2*p1)
    #   0+1+0 = 1
    #   1+0+0 = 1
    #   0+0+10 = 10 (No)
    #   So P[S < 2 | X4=1] = p0^3 + 3*p0^2*p1
    
    # If X4=10, 2*X4=20. S must be < 20.
    #   Max S = 10+10+10 = 30.
    #   S < 20 excludes cases where sum >= 20.
    #   Possible sums >= 20:
    #   10+10+10 = 30 (pM^3)
    #   10+10+1 = 21 (3*pM^2*p1)
    #   10+10+0 = 20 (Strictly < 20, so 20 is NOT < 20. So 10+10+0 is excluded.)
    #   Wait, strict inequality: S < 20.
    #   S=20 is not < 20.
    #   So excluded sums are 20, 21, 30.
    #   P[S < 20 | X4=10] = 1 - P[S >= 20]
    #   P[S >= 20] = pM^3 + 3*pM^2*p1 + 3*pM^2*p0  (Note: 10+10+0=20, prob 3*pM^2*p0)
    #   Wait, is 10+10+0 the only way to get 20? Yes.
    #   Is 10+10+1=21? Yes.
    #   Is 10+10+10=30? Yes.
    #   Are there others? 10+9+... no 9. 
    #   So P[S < 20 | X4=10] = 1 - (pM^3 + 3*pM^2*p1 + 3*pM^2*p0)
    
    # Total P = P[X4=0]*P[S<0|X4=0] + P[X4=1]*P[S<2|X4=1] + P[X4=10]*P[S<20|X4=10]
    # P[X4=0] = p0
    # P[X4=1] = p1
    # P[X4=10] = pM
    
    # P = p0*0 + p1*(p0^3 + 3*p0^2*p1) + pM*(1 - (pM^3 + 3*pM^2*p1 + 3*pM^2*p0))
    # P = p1*p0^3 + 3*p1*p0^2*p1 + pM - pM^4 - 3*pM^3*p1 - 3*pM^3*p0
    # P = p1*p0^2*(p0 + 3*p1) + pM - pM^4 - 3*pM^3*(p1 + p0)
    
    # Let's verify this formula with a quick numerical check for a specific case.
    # If this formula is correct, we can optimize it analytically/numerically very fast.
    
    def calc_prob(p0, p1, pM):
        # p0 + p1 + pM = 1
        term1 = p1 * (p0**3 + 3*p0**2*p1)
        term2 = pM * (1 - (pM**3 + 3*pM**2*p1 + 3*pM**2*p0))
        return term1 + term2

    # Optimize p0, p1. pM = 1 - p0 - p1.
    # Constraints: p0 > 0, p1 > 0, pM > 0.
    
    # Grid search
    # p0 in [0.01, 0.98]
    # p1 in [0.01, 0.98 - p0]
    
    # Since the function is smooth, we can use a coarse grid and then refine.
    # Or just use scipy optimize if available, but standard grid is safer.
    
    # Let's try a grid of 100x100.
    # p0: 0.01 to 0.98
    # p1: 0.01 to 0.98 - p0
    
    # To save time, we can restrict the range.
    # Intuitively, pM should be significant to get the term2 contribution high, 
    # but p0 and p1 need to be high enough to make term1 non-negligible.
    
    # Let's do a multi-scale grid search.
    
    # Scale 1: Coarse
    step1 = 0.05
    p0_vals = np.arange(0.01, 0.98, step1)
    best_local = -np.inf
    best_p0 = 0.5
    best_p1 = 0.5
    
    for p0 in p0_vals:
        # p1 max is 0.98 - p0
        p1_max = 0.98 - p0
        if p1_max < 0.01: continue
        p1_vals = np.arange(0.01, p1_max, step1)
        for p1 in p1_vals:
            pM = 1.0 - p0 - p1
            if pM <= 0: continue
            p = calc_prob(p0, p1, pM)
            if p > best_local:
                best_local = p
                best_p0 = p0
                best_p1 = p1
                
    # Check time
    if time.time() > deadline - 10:
         # Return best found so far
         pM = 1.0 - best_p0 - best_p1
         weights = np.array([best_p0, best_p1, pM])
         return support, weights

    # Scale 2: Fine around best
    step2 = 0.005
    p0_start = max(0.01, best_p0 - 0.1)
    p0_end = min(0.98, best_p0 + 0.1)
    p0_vals2 = np.arange(p0_start, p0_end, step2)
    
    for p0 in p0_vals2:
        p1_max = 0.98 - p0
        if p1_max < 0.01: continue
        p1_start = max(0.01, best_p1 - 0.1)
        p1_end = min(p1_max, best_p1 + 0.1)
        p1_vals2 = np.arange(p1_start, p1_end, step2)
        for p1 in p1_vals2:
            pM = 1.0 - p0 - p1
            if pM <= 0: continue
            p = calc_prob(p0, p1, pM)
            if p > best_local:
                best_local = p
                best_p0 = p0
                best_p1 = p1
                
    # Scale 3: Very Fine
    step3 = 0.0005
    p0_start = max(0.01, best_p0 - 0.02)
    p0_end = min(0.98, best_p0 + 0.02)
    p0_vals3 = np.arange(p0_start, p0_end, step3)
    
    for p0 in p0_vals3:
        p1_max = 0.98 - p0
        if p1_max < 0.01: continue
        p1_start = max(0.01, best_p1 - 0.02)
        p1_end = min(p1_max, best_p1 + 0.02)
        p1_vals3 = np.arange(p1_start, p1_end, step3)
        for p1 in p1_vals3:
            pM = 1.0 - p0 - p1
            if pM <= 0: continue
            p = calc_prob(p0, p1, pM)
            if p > best_local:
                best_local = p
                best_p0 = p0
                best_p1 = p1

    pM = 1.0 - best_p0 - best_p1
    weights = np.array([best_p0, best_p1, pM])
    
    # Ensure validity
    if np.any(weights < -1e-9) or np.sum(weights) <= 1e-9:
        # Fallback to uniform 2-point if something went wrong
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    return support, weights
