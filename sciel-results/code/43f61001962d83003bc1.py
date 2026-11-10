import numpy as np
import time
import sys

def _precompute_tables():
    """Precompute index mappings for fast bincount-based ratio evaluation."""
    n = 10
    y_idx, x_idx = np.indices((n, n))
    
    # x - y range: -9 to 9 (19 values)
    diff = x_idx - y_idx
    diff_idx = diff + 9
    
    # x range: 0 to 9 (10 values)
    x_idx_flat = x_idx
    
    # x + y range: 0 to 18 (19 values)
    sum_idx = x_idx + y_idx
    
    # x + 2y range: 0 to 27 (28 values)
    sum2_idx = x_idx + 2 * y_idx
    
    return diff_idx.ravel(), x_idx_flat.ravel(), sum_idx.ravel(), sum2_idx.ravel()

# Pre-compute indices for fast evaluation
DIFF_IDX, X_IDX, SUM_IDX, SUM2_IDX = _precompute_tables()

def _fast_ratio(P):
    """
    Fast computation of D/M ratio.
    P is a 10x10 array.
    Returns 0.0 if invalid.
    """
    if P.shape != (10, 10):
        return 0.0
    if not np.all(np.isfinite(P)):
        return 0.0
    if np.any(P < -1e-9):
        return 0.0
    total_sum = np.sum(P)
    if total_sum <= 1e-9:
        return 0.0
    
    # Normalize for probability calculations
    p = P / total_sum
    
    # Calculate pooled probabilities using bincount
    # H_{-1}: distribution of x - y
    counts_diff = np.bincount(DIFF_IDX, weights=p.ravel(), minlength=19)
    probs_diff = counts_diff / counts_diff.sum()
    probs_diff = probs_diff[probs_diff > 0]
    H_diff = -np.sum(probs_diff * np.log2(probs_diff)) if len(probs_diff) > 0 else 0.0
    
    # H_0: distribution of x
    counts_x = np.bincount(X_IDX, weights=p.ravel(), minlength=10)
    probs_x = counts_x / counts_x.sum()
    probs_x = probs_x[probs_x > 0]
    H_x = -np.sum(probs_x * np.log2(probs_x)) if len(probs_x) > 0 else 0.0
    
    # H_1: distribution of x + y
    counts_sum = np.bincount(SUM_IDX, weights=p.ravel(), minlength=19)
    probs_sum = counts_sum / counts_sum.sum()
    probs_sum = probs_sum[probs_sum > 0]
    H_sum = -np.sum(probs_sum * np.log2(probs_sum)) if len(probs_sum) > 0 else 0.0
    
    # H_2: distribution of x + 2y
    counts_sum2 = np.bincount(SUM2_IDX, weights=p.ravel(), minlength=28)
    probs_sum2 = counts_sum2 / counts_sum2.sum()
    probs_sum2 = probs_sum2[probs_sum2 > 0]
    H_sum2 = -np.sum(probs_sum2 * np.log2(probs_sum2)) if len(probs_sum2) > 0 else 0.0
    
    M = max(H_x, H_sum, H_sum2)
    
    if M <= 1e-9:
        return 0.0
        
    return H_diff / M

def _generate_structured_candidates():
    """
    Generate a set of structured initial candidates based on arithmetic properties.
    We want to maximize H(x-y) and minimize M = max(H(x), H(x+y), H(x+2y)).
    Ideally, H(x) is low (concentrated x), but H(x-y) is high (spread out differences).
    This suggests putting mass on cells where x is fixed (or few values) but y varies widely,
    BUT we must check H(x+y) and H(x+2y).
    
    If x is constant c, then x-y = c-y, so H(x-y) = H(y).
    x+y = c+y, so H(x+y) = H(y).
    x+2y = c+2y, so H(x+2y) = H(2y).
    M = max(0, H(y), H(2y)) = max(H(y), H(2y)).
    Ratio = H(y) / max(H(y), H(2y)).
    If y is uniform, H(y) = log2(10) ~ 3.32.
    2y values: 0,2,4,6,8,10,12,14,16,18. All distinct. H(2y) = log2(10) ~ 3.32.
    Ratio = 1.0.
    
    Can we do better?
    We want H(x-y) > M.
    Let's try to make H(x) small.
    If we put mass on a diagonal x-y = k, then H(x-y) = 0. Bad.
    If we put mass on a vertical line x=c, ratio is 1.0 (as above).
    
    What if we mix?
    Consider a distribution supported on a subset of cells.
    Let's try to maximize the ratio by finding a subset of cells S such that the projection
    onto (x-y) is "more spread out" (higher entropy) than the projections onto x, x+y, x+2y.
    
    Actually, note that the map (x,y) -> (x-y, x) is invertible (given x-y and x, we get y).
    So H(x-y, x) = H(x,y).
    H(x-y) <= H(x,y).
    Also H(x) <= H(x,y).
    
    We want H(x-y) / max(H(x), H(x+y), H(x+2y)) to be large.
    This implies H(x-y) should be close to H(x,y) and H(x) should be much smaller than H(x,y).
    This happens if x is determined by x-y with high probability? No, that would make H(x) ~ H(x-y).
    We want H(x) << H(x-y). This means x is highly concentrated, while x-y is spread out.
    This implies y is spread out, and x is concentrated.
    As analyzed, if x is constant, ratio is 1.
    
    Is it possible to get ratio > 1?
    Let's check the upper bound.
    H(x-y) <= H(x,y).
    M >= H(x).
    If H(x) is small, M is small? No, M is max of three entropies.
    If x is constant, H(x)=0. M = max(H(x+y), H(x+2y)).
    If y is uniform, H(x+y) = H(y) and H(x+2y) = H(2y).
    For uniform y in 0..9, H(y) = log2(10).
    Values of 2y: 0,2,...,18. 10 distinct values. H(2y) = log2(10).
    So M = log2(10). Ratio = 1.
    
    What if y is not uniform?
    Let y take values 0 and 9 with prob 0.5 each.
    H(y) = 1 bit.
    x-y: -9 and 9 (if x=0). H(x-y) = 1 bit.
    x+y: 0 and 9. H(x+y) = 1 bit.
    x+2y: 0 and 18. H(x+2y) = 1 bit.
    Ratio = 1.
    
    It seems 1.0 is a strong baseline for "vertical lines".
    Can we exceed 1.0?
    We need H(x-y) > max(H(x), H(x+y), H(x+2y)).
    
    Let's try a 2x2 block of cells.
    (0,0), (0,1), (1,0), (1,1).
    x: 0,0,1,1. H(x) = 1.
    x-y: 0, -1, 1, 0. Values: 0 (prob 0.5), -1 (0.25), 1 (0.25).
    H(x-y) = -0.5*log2(0.5) - 0.25*log2(0.25) - 0.25*log2(0.25) = 0.5 + 0.5 + 0.5 = 1.5.
    x+y: 0, 1, 1, 2. Values: 0(0.25), 1(0.5), 2(0.25). H(x+y) = 1.5.
    x+2y: 0, 2, 1, 3. Values: 0,1,2,3 all 0.25. H(x+2y) = 2.
    M = max(1, 1.5, 2) = 2.
    Ratio = 1.5 / 2 = 0.75. Worse.
    
    Try a different shape.
    We want H(x+2y) to be small.
    x+2y = k.
    If we pick cells on the line x+2y = k, then H(x+2y)=0.
    Let's pick cells on x+2y=0. Only (0,0). H=0.
    Pick cells on x+2y=1. (1,0).
    Pick cells on x+2y=2. (0,1), (2,0).
    
    Let's try to construct a distribution where x is concentrated, but x-y is spread.
    This requires y to vary.
    If x is concentrated at x=0, y varies.
    H(x)=0.
    H(x-y) = H(-y) = H(y).
    H(x+y) = H(y).
    H(x+2y) = H(2y).
    Ratio = H(y) / max(H(y), H(2y)).
    Since H(2y) <= H(y) is not necessarily true?
    Let y be uniform on 0..9. H(y)=log2(10). H(2y)=log2(10). Ratio 1.
    Let y be 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.
    What if we choose y values such that 2y has collisions?
    2y is always distinct for distinct integer y.
    So H(2y) = H(y) always for uniform-like distributions on distinct integers?
    No, entropy depends on probabilities.
    If p_y is the distribution of y, then p_{2y}(2y) = p_y(y).
    So the multiset of probabilities is the same.
    Thus H(2y) = H(y) ALWAYS for any distribution on distinct integers mapped by an injective function.
    Since y -> 2y is injective on integers, H(2y) = H(y).
    Therefore, for any distribution with x constant, M = H(y), D = H(y), Ratio = 1.
    
    So we MUST have x vary to get Ratio > 1?
    If x varies, H(x) > 0.
    We need H(x-y) > H(x).
    And H(x-y) > H(x+y) and H(x-y) > H(x+2y).
    
    Let's try a distribution where x takes few values, y takes many.
    Example: x in {0, 1}, y in 0..9.
    If p(x=0)=0.5, p(x=1)=0.5.
    y independent uniform.
    H(x) = 1.
    x-y:
    If x=0, x-y = -y.
    If x=1, x-y = 1-y.
    The sets {-9..0} and {-8..1} overlap.
    This will increase entropy of x-y compared to H(y)?
    H(x,y) = H(x) + H(y) = 1 + log2(10).
    H(x-y) is bounded by H(x,y).
    
    Let's compute for x uniform on {0,1}, y uniform on 0..9.
    P(x=0, y=j) = 0.05 for j=0..9.
    P(x=1, y=j) = 0.05 for j=0..9.
    
    x-y values:
    x=0: 0, -1, ..., -9.
    x=1: 1, 0, ..., -8.
    
    Counts for x-y:
    1: only from x=1, y=0. Prob 0.05.
    0: from x=0,y=0 and x=1,y=1. Prob 0.10.
    -1: from x=0,y=1 and x=1,y=2. Prob 0.10.
    ...
    -8: from x=0,y=8 and x=1,y=9. Prob 0.10.
    -9: from x=0,y=9. Prob 0.05.
    
    Probs: 0.05, 0.10 (x8), 0.05.
    H(x-y) = -2*0.05*log2(0.05) - 8*0.10*log2(0.10)
    log2(0.05) ~ -4.32. 2*0.05*4.32 = 0.432.
    log2(0.10) ~ -3.32. 8*0.10*3.32 = 2.656.
    H(x-y) ~ 3.088.
    
    H(x) = 1.
    
    x+y values:
    x=0: 0..9.
    x=1: 1..10.
    0: 0.05
    1: 0.10
    ...
    9: 0.10
    10: 0.05
    Same distribution as x-y. H(x+y) = 3.088.
    
    x+2y values:
    x=0: 0, 2, 4, ..., 18.
    x=1: 1, 3, 5, ..., 19.
    All 20 values are distinct!
    Each has prob 0.05.
    H(x+2y) = log2(20) ~ 4.32.
    
    M = max(1, 3.088, 4.32) = 4.32.
    Ratio = 3.088 / 4.32 ~ 0.71.
    
    The term x+2y is the killer. It spreads out too much.
    We need to minimize H(x+2y).
    This suggests choosing x, y such that x+2y takes few values.
    i.e., points close to lines x+2y = k.
    
    If we put mass on a single line x+2y=k, H(x+2y)=0.
    Then M = max(H(x), H(x+y)).
    We want H(x-y) > max(H(x), H(x+y)).
    
    Let's try a segment on x+2y=0. Only (0,0). Ratio undefined (M=0).
    Let's try a segment on x+2y=1. Only (1,0).
    Let's try a segment on x+2y=2. (0,1), (2,0).
    Let's try a larger set where x+2y is constant or takes few values.
    
    Consider the set of points (x,y) such that x+2y is in {0, 1}.
    (0,0) -> 0
    (1,0) -> 1
    (0,1) -> 2 (no)
    
    How about we just use the fast ratio evaluator to search the space of log-probabilities?
    The space is 100D.
    We can use a random walk / hill climbing in the log-prob space.
    
    Strategy:
    1. Start with a few structured candidates (vertical lines, horizontal lines, diagonals).
    2. Use a local search (coordinate ascent or random perturbations) to refine.
    3. Use the fast bincount ratio to evaluate quickly.
    """
    candidates = []
    
    # Uniform
    candidates.append(np.ones((10, 10)) / 100.0)
    
    # Vertical lines (x = c)
    for c in range(10):
        P = np.zeros((10, 10))
        P[:, c] = 1.0 / 10.0
        candidates.append(P)
        
    # Horizontal lines (y = c)
    for c in range(10):
        P = np.zeros((10, 10))
        P[c, :] = 1.0 / 10.0
        candidates.append(P)
        
    # Diagonals x-y = k
    for k in range(-9, 10):
        P = np.zeros((10, 10))
        count = 0
        for y in range(10):
            x = y + k
            if 0 <= x < 10:
                P[y, x] = 1.0
                count += 1
        if count > 0:
            P /= count
            candidates.append(P)
            
    # Anti-diagonals x+y = k
    for k in range(19):
        P = np.zeros((10, 10))
        count = 0
        for y in range(10):
            x = k - y
            if 0 <= x < 10:
                P[y, x] = 1.0
                count += 1
        if count > 0:
            P /= count
            candidates.append(P)
            
    # Lines x+2y = k
    for k in range(28):
        P = np.zeros((10, 10))
        count = 0
        for y in range(10):
            x = k - 2 * y
            if 0 <= x < 10:
                P[y, x] = 1.0
                count += 1
        if count > 0:
            P /= count
            candidates.append(P)
            
    return candidates

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600 - 10
    
    # Get initial candidates
    candidates = _generate_structured_candidates()
    
    best_P = None
    best_score = 0.0
    
    # Evaluate all structured candidates
    for P in candidates:
        score = _fast_ratio(P)
        if score > best_score:
            best_score = score
            best_P = P.copy()
            
    # Local Search / Hill Climbing
    # We will perturb the log-probabilities
    if best_P is None:
        best_P = np.ones((10, 10)) / 100.0
        best_score = _fast_ratio(best_P)
        
    # Work in log space for stability
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    log_P = np.log(best_P + eps)
    
    step_size = 0.1
    n_iter = 0
    max_iter = 200000  # Safety cap, time check is primary
    
    while time.time() < deadline and n_iter < max_iter:
        n_iter += 1
        
        # Random perturbation
        # Choose a few random cells to perturb
        n_perturb = np.random.randint(1, 5)
        indices = np.random.choice(100, n_perturb, replace=False)
        
        old_log_P = log_P.copy()
        
        # Perturb
        for idx in indices:
            log_P.flat[idx] += np.random.randn() * step_size
            
        # Convert to probabilities
        P_new = np.exp(log_P)
        P_new -= P_new.min() # Ensure non-negative
        total = P_new.sum()
        if total <= 0:
            log_P = old_log_P
            continue
        P_new /= total
        
        score_new = _fast_ratio(P_new)
        
        if score_new > best_score:
            best_score = score_new
            best_P = P_new.copy()
            # Reduce step size slightly on success? Or keep.
            # step_size *= 0.99
        else:
            # Revert
            log_P = old_log_P
            # Increase step size if stuck?
            # For now, keep simple.
            
        # Occasional larger jump
        if n_iter % 100 == 0:
            step_size = np.random.uniform(0.05, 0.5)
            
    return best_P

if __name__ == "__main__":
    P = arithmetic_kakeya_distribution()
    print(_fast_ratio(P))
