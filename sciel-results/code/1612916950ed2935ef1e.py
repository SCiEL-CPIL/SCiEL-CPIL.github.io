import numpy as np
import time

# Global timer
_start_time = time.time()
_TIMEOUT = 140.0 # Safety margin before 150s limit

def _solve_dp(w, v, C, scale):
    """
    Solves 0/1 knapsack exactly using DP on scaled integer weights.
    Returns (list of indices in optimal solution, total value).
    """
    n = len(w)
    if n == 0:
        return [], 0.0
        
    cap_int = int(round(C * scale))
    if cap_int < 0:
        return [], 0.0

    # dp[i][c] = max value using first i items with capacity c
    # Using numpy for the inner loop might be faster, but standard Python loop is okay for 125k ops.
    # Let's try to optimize the inner loop if possible, but 125k is small.
    
    dp = np.zeros((n + 1, cap_int + 1), dtype=np.float64)
    
    for i in range(1, n + 1):
        wi = int(round(w[i-1] * scale))
        vi = v[i-1]
        # If item weight is 0, it always fits (if value > 0).
        # If wi > cap_int, it can never fit.
        
        # Copy previous row
        # dp[i, :] = dp[i-1, :]
        # Then update for taking item i-1
        
        if wi <= cap_int:
            # For c from wi to cap_int:
            # dp[i, c] = max(dp[i-1, c], dp[i-1, c-wi] + vi)
            # We can do this with vectorization for speed
            
            prev_row = dp[i-1, :]
            take_row = prev_row[:cap_int - wi + 1] + vi
            
            # Initialize with prev row
            dp[i, :] = prev_row
            
            # Update where c >= wi
            # Indices wi to cap_int
            # Compare prev_row[wi:] with take_row
            # take_row corresponds to c=wi...cap_int
            
            # Let's just do a simple loop for clarity and correctness, 125k is fast.
            # But vectorized is better.
            # dp[i, wi:] = np.maximum(prev_row[wi:], take_row)
            
            dp[i, wi:] = np.maximum(prev_row[wi:], take_row)
        else:
            # Item too heavy, copy previous
            dp[i, :] = dp[i-1, :]

    # Backtrack
    selected = []
    c = cap_int
    for i in range(n, 0, -1):
        # Check if item i-1 was taken
        # If dp[i, c] != dp[i-1, c], it was taken.
        # Note: Floating point comparison.
        if dp[i, c] > dp[i-1, c] + 1e-9:
            selected.append(i - 1)
            wi = int(round(w[i-1] * scale))
            c -= wi
            if c < 0:
                # Should not happen if DP is correct
                break
        # else: not taken

    selected.reverse()
    return selected, dp[n, cap_int]

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Fallback: Simple greedy by density
        if len(remaining_weights) == 0:
            return -1
        w = np.array(remaining_weights)
        v = np.array(remaining_values)
        feasible = np.where(w <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v[feasible] / w[feasible]
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0:
        return -1
        
    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)
    C = remaining_capacity
    
    # Filter feasible items first to reduce work
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1
        
    w_f = w[feasible_mask]
    v_f = v[feasible_mask]
    idx_f = np.where(feasible_mask)[0]
    
    if len(w_f) == 0:
        return -1
        
    # Lagrangian Relaxation Heuristic
    # We want to find lambda such that sum(w_i * 1[v_i - lambda*w_i > 0]) approx C
    # Then pick item with max v_i - lambda*w_i > 0
    
    # Binary search for lambda
    # Lower bound: 0
    # Upper bound: max density + epsilon
    
    max_density = np.max(v_f / w_f) if np.any(w_f > 0) else 1.0
    if not np.isfinite(max_density):
        max_density = 1.0
        
    lo = 0.0
    hi = max_density * 1.1 + 1e-6
    
    # If all values are 0, return -1 or any item
    if np.max(v_f) <= 0:
        return -1

    best_lambda = 0.0
    best_score = -np.inf
    best_idx_rel = -1
    
    # We can just evaluate a few candidate lambdas or do binary search
    # Given small N, we can afford a precise binary search
    
    # However, we need to be careful: the set of items included changes discontinuously.
    # We want a lambda that "tightens" the knapsack.
    
    # Let's try to find the lambda where the total weight of "accepted" items is closest to C from below.
    # Or simply, iterate a few steps of binary search.
    
    for _ in range(20):
        mid = (lo + hi) / 2.0
        scores = v_f - mid * w_f
        # Items with positive score are "accepted" in the relaxation
        accepted = scores > 0
        total_w = np.sum(w_f[accepted])
        
        if total_w > C:
            # Lambda is too low, need higher lambda to reduce accepted weight
            lo = mid
        else:
            # Lambda is high enough (or too high)
            # We prefer a lambda that fills the capacity more, so if total_w <= C, this is a valid candidate
            # We want to maximize the minimum score? No, we want to pick the best item.
            # The "best" lambda is one that leaves little slack or where the marginal item fits.
            
            # Let's store this as a candidate
            # We want to pick the item with max score among accepted items.
            # If no items accepted, it's not useful.
            if np.any(accepted):
                max_score_in_accepted = np.max(scores[accepted])
                # We want to balance total_w close to C.
                # If total_w is much smaller than C, maybe we can lower lambda to include more/better items?
                # Actually, for a single pick, we just want the item with highest (v - lambda w).
                # The choice of lambda affects the ranking.
                # A common heuristic is to use the lambda from the fractional knapsack solution.
                
                # Let's keep the lambda that maximizes the "tightness" or just use the one from the last step where total_w <= C.
                # Actually, if we use the lambda where total_w <= C, and we pick the max score item,
                # we are essentially picking the item that is "most worth it" given the shadow price.
                
                # To improve, we can check if lowering lambda slightly (to include more items) yields a better top item?
                # No, if we lower lambda, scores increase for heavy items.
                
                # Let's just use the current mid if it's feasible (total_w <= C).
                # We want to find the lambda such that the knapsack is "full".
                # If total_w < C, we might be able to lower lambda to include more items, potentially finding a better one?
                # Not necessarily. The best item for the *next* step is the one with highest marginal value.
                
                # Let's try to converge to the lambda where the fractional knapsack is optimal.
                # This lambda is the density of the item that would be partially filled.
                
                # For now, let's just pick the best item under this lambda.
                # We will refine the lambda selection in the next iteration.
                hi = mid # Try to increase lambda to see if we can tighten? No, if total_w <= C, we can afford to lower lambda to include more.
                # Wait. If total_w <= C, we have slack. Lowering lambda increases scores of heavy items.
                # This might bring a heavy, high-value item into the "accepted" set.
                # So we should move lo up? No, lo is the lower bound.
                # If total_w <= C, the constraint is not tight. We can lower lambda (move hi down? No, hi is upper bound).
                # We want to find the lambda where total_w is close to C.
                # If total_w < C, we need to *decrease* lambda to include more items (increase total_w).
                # So hi = mid.
                
                # If total_w > C, we need to *increase* lambda to exclude items (decrease total_w).
                # So lo = mid.
                
                # This converges to the lambda where total_w crosses C.
                # Let's record the best candidate found so far that is feasible.
                
                # Actually, we just need *a* good lambda.
                # Let's just use the lambda from the binary search that keeps total_w <= C.
                
                # To be safe, let's just pick the item with max (v - mid*w) among those with positive score.
                # If none, break.
                
                idxs_accepted = np.where(accepted)[0]
                if len(idxs_accepted) > 0:
                    best_score_candidate = np.max(scores[idxs_accepted])
                    # We want to maximize the score?
                    # The score v - lambda w is the "profit" in the Lagrangian.
                    # The item with the highest profit is the one we should pick.
                    
                    # We will store the best candidate (lambda, item) found.
                    # How to define "best"?
                    # Usually, the lambda from the fractional solution is the best guide.
                    # So we want the lambda where the fractional knapsack is tight.
                    
                    # Let's just update the "current best" if this lambda is feasible.
                    # We'll keep the one that maximizes the top item's score?
                    # Or just the one closest to filling the capacity?
                    
                    # Let's just pick the item corresponding to the current mid if it's feasible.
                    # We'll refine by checking the neighbors if time permits, but 20 iterations is enough to get close.
                    
                    # For the purpose of this step, let's just take the best item under the current mid.
                    # We will return the result from the last feasible mid.
                    
                    # To ensure we don't miss a better item from a slightly different lambda,
                    # we can collect a few candidates.
                    pass
            
            # Logic for binary search direction:
            # If total_w > C, lambda is too low -> increase lambda (lo = mid)
            # If total_w <= C, lambda is high enough -> we can try lower lambda to include more?
            # If we lower lambda, we include more items. The "best" item might change.
            # However, the standard Lagrangian dual variable is the one where the constraint is tight.
            # So if total_w <= C, we should try to *decrease* lambda to fill the capacity?
            # Yes. So hi = mid.
            
            # Wait, if total_w <= C, and we set hi = mid, we are searching for a smaller lambda.
            # Smaller lambda -> higher scores for heavy items.
            # This seems correct to find the "critical" lambda.

    # After binary search, we have a range [lo, hi].
    # The lambda at the boundary is the most informative.
    # Let's evaluate the top few items for lambda = lo, hi, and maybe (lo+hi)/2.
    
    candidates = []
    for lam in [lo, hi, (lo+hi)/2.0]:
        scores = v_f - lam * w_f
        # Only consider items that fit and have positive score
        valid = (w_f <= C) & (scores > 0)
        if np.any(valid):
            idxs = np.where(valid)[0]
            max_idx_rel = idxs[np.argmax(scores[idxs])]
            candidates.append((scores[max_idx_rel], max_idx_rel))
    
    if candidates:
        # Pick the candidate with the highest score
        # But wait, scores depend on lambda. Comparing scores from different lambdas is tricky.
        # However, if we pick the item with the highest "Lagrangian value" under the optimal lambda, it should be good.
        # Let's just pick the one with the highest score among the candidates.
        # Or better, just pick the item that is best under the lambda that filled the knapsack most tightly.
        
        # Let's refine: Pick the item that maximizes v_i - lambda*w_i for the lambda that minimizes |sum(w) - C|.
        
        # Simple approach: Just use the lambda from the last iteration of binary search that was feasible.
        # Let's just use the result from the last 'hi' or 'lo' that was feasible.
        
        # Actually, the most robust simple heuristic is:
        # 1. Find lambda* such that sum(w_i | v_i - lambda* w_i > 0) <= C is maximized (closest to C).
        # 2. Pick argmax (v_i - lambda* w_i) among those items.
        
        # My binary search above converges to the crossing point.
        # Let's just use the lambda = (lo+hi)/2.
        
        lam_final = (lo + hi) / 2.0
        scores = v_f - lam_final * w_f
        valid = (w_f <= C) & (scores > 0)
        
        if np.any(valid):
            idxs = np.where(valid)[0]
            best_rel = idxs[np.argmax(scores[idxs])]
            return int(idx_f[best_rel])
            
    # Fallback if Lagrangian fails (e.g. all scores negative or no feasible items)
    # Use Density
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v_f / w_f
    densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
    
    # Only pick if density is positive?
    if np.max(densities) <= 0:
        return -1
        
    best_local = np.argmax(densities)
    return int(idx_f[best_local])
