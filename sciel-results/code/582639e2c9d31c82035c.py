import numpy as np
import time

_START = None
_CALL_COUNT = 0
_DEADLINE_MARGIN = 5.0  # seconds before hard limit to fall back

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time()

def _fast_density_index(weights, values, cap):
    """Fallback: pick max density among feasible items."""
    if len(weights) == 0:
        return -1
    feasible = weights <= cap
    if not feasible.any():
        return -1
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(weights > 0, values / weights, 0.0)
    fd = np.where(feasible, density, -np.inf)
    return int(np.argmax(fd))

def _scaled_dp_select(capacity, weights, values):
    """
    Uses value scaling to perform a DP approximation.
    Returns the index of the best first item to pick.
    """
    n = len(weights)
    if n == 0:
        return -1
    
    # Filter feasible items to reduce DP size if possible, but we need original indices
    feasible_mask = weights <= capacity
    if not feasible_mask.any():
        return -1
    
    # To keep DP fast, we scale values.
    # We want the DP table size to be manageable.
    # Let max_val = max(values).
    # Scale factor S such that max_scaled_val is small (e.g., 500).
    # S = max_val / 500.
    # Scaled value v_i' = round(v_i / S) = round(v_i * 500 / max_val).
    
    max_val = np.max(values)
    if max_val == 0:
        return int(np.argmax(feasible_mask))
        
    target_scaled_max = 500
    scale = max_val / target_scaled_max
    scaled_values = np.round(values / scale).astype(int)
    max_scaled = int(np.max(scaled_values))
    
    # If max_scaled is 0, all values are tiny, just pick max value
    if max_scaled == 0:
         fv = np.where(feasible_mask, values, -np.inf)
         return int(np.argmax(fv))

    # DP Table: dp[c] = max scaled value achievable with capacity c
    # Capacity is continuous, but weights are floats.
    # This is tricky. Standard DP requires integer capacities.
    # Since capacity is 12.5 and weights are likely small floats, we can scale capacity too?
    # Or we can use a dictionary-based DP for sparse states?
    # Given N=100, and capacity 12.5, if weights are ~0.1-1.0, there aren't many distinct capacity states.
    # However, float precision issues make exact DP hard.
    
    # Alternative: Approximate by discretizing capacity.
    # Let's try a different scaling: Scale weights to integers?
    # If we scale weights to integers, we lose precision.
    
    # Let's stick to the "Value Scaling" idea but applied to a *heuristic* DP?
    # Actually, the prompt suggests: "dynamic programming approximation via value scaling to compute an exact optimal substructure for the remaining capacity... leveraging the pseudo-polynomial nature of the DP approach on quantized values."
    
    # This usually implies:
    # 1. Scale values to integers.
    # 2. Use DP over *value*? No, that's for min weight for max value.
    # 3. Or DP over *capacity*?
    
    # If we do DP over capacity, we need integer capacity.
    # Let's assume weights can be scaled to integers reasonably.
    # If weights are e.g. [0.1, 0.2, ...], we can multiply by 10 or 100.
    # Let's find a common multiplier for weights to make them integers?
    # That's risky if weights are irrational.
    
    # Better approach for "Value Scaling DP" in Knapsack context often refers to:
    # Maximizing value subject to weight <= C.
    # Approximate by scaling values down.
    # But the state space is still Capacity.
    
    # Let's try a different interpretation:
    # Use the "Approximation Algorithm" for Knapsack:
    # 1. Scale values by factor.
    # 2. Run DP on scaled values? No, DP is on Capacity.
    
    # Let's try a **Grid Search / DP on discretized capacity**.
    # Discretize capacity into M bins.
    # M = 100.
    # dp[j] = max value for capacity bin j.
    # This is O(N*M). With N=100, M=100, it's 10,000 ops. Very fast.
    
    M = 100
    cap_max = capacity
    
    # Map capacity [0, cap_max] to indices [0, M]
    # bin_width = cap_max / M
    
    # Initialize dp with -inf
    dp = np.full(M + 1, -np.inf)
    dp[0] = 0.0
    
    # We need to track the *path* or at least the *first item* chosen in the optimal solution.
    # Standard DP doesn't easily give the "first item" if we process items one by one.
    # We can do a "Forward DP" where we decide the *next* item.
    # But that's exponential if we don't aggregate.
    
    # Alternative: 
    # Compute the optimal value V* for the full set using discretized DP.
    # Then, for each candidate first item i, compute the optimal value of the *remaining* set with capacity C - w_i.
    # Pick i that maximizes v_i + V_rem(C - w_i).
    # This requires running DP for each candidate i.
    # N candidates * O(N*M) = O(N^2 * M) = 100 * 100 * 100 = 1,000,000 ops.
    # This is feasible within 150s? Yes, 1M ops is milliseconds in NumPy/Python if optimized, or even pure Python loops might be okay if we are careful, but NumPy is better.
    
    # Let's implement this "Look-ahead DP" strategy.
    
    def compute_dp_value(rem_cap, rem_weights, rem_values):
        """
        Computes max value using discretized DP.
        rem_cap: float
        rem_weights: array of floats
        rem_values: array of floats
        Returns: max value (float)
        """
        if rem_cap <= 0 or len(rem_weights) == 0:
            return 0.0
        
        # Discretize capacity
        # Use a fixed number of bins for the *remaining* capacity?
        # Or global bins?
        # Using global bins [0, original_capacity] is safer for consistency.
        # But we are passing rem_cap which is smaller.
        # Let's use bins relative to rem_cap for better resolution?
        # Or just use the same M bins for the range [0, original_capacity]?
        # If we use [0, original_capacity], the resolution is fixed.
        
        # Let's use a local discretization for better resolution on the smaller capacity.
        local_M = 50
        if rem_cap <= 0:
            return 0.0
        
        # dp[j] = max value for capacity <= j * (rem_cap / local_M)
        # Actually, let's just use a dictionary or array of size local_M+1
        # dp[j] represents max value for capacity <= j * step
        step = rem_cap / local_M
        
        dp = np.full(local_M + 1, -np.inf)
        dp[0] = 0.0
        
        # Process each item
        for w, v in zip(rem_weights, rem_values):
            if w > rem_cap:
                continue
            # Update dp in reverse
            # For each bin j, if we take this item, we need capacity j*step - w
            # We need to find the largest k such that k*step <= j*step - w
            # k <= j - w/step
            # k = floor(j - w/step)
            
            # This is O(M) per item. Total O(N*M).
            # We can vectorize this?
            # For a given item w, the new value for bin j is dp[j] vs dp[k] + v where k = floor(j - w/step).
            
            # Let's do it with a loop for clarity, it's small (50*100=5000)
            # But we are doing this for N candidates? 100 * 5000 = 500,000.
            # Python loop of 500k is ~0.1s. Acceptable.
            
            # To be safe and fast, let's use NumPy if possible, but the dependency on j makes it tricky.
            # Let's stick to Python loop for the inner DP update, it's small enough.
            
            # Actually, we can update dp in place from high to low.
            # new_dp[j] = max(dp[j], dp[k] + v)
            # where k = max(0, int(j - w/step))
            
            # We must be careful: dp[k] should be the value *before* adding this item.
            # So we need a copy or iterate backwards.
            
            old_dp = dp.copy()
            for j in range(local_M, -1, -1):
                # Capacity available for previous items: j*step - w
                # If j*step - w < 0, we can't take it (but we checked w <= rem_cap, so for high j it's fine)
                if j * step < w:
                    continue
                k = int((j * step - w) / step)
                # k might be slightly off due to float, clamp
                if k > local_M:
                    k = local_M
                if k < 0:
                    k = 0
                # old_dp[k] is the best value for capacity <= k*step
                # Is it correct to use k = floor((j*step - w)/step)?
                # Yes, because we want the largest bin index that fits.
                val = old_dp[k] + v
                if val > dp[j]:
                    dp[j] = val
                    
        return float(dp[local_M])

    # Main Logic:
    # 1. Compute V_total = compute_dp_value(capacity, weights, values)
    # 2. For each feasible item i:
    #    a. rem_cap = capacity - weights[i]
    #    b. rem_weights = weights excluding i
    #    c. rem_values = values excluding i
    #    d. V_rem = compute_dp_value(rem_cap, rem_weights, rem_values)
    #    e. Score = values[i] + V_rem
    # 3. Pick i with max Score.
    
    # Optimization: 
    # If N is large, this is O(N^2 * M).
    # With N=100, M=50, it's 500,000 operations.
    # In Python, 500k iterations with some math is fast (< 0.5s).
    # We have 150s budget. We call this function ~100 times (once per item selection).
    # 100 * 0.5s = 50s. This is tight but should fit.
    # We can reduce M to 20 to be safer?
    # Or reduce N by only considering top K candidates?
    # The DP itself is an approximation. 
    # Let's use M=20 for speed. 100 * 100 * 20 = 200,000 ops per call. 100 calls = 20M ops. ~2-5s. Safe.
    
    local_M = 20
    
    # Pre-compute V_total? Not strictly needed for comparison, but good for sanity.
    # Actually, we just need to compare scores.
    
    best_idx = -1
    best_score = -np.inf
    
    feasible_indices = np.where(weights <= capacity)[0]
    
    # If too many feasible items, maybe limit to top 20 by density to save time?
    # But the DP is the heuristic. Let's try all feasible if N <= 50, else top 20.
    # N is 100. Let's limit candidates to top 20 by density to ensure speed.
    
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(weights > 0, values / weights, 0.0)
    
    # Sort feasible by density
    feas_densities = density[feasible_indices]
    top_k = min(20, len(feasible_indices))
    # Get indices of top K densities within the feasible set
    # argsort on feas_densities
    order = np.argsort(-feas_densities)
    candidate_indices = feasible_indices[order[:top_k]]
    
    for i in candidate_indices:
        w_i = float(weights[i])
        v_i = float(values[i])
        
        rem_cap = capacity - w_i
        if rem_cap < 0:
            continue
            
        # Create remaining arrays
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rem_weights = weights[mask]
        rem_values = values[mask]
        
        # Filter rem arrays to only feasible for rem_cap? 
        # The DP function handles infeasible items by skipping them (w > rem_cap).
        # But passing 99 items is fine.
        
        v_rem = compute_dp_value(rem_cap, rem_weights, rem_values)
        
        score = v_i + v_rem
        
        if score > best_score:
            best_score = score
            best_idx = i
            
        # Time check
        if time.time() - _START > (150.0 - _DEADLINE_MARGIN):
            break
            
    if best_idx < 0:
        best_idx = _fast_density_index(weights, values, capacity)
        
    return int(best_idx)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _CALL_COUNT
    if remaining_capacity is None:
        return -1
    cap = float(remaining_capacity)
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    if n == 0 or cap <= 0:
        return -1
    
    now = _now()
    elapsed = now - _START
    _CALL_COUNT += 1
    
    # Safety check
    if elapsed > (150.0 - _DEADLINE_MARGIN):
        return _fast_density_index(weights, values, cap)
        
    # Use the DP-based heuristic
    # We only use this if we have enough time.
    # DP is O(N^2 * M). With N=100, M=20, it's ~20k ops per candidate? No, M is inner loop.
    # compute_dp_value is O(N_rem * M).
    # We call it for K candidates. K=20.
    # Total ops ~ 20 * (100 * 20) = 40,000 ops.
    # This is very fast.
    
    return _scaled_dp_select(cap, weights, values)
