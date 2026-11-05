"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add.
    Returns an index into the remaining (feasible) arrays, or -1 to stop.
    
    Strategy: 1-step look-ahead with exact DP for the subproblem.
    For each feasible candidate, compute the optimal value of completing
    the knapsack after picking that item, using DP. Pick the best candidate.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter to feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 0:
        return -1
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # Scale weights to integers for DP
    # Use a scale factor to convert float weights to integers
    # Capacity is ~12.5, so scale by 100 gives max ~1250
    scale = 100
    cap_int = int(round(remaining_capacity * scale))
    
    # If capacity is too large for DP, fall back to density
    if cap_int > 100000:
        densities = remaining_values / remaining_weights
        return int(np.argmax(densities[feasible_mask]))
    
    # For each feasible candidate, evaluate: value + optimal_subproblem
    # To save time, limit the number of candidates we fully evaluate
    max_candidates = min(len(feasible_indices), 30)
    
    # Sort feasible candidates by density descending, take top max_candidates
    feasible_densities = remaining_values[feasible_mask] / remaining_weights[feasible_mask]
    sorted_local = np.argsort(-feasible_densities)
    top_local = sorted_local[:max_candidates]
    candidates = feasible_indices[top_local]
    
    best_score = -1
    best_idx = int(candidates[0])
    
    # DP function: given list of (weight, value) pairs and capacity, return optimal value
    def dp_optimal(weights_list, values_list, cap):
        """Exact DP for 0/1 knapsack with integer capacity."""
        if cap <= 0 or len(weights_list) == 0:
            return 0.0
        # dp[c] = max value with capacity c
        dp = np.zeros(cap + 1, dtype=np.float64)
        for w, v in zip(weights_list, values_list):
            wi = int(round(w * scale))
            if wi > cap:
                continue
            # Update in reverse
            for c in range(cap, wi - 1, -1):
                candidate_val = dp[c - wi] + v
                if candidate_val > dp[c]:
                    dp[c] = candidate_val
        return dp[cap]
    
    # Precompute scaled weights and values for all remaining items
    all_weights_scaled = (remaining_weights * scale).astype(np.int64)
    
    for cand in candidates:
        cand_weight = int(round(remaining_weights[cand] * scale))
        cand_value = remaining_values[cand]
        remaining_cap = cap_int - cand_weight
        if remaining_cap < 0:
            continue
        
        # Build subproblem: all items except cand
        # Get indices of all other items
        other_mask = np.ones(n, dtype=bool)
        other_mask[cand] = False
        other_indices = np.where(other_mask & feasible_mask)[0]
        
        if len(other_indices) == 0:
            total = cand_value
        else:
            sub_weights = remaining_weights[other_indices]
            sub_values = remaining_values[other_indices]
            total = cand_value + dp_optimal(sub_weights, sub_values, remaining_cap)
        
        if total > best_score:
            best_score = total
            best_idx = int(cand)
    
    return best_idx
# EVOLVE-BLOCK-END
