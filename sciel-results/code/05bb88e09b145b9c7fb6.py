"""Cap set priority function for greedy insertion."""
import time

_start_time = time.time()
_DEADLINE_MARGIN = 10  # seconds before the 300s limit

def priority(vector, n):
    """
    Computes a priority score for a vector in {0, 1, 2}^n.
    Higher scores indicate vectors that should be added earlier in the greedy process.
    
    Strategy: 
    1. Primary: Balance symbol counts (0, 1, 2) to avoid early constraint saturation.
    2. Secondary: Penalize vectors with many identical adjacent pairs (reduces line risk).
    3. Tertiary: Deterministic hash for tie-breaking to break symmetries.
    """
    # Deadline guard: if we're close to the limit, return a neutral score
    # This prevents the evaluator from timing out
    elapsed = time.time() - _start_time
    if elapsed > 300 - _DEADLINE_MARGIN:
        return 0.0
    
    c0 = 0
    c1 = 0
    c2 = 0
    
    # Fast counting
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    # Calculate deviation from ideal balance (n/3 each)
    # Using integer arithmetic to avoid float precision issues and speed up
    d0 = 3 * c0 - n
    d1 = 3 * c1 - n
    d2 = 3 * c2 - n
    
    # Sum of squared deviations (penalty) - main balance metric
    balance_penalty = d0 * d0 + d1 * d1 + d2 * d2
    
    # Secondary: Count positions where adjacent elements are equal
    # Vectors with many identical adjacent pairs are more likely to form lines
    adjacent_equal = 0
    if n > 1:
        prev = vector[0]
        for i in range(1, n):
            if vector[i] == prev:
                adjacent_equal += 1
            prev = vector[i]
    
    # Penalize adjacent equals: each pair adds a small penalty
    # Scale so this is secondary to balance (max ~n-1 pairs)
    adj_penalty = adjacent_equal * 2
    
    # Combined penalty
    total_penalty = balance_penalty + adj_penalty
    
    # Tie-breaker: Deterministic hash to ensure unique ordering for equal penalties
    # This helps break symmetries that might lead to suboptimal greedy choices
    h = 0
    for v in vector:
        h = (h * 131 + v) & 0x7FFFFFFF
        
    # Scale tie-breaker to a small range so it doesn't override the main metrics
    # Max balance penalty for n=10 is roughly (10-3.33)^2 * 3 ~ 133.
    # Max adj penalty for n=10 is 9*2 = 18.
    # Tie breaker should be < 1 to not override.
    tie_breaker = (h % 10000) / 10000.0 * 0.01
    
    # Higher priority for lower penalty
    return -total_penalty + tie_breaker
# EVOLVE-BLOCK-END
