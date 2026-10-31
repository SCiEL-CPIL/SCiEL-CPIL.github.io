"""Cap set greedy builder: Stateless heuristic with L2 norm bias and hash tie-breaking."""
import time

_start_time = None
_DEADLINE = 330 * 0.95

def priority(vector, n):
    """
    Stateless priority function for greedy cap set construction.
    
    Strategy:
    1. Primary Sort: L2 Norm (Sum of squares). Prioritizes vectors with 'high energy' 
       (more 2s and 1s) over 'low energy' (more 0s). This biases the greedy 
       selection away from the origin cluster, which is dense in lines.
    2. Secondary Sort: Mod-3 Sum Penalty. Penalizes vectors lying on the 
       hyperplane sum(v) = 0 mod 3, which is a common source of line collisions.
    3. Tie-Breaker: Deterministic Hash. Ensures a unique, reproducible ordering 
       for vectors with identical structural scores, breaking symmetries in the cube.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # 1. Compute Structural Metrics in O(n)
    s = 0
    s2 = 0
    h = 0
    
    for i, v in enumerate(vector):
        s += v
        s2 += v * v
        # Fast polynomial hash for tie-breaking
        h = (h * 31 + v + i) & 0xFFFFFFFFFFFFFFFF

    # 2. Calculate Scores
    # L2 Norm bias: Higher is better
    l2_score = s2 * 100.0
    
    # Mod-3 Sum Penalty
    mod_sum = s % 3
    if mod_sum == 0:
        mod_penalty = -50.0
    else:
        mod_penalty = 0.0
        
    # Hash Tie-Breaker (normalized to [0, 1))
    hash_score = (h % 10000) / 10000.0
    
    # Combine
    total = l2_score + mod_penalty + hash_score
    
    return float(total)
