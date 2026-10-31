"""Cap set priority: Geometric centering with hash-based diversity for greedy selection."""
import time

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 325  # seconds, safety margin before 330s kill

# Precomputed constants for hashing
_MASK = (1 << 64) - 1

def _compute_hash(vector, seed):
    """
    Fast deterministic hash for tie-breaking.
    Returns a float in [0, 1).
    """
    h = seed
    for v in vector:
        # Simple mixing
        h = (h * 1000003 + v * 17 + 11) & _MASK
    return h / _MASK

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Higher priority = selected earlier.
    
    Strategy:
    1. Geometric Centering: Prefer vectors far from the corners (0...0 and 2...2).
       This is measured by the product of distances to these corners.
    2. Balance: Prefer vectors where counts of 0s and 2s are similar.
    3. Hash Diversity: Add a deterministic pseudo-random component to break ties
       and explore different structural configurations, which can lead to larger
       cap sets by avoiding systematic biases in the greedy choice.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -1e18
    
    count_0 = 0
    count_1 = 0
    count_2 = 0
    
    for v in vector:
        if v == 0:
            count_0 += 1
        elif v == 1:
            count_1 += 1
        else:
            count_2 += 1
            
    # 1. Geometric Center Score
    # Distance to all-zeros: sum(v) = count_1 + 2*count_2
    # Distance to all-twos: sum(2-v) = 2*count_0 + count_1
    dist_zero = count_1 + 2 * count_2
    dist_two = 2 * count_0 + count_1
    center_score = dist_zero * dist_two
    
    # 2. Balance Score
    # Penalize imbalance between 0s and 2s
    balance_diff = abs(count_0 - count_2)
    balance_score = -balance_diff * 0.5
    
    # Diversity bonus: reward having all three values
    diversity = 0.0
    if count_0 > 0 and count_1 > 0 and count_2 > 0:
        diversity = 1.0 + min(count_0, count_1, count_2) * 0.1
    
    # 3. Hash-based Diversity
    # Use multiple seeds to create a robust, non-correlated random component
    h1 = _compute_hash(vector, seed=12345)
    h2 = _compute_hash(vector, seed=67890)
    h3 = _compute_hash(vector, seed=13579)
    
    # Combine hash scores
    # Weight them so they don't override the structural score but provide fine-grained ordering
    hash_score = (h1 * 0.5 + h2 * 0.3 + h3 * 0.2)
    
    # Final Combination
    # Center score is dominant (magnitude ~ n^2)
    # Hash score is small (magnitude ~ 1)
    # Balance and diversity are small corrections
    total_priority = center_score + balance_score + diversity + hash_score * 0.8
    
    return total_priority
# EVOLVE-BLOCK-END
