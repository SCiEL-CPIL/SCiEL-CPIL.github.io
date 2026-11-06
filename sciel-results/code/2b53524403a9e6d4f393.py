"""Cap set priority function using structured heuristics."""
import time
import hashlib

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 300s kill

def _get_priority(vector, n):
    """
    Compute priority for a vector in {0,1,2}^n.
    
    Strategy: Use multiple structural features to order vectors
    so that greedy insertion yields a large cap set.
    """
    # Feature 1: Sum of coordinates mod 3
    s = sum(vector)
    sum_mod3 = s % 3
    
    # Feature 2: Count of 1s (middle values in 3-APs)
    count_1 = sum(1 for v in vector if v == 1)
    
    # Feature 3: Count of 0s and 2s
    count_0 = sum(1 for v in vector if v == 0)
    count_2 = sum(1 for v in vector if v == 2)
    
    # Feature 4: Weighted sum with alternating signs to break symmetry
    weighted = 0
    for i, v in enumerate(vector):
        # Use a pattern that avoids 3-AP structures
        weight = 1 if i % 3 == 0 else (-1 if i % 3 == 1 else 0)
        weighted += weight * v
    
    # Feature 5: Hash-based tiebreaker for determinism
    h = hashlib.md5(str(vector).encode()).hexdigest()
    hash_val = int(h[:8], 16) / 0xFFFFFFFF
    
    # Combine features:
    # - Prefer vectors with sum_mod3 == 0 (one coset tends to be cap-set-like)
    # - Prefer vectors with extreme count_1 (few or many 1s)
    # - Use weighted sum as secondary
    # - Use hash as final tiebreaker
    
    # Score components (higher is better for priority)
    # Coset bonus: sum_mod3 == 0 gets highest
    coset_score = [2, 0, 1][sum_mod3]  # sum_mod3=0 -> 2, 1 -> 0, 2 -> 1
    
    # Middle value penalty: vectors with count_1 close to n/3 are penalized
    # (they're more likely to be middle elements of 3-APs)
    mid_dev = abs(count_1 - n / 3.0)
    mid_score = mid_dev  # Higher deviation from n/3 is better
    
    # Weighted structure score
    weight_score = weighted
    
    # Normalize and combine
    # Primary: coset (most important structural feature)
    # Secondary: middle value deviation
    # Tertiary: weighted sum
    # Quaternary: hash for deterministic ordering
    
    priority = (
        coset_score * 10000 +
        mid_score * 100 +
        weight_score * 10 +
        hash_val
    )
    
    return float(priority)

def priority(vector, n):
    """
    Priority function for cap set greedy construction.
    
    Args:
        vector: tuple of values in {0, 1, 2}
        n: length of the vector
    
    Returns:
        float: priority value (higher = added earlier)
    """
    # Time guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Return a default priority to avoid hanging
        return 0.0
    
    try:
        return _get_priority(vector, n)
    except Exception:
        return 0.0
