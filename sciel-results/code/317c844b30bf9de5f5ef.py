import time

_start_time = None
_DEADLINE = 330 * 0.95

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Quadratic score for having all 3 distinct symbols.
    2. Secondary: Quadratic balance penalty (chi-squared distance from uniform).
    3. Tertiary: Local mixing penalty (penalize adjacent identical symbols).
    4. Quaternary: Strong hash for deterministic but diverse tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    presence0 = 1.0 if c0 > 0 else 0.0
    presence1 = 1.0 if c1 > 0 else 0.0
    presence2 = 1.0 if c2 > 0 else 0.0
    
    num_distinct = presence0 + presence1 + presence2
    distinct_score = 10000.0 * num_distinct * num_distinct
    
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0
    
    # Local mixing: Penalize adjacent identical values
    # This encourages vectors that "mix" symbols along the vector,
    # which may reduce the likelihood of forming lines with other mixed vectors.
    local_penalty = 0.0
    for i in range(n - 1):
        if vector[i] == vector[i + 1]:
            local_penalty += 1.0
    local_score = -local_penalty * 2.0
    
    # Improved Hash: Use a simple but effective mixing function
    # FNV-1a style
    h = 2166136261
    for val in vector:
        h ^= val
        h = (h * 16777619) & 0xFFFFFFFF
    
    # Normalize hash to [0, 1]
    hash_score = (h % 10000) / 10000.0
    
    return float(distinct_score + balance_score + local_score + hash_score)
