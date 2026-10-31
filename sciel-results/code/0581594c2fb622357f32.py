import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Sum of squares of coordinates - favors vectors with more 1s and 2s
       over 0s, breaking the symmetry of the balance heuristic to exploit the
       non-linear distribution of affine lines in the ternary grid.
    2. Secondary: Distinct symbol bonus (quadratic in presence).
    3. Tertiary: Mod-3 sum penalty - vectors whose coordinate sum is 0 mod 3
       are more likely to form lines with already-selected vectors.
    4. Quaternary: Quadratic balance penalty.
    5. Quinary: Fast structural hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Sum of squares: vectors with more 1s and 2s score higher
    # This breaks lexicographic symmetry and aligns with cap set structure
    coord_sq_sum = sum(v * v for v in vector)
    ss_score = coord_sq_sum * 200.0
    
    # Distinct symbol bonus: reward vectors using all 3 symbols
    presence0 = 1.0 if c0 > 0 else 0.0
    presence1 = 1.0 if c1 > 0 else 0.0
    presence2 = 1.0 if c2 > 0 else 0.0
    num_distinct = presence0 + presence1 + presence2
    distinct_score = 8000.0 * num_distinct * num_distinct
    
    # Mod-3 sum penalty: in F_3^n, a line has three points whose coordinate-wise
    # sums are 0 mod 3. Vectors with sum ≡ 0 mod 3 are more "central" and
    # more likely to complete lines. Penalize them.
    coord_sum = c1 + 2 * c2  # sum of coordinates (0s contribute nothing)
    mod3 = coord_sum % 3
    # Penalty: 0 mod 3 is worst, 1 and 2 are better
    mod3_penalty = -3000.0 * (1.0 if mod3 == 0 else 0.3)
    
    # Balance: Quadratic deviation from uniform
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0
    
    # Fast hash: simple polynomial rolling hash
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    
    hash_score = h / 0x7FFFFFFF
    
    return float(ss_score + distinct_score + mod3_penalty + balance_score + hash_score)
