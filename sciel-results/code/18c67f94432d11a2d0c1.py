import time
import math

_start_time = time.time()
_DEADLINE = _start_time + 280.0

# Precompute powers of 2 for base-3 encoding (up to reasonable n)
# We'll compute on the fly but cache per n
_powers_cache = {}

def _get_powers(n):
    if n not in _powers_cache:
        _powers_cache[n] = [3**i for i in range(n)]
    return _powers_cache[n]

# Precompute a "structural" score for each possible vector? No, 3^n is too large.
# Instead, we use a modular arithmetic based heuristic.
# Key insight: In a cap set, for any 3 distinct points x,y,z, we cannot have x+y+z = 0 mod 3.
# This is equivalent to saying that for any two points x,y in the set, the point z = -x-y mod 3
# cannot be in the set.
#
# A static priority function cannot know the set. However, we can prioritize vectors
# that are "harder" to combine with others to form lines.
#
# Alternative structural approach: Use the **sum of coordinates** and **sum of squares** mod 3.
# Or better: Use a **quadratic form** that correlates with cap-set structure.
#
# Let's try a completely different family: **Randomized Hash with Structural Bias**
# We combine:
# 1. A strong deterministic hash (for diversity)
# 2. A structural penalty based on "line-completion likelihood"
#    For a vector v, how many pairs (a,b) exist such that a+b+v = 0 mod 3?
#    This is equivalent to: for each coordinate, the value v[i] determines which
#    pairs of values (a[i], b[i]) would complete a line.
#    Specifically, if v[i] = 0, then a[i]+b[i] must be 0 mod 3 -> (0,0) or (1,2) or (2,1)
#    If v[i] = 1, then a[i]+b[i] must be 2 mod 3 -> (0,2) or (2,0) or (1,1)
#    If v[i] = 2, then a[i]+b[i] must be 1 mod 3 -> (0,1) or (1,0) or (2,2)
#    In all cases, there are exactly 3 pairs per coordinate.
#    So this doesn't differentiate vectors.
#
# Let's try: **Distance from the "center" in the group algebra sense**
# Map {0,1,2}^n to Z_3^n. Compute the "weight" as the sum of coordinates.
# Vectors with sum mod 3 == 0 are more likely to be part of lines? Not necessarily.
#
# New idea: **Use the Hamming weight distribution relative to the all-ones vector**
# Or: **Use a quadratic form Q(v) = sum_{i<j} v[i]*v[j] mod 3**
# This captures pairwise interactions.
#
# Let's implement:
# score = -Q(v) + hash_breaker
# where Q(v) = sum_{i<j} v[i]*v[j] mod 3, mapped to a real penalty.

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    # Compute quadratic form: sum_{i<j} v[i]*v[j]
    # This is O(n^2), which might be slow for large n.
    # Optimization: Q(v) = ( (sum v)^2 - sum v^2 ) / 2
    # In mod 3, division by 2 is multiplication by 2 (since 2*2=4=1 mod 3)
    # So Q_mod3 = 2 * ( (sum_v^2 - sum_v2) mod 3 ) mod 3
    
    sum_v = 0
    sum_v2 = 0
    for v in vector:
        sum_v += v
        sum_v2 += v * v
    
    sum_v_mod3 = sum_v % 3
    sum_v2_mod3 = sum_v2 % 3
    
    # Q_mod3 = 2 * (sum_v_mod3^2 - sum_v2_mod3) % 3
    q_val = (2 * (sum_v_mod3 * sum_v_mod3 - sum_v2_mod3)) % 3
    
    # We want to MINIMIZE the quadratic form? Or maximize?
    # Let's try: vectors with q_val == 0 are "neutral", q_val == 1 or 2 are "structured".
    # Let's penalize q_val > 0.
    
    penalty = q_val * 100.0
    
    # Also add a balance penalty
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    ideal = n / 3.0
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Deterministic hash for tie-breaking
    h = 0
    for v in vector:
        h = (h * 31 + v) & 0xFFFFFFFF
    
    score = -deviation * 50.0 - penalty + h / 4294967296.0
    
    return float(score)
