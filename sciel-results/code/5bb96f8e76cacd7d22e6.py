"""Cap set greedy construction with heuristic priority."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _is_cap_set(vecs, n):
    """Check if a set of vectors in {0,1,2}^n is a cap set (no 3-term AP)."""
    vecs = list(vecs)
    m = len(vecs)
    if m < 3:
        return True
    # For each pair, check if the third point of the AP is in the set
    # AP: a, b, c where a + c = 2b (mod 3)
    # So for each pair (a, c), b = (a + c) / 2 mod 3
    # Actually, in Z_3^n, a 3-term AP is a, a+d, a+2d
    # Equivalently: x, y, z are in AP iff x + z = 2y mod 3
    # So for each y in the set, and each x in the set, z = 2y - x mod 3 must not be in the set
    # More efficiently: for each pair (x, z), check if (x+z)*inv2 mod 3 is in the set
    # inv2 mod 3 = 2 since 2*2 = 4 = 1 mod 3
    vec_set = set(vecs)
    for i in range(m):
        for j in range(i + 1, m):
            x = vecs[i]
            z = vecs[j]
            # y = (x + z) * 2 mod 3
            y = tuple((xi + zi) * 2 % 3 for xi, zi in zip(x, z))
            if y in vec_set and y != x and y != z:
                return False
    return True

def _is_cap_set_fast(vecs, n):
    """Faster cap set check using a set for O(1) lookups."""
    vec_set = set(vecs)
    m = len(vecs)
    for i in range(m):
        for j in range(i + 1, m):
            x = vecs[i]
            z = vecs[j]
            y = tuple((xi + zi) * 2 % 3 for xi, zi in zip(x, z))
            if y in vec_set and y != x and y != z:
                return False
    return True

def _can_add(vec, current_set, n):
    """Check if adding vec to current_set keeps it a cap set."""
    # For each existing vector v in current_set, check if {vec, v, w} forms an AP for some w
    # AP condition: for any three distinct points a, b, c in the set, a + c != 2b mod 3
    # When adding vec, we need to check all pairs (vec, v) for v in current_set:
    #   - Is there w in current_set such that vec + w = 2v or v + w = 2*vec or vec + v = 2w?
    # More simply: for each v in current_set, compute the "third point" that would form an AP
    # with vec and v, and check if it's in current_set.
    # APs involving vec:
    # 1) vec, v, w where vec + w = 2v => w = 2v - vec
    # 2) v, vec, w where v + w = 2*vec => w = 2*vec - v
    # 3) vec, w, v where vec + v = 2w => w = (vec + v) * 2
    # But cases 1 and 3 are symmetric. Let's just check:
    # For each v in current_set:
    #   w1 = tuple((2*vi - veci) % 3 for vi, veci in zip(v, vec))  # vec + w1 = 2v
    #   w2 = tuple((2*veci - vi) % 3 for vi, veci in zip(v, vec))  # v + w2 = 2*vec
    #   w3 = tuple((veci + vi) * 2 % 3 for vi, veci in zip(v, vec)) # vec + v = 2*w3
    # If any of w1, w2, w3 is in current_set (and distinct from vec and v), we have a conflict.
    # Actually w1, w2, w3 are all the same computation viewed differently. Let me re-derive.
    # A 3-term AP in Z_3^n: points a, b, c with b = (a+c)/2 = 2*(a+c) mod 3.
    # So for the new point vec and existing points v1, v2, we need:
    # - vec is not the middle: 2*vec != v1 + v2 for any pair v1,v2 in current_set
    # - v is not the middle: for each v in current_set, 2*v != vec + v' for any v' in current_set
    #   This means: for each v in current_set, v' = 2*v - vec should not be in current_set (and != vec)
    # - vec and v are endpoints: for each v in current_set, the midpoint m = 2*(vec+v) should not be in current_set (and != vec, v)
    
    cur_set = set(current_set)
    for v in current_set:
        # Case: vec is one endpoint, v is the other, check midpoint
        mid = tuple((veci + vi) * 2 % 3 for vi, veci in zip(v, vec))
        if mid in cur_set and mid != vec and mid != v:
            return False
        # Case: vec is the middle, v is an endpoint
        # Then the other endpoint w satisfies: 2*vec = v + w => w = 2*vec - v
        w = tuple((2*veci - vi) % 3 for vi, veci in zip(v, vec))
        if w in cur_set and w != vec and w != v:
            return False
    return True

def priority(vector, n):
    """
    Heuristic priority for greedy cap set construction.
    Vectors with higher priority are added first.
    """
    # Convert to tuple if needed
    vec = tuple(vector)
    
    # Heuristic: prefer vectors with balanced digit distribution
    # and low "interaction" with other common vectors
    # 
    # Strategy: assign priority based on:
    # 1. Number of 1s (middle value) - vectors with more 1s may be more "central"
    # 2. Sum of digits mod something
    # 3. A hash-based tiebreaker for determinism
    #
    # Better heuristic: use the fact that cap sets in Z_3^n have structure.
    # Vectors that are "farther apart" in some sense are less likely to form APs.
    #
    # Let's use: priority = - (number of pairs of coordinates that are equal)
    # This favors vectors with diverse coordinate values.
    
    # Count how many coordinates are 0, 1, 2
    counts = [0, 0, 0]
    for d in vec:
        counts[d] += 1
    
    # Penalize vectors that have many repeated coordinate values
    # A vector with all same digits is bad for cap sets
    penalty = counts[0]**2 + counts[1]**2 + counts[2]**2
    
    # Also use a small hash for tiebreaking to avoid all zeros
    h = hash(vec) % 1000
    
    # Higher priority = better
    # We want to add vectors that are "spread out"
    return -penalty * 0.001 + h * 0.0001

# EVOLVE-BLOCK-END
