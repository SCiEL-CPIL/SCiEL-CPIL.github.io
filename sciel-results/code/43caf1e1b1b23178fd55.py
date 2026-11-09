"""Cap set greedy priority with deadline guard and combinatorial heuristics."""
import time
import itertools

# Global deadline guard: kill at 330s, safety margin 15s
_DEADLINE = time.time() + 315.0

def _time_left():
    return _DEADLINE - time.time()

def _is_cap_set(vecs, n):
    """Check if a set of vectors in {0,1,2}^n is a cap set (no 3-term AP)."""
    # A 3-term AP: a, b, c with b = (a+c)/2, i.e., a+c = 2b coordinate-wise
    # In {0,1,2}, 2b has coordinates in {0,2,4}. But we work mod 3? 
    # Actually cap set in {0,1,2}^n means no three distinct points x,y,z with x+z=2y (integer arithmetic).
    # Equivalently, no x != z such that (x+z)/2 is in the set and equals y.
    # Since x_i, z_i in {0,1,2}, x_i+z_i in {0,1,2,3,4}, and (x_i+z_i)/2 is integer only if x_i+z_i is even.
    # So x_i+z_i must be in {0,2,4} for each coordinate, meaning x_i and z_i have same parity.
    
    s = set(vecs)
    for a in vecs:
        for b in vecs:
            if a == b:
                continue
            # Check if c = 2b - a is in the set
            c = tuple(2*bi - ai for ai, bi in zip(a, b))
            if all(0 <= ci <= 2 for ci in c):
                if c in s and c != a and c != b:
                    return False
    return True

def _would_create_ap(new_vec, existing_vecs, n):
    """Check if adding new_vec to existing_vecs creates a 3-term AP."""
    s = set(existing_vecs)
    s.add(new_vec)
    # Only need to check APs involving new_vec
    for a in existing_vecs:
        # Check if new_vec, a, c form AP: c = 2a - new_vec
        c = tuple(2*ai - ni for ai, ni in zip(a, new_vec))
        if all(0 <= ci <= 2 for ci in c):
            if c in s and c != a and c != new_vec:
                return True
        # Check if a, new_vec, c form AP: c = 2*new_vec - a
        c = tuple(2*ni - ai for ai, ni in zip(a, new_vec))
        if all(0 <= ci <= 2 for ci in c):
            if c in s and c != a and c != new_vec:
                return True
        # Check if new_vec, c, a form AP: c = (a + new_vec)/2
        # a_i + new_vec_i must be even
        valid = True
        c = []
        for ai, ni in zip(a, new_vec):
            if (ai + ni) % 2 != 0:
                valid = False
                break
            c.append((ai + ni) // 2)
        if valid:
            c = tuple(c)
            if c in s and c != a and c != new_vec:
                return True
    return False

def _spatial_spread(vec, n):
    """Compute spatial spread of 1-positions: sum of pairwise distances between positions where vec[i]==1."""
    ones = [i for i in range(n) if vec[i] == 1]
    if len(ones) < 2:
        return 0
    total = 0
    for i in range(len(ones)):
        for j in range(i+1, len(ones)):
            total += abs(ones[i] - ones[j])
    return total

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Returns a float; higher priority vectors are considered first.
    Uses a combination of:
    1. Balanced count of 1s (prefer vectors with ~n/3 ones)
    2. Spatial spread of 1-positions (prefer spread out)
    3. Lexicographic tie-breaker
    """
    # Time guard
    if _time_left() < 5.0:
        return 0.0
    
    v = tuple(vector)
    count_ones = sum(1 for x in v if x == 1)
    
    # Primary: prefer vectors with count close to n/3
    target = n / 3.0
    balance_score = -abs(count_ones - target)
    
    # Secondary: spatial spread
    spread = _spatial_spread(v, n)
    
    # Tertiary: slight preference for more 2s (to balance the set)
    count_twos = sum(1 for x in v if x == 2)
    
    # Combine: balance is dominant, spread is secondary
    # Scale spread to be comparable
    max_spread = n * (n - 1) // 2
    if max_spread > 0:
        spread_norm = spread / max_spread
    else:
        spread_norm = 0.0
    
    score = balance_score * 10.0 + spread_norm * 5.0 + count_twos * 0.1
    
    return score
