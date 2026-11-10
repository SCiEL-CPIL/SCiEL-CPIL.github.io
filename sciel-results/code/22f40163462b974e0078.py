"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find the longest nonempty interval [start, start+length-1] contained in [1, 1008]
    such that for every x, y in the interval (including x=y), x*y % 1009 != 1.
    
    This is equivalent to: the interval must not contain any pair {a, b} where
    a*b ≡ 1 (mod 1009). Note that if a=b, then a^2 ≡ 1 (mod 1009), so a must be
    a self-inverse residue. The self-inverse residues mod 1009 are 1 and 1008
    (since 1009 is prime, x^2 ≡ 1 mod 1009 has solutions x ≡ ±1).
    
    For distinct a ≠ b, a*b ≡ 1 (mod 1009) means b ≡ a^{-1} (mod 1009).
    
    So the interval must not contain:
    - 1 or 1008 (self-inverse)
    - Any pair (a, a^{-1} mod 1009) for a ≠ a^{-1}
    
    We use a sweep-line / precompute approach: for each left boundary, compute
    the maximum valid right boundary by tracking exclusion constraints from
    inverse relationships.
    """
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    P = 1009
    MAX = 1008  # residues 1..1008
    
    # Precompute modular inverses
    inv = [0] * (P + 1)
    for x in range(1, P):
        inv[x] = pow(x, P - 2, P)
    
    # Identify self-inverse residues (x^2 ≡ 1 mod P)
    self_inverse = set()
    for x in range(1, P):
        if x * x % P == 1:
            self_inverse.add(x)
    # self_inverse = {1, 1008}
    
    # For each position i in 1..1008, determine the "exclusion" it imposes:
    # If i is self-inverse, it cannot be in any valid interval.
    # If i is not self-inverse, then inv[i] cannot be in the same interval as i.
    # 
    # Sweep-line approach:
    # For a fixed left boundary L, we want to find the maximum R such that
    # [L, R] contains no forbidden pairs.
    # 
    # Key insight: as we sweep L from 1 to 1008, we can maintain the minimum
    # "right exclusion" for each left. Specifically, for each element x in the
    # interval, if inv[x] > x, then the interval cannot extend to inv[x] if x
    # is included. But we need to be more careful.
    #
    # Alternative: precompute for each L the smallest R such that [L, R] is invalid,
    # then the max valid R for left L is that value minus 1.
    #
    # Let's think differently. For each pair (a, b) with a*b ≡ 1 mod P and a ≠ b,
    # assume a < b. Then any interval containing both a and b is invalid.
    # Also, any interval containing 1 or 1008 is invalid.
    #
    # So for a given left L, the interval [L, R] is valid iff:
    #   1. L > 1 and R < 1008 (or more precisely, 1 not in [L,R] and 1008 not in [L,R])
    #   2. For no pair (a,b) with a<b and a*b≡1 mod P do we have L ≤ a and b ≤ R.
    #
    # For condition 2: for each pair (a,b) with a<b, if L ≤ a, then R must be < b.
    # So for a given L, R_max = min over all pairs (a,b) with L≤a<b of (b-1),
    # also bounded by 1007 (to exclude 1008) and by L-1 if L≤1 (to exclude 1).
    #
    # More precisely:
    # - If L ≤ 1, we need R < 1, which is impossible, so L must be ≥ 2.
    #   Wait, 1 is self-inverse, so if L ≤ 1 ≤ R, invalid. So we need R < 1 or L > 1.
    #   Since L ≥ 1, we need L ≥ 2 or R < 1. R < 1 means empty, so L ≥ 2.
    # - If R ≥ 1008, we need L > 1008, impossible. So R ≤ 1007.
    # - For each inverse pair (a, b) with a < b: if L ≤ a, then R ≤ b - 1.
    #
    # So for a given L (with L ≥ 2), the maximum valid R is:
    #   R_max(L) = min(1007, min{b - 1 : (a,b) is an inverse pair with a < b and L ≤ a})
    #
    # Then the valid interval length for left L is max(0, R_max(L) - L + 1).
    #
    # We can precompute this efficiently:
    # 1. Collect all inverse pairs (a, b) with a < b.
    # 2. Sort them by a in descending order.
    # 3. As L decreases from 1008 to 2, we add pairs where a ≥ L, and track the
    #    minimum b among added pairs.
    
    # Collect inverse pairs with a < b
    pairs = []  # list of (a, b) with a < b and a*b ≡ 1 mod P
    seen_pairs = set()
    for x in range(1, P):
        if x in self_inverse:
            continue
        y = inv[x]
        if x < y:
            pairs.append((x, y))
    
    # Sort pairs by a in descending order
    pairs.sort(key=lambda p: -p[0])
    
    # Now sweep L from 1008 down to 2
    # For each L, we want min b among all pairs with a >= L
    # We can do this by iterating L from 1008 to 2, adding pairs with a == L
    
    # Build a list: for each a, the list of b's where (a,b) is a pair
    from collections import defaultdict
    a_to_bs = defaultdict(list)
    for a, b in pairs:
        a_to_bs[a].append(b)
    
    # Sweep L from 1008 down to 2
    min_b = 1008  # start with no constraint (b can be up to 1008, but we cap at 1007)
    best_len = 0
    best_s = 2
    best_e = 2
    
    # We need min b among all pairs with a >= L
    # Let's precompute: for each L, min_b_ge_L = min{b : (a,b) in pairs, a >= L}
    # We can do this with a suffix minimum over a.
    
    # Create an array min_b_at_a[a] = min b for pairs with that a, or infinity if none
    INF = 10**9
    min_b_at_a = [INF] * (P + 1)
    for a, b in pairs:
        if b < min_b_at_a[a]:
            min_b_at_a[a] = b
    
    # Suffix minimum: suffix_min[a] = min{min_b_at_a[a'], a' >= a}
    suffix_min = [INF] * (P + 2)
    for a in range(P, 0, -1):
        suffix_min[a] = min(min_b_at_a[a], suffix_min[a + 1])
    
    # Now for each L from 2 to 1007 (since R must be ≤ 1007 to exclude 1008,
    # and L must be ≥ 2 to exclude 1):
    #   R_max = min(1007, suffix_min[L] - 1) if suffix_min[L] < INF else 1007
    #   Wait, suffix_min[L] is the min b among pairs with a >= L.
    #   The constraint is: for any pair (a,b) with a >= L, we need R < b, i.e., R ≤ b-1.
    #   So R_max = min(1007, suffix_min[L] - 1)
    #   But also R >= L for a nonempty interval.
    
    for L in range(2, 1008):  # L from 2 to 1007
        if suffix_min[L] < INF:
            R_max = min(1007, suffix_min[L] - 1)
        else:
            R_max = 1007
        
        if R_max < L:
            continue
        
        length = R_max - L + 1
        if length > best_len:
            best_len = length
            best_s = L
            best_e = R_max
    
    # Verify the result
    if best_len > 0:
        s = best_s
        e = best_e
        valid = True
        for x in range(s, e + 1):
            if x in self_inverse:
                valid = False
                break
            y = inv[x]
            if s <= y <= e:
                valid = False
                break
        if valid:
            return [s, best_len]
    
    # Fallback: brute force with two-pointer
    group_id = [0] * (P + 1)
    group_counter = 0
    assigned = [False] * (P + 1)
    for x in range(1, P):
        if x in self_inverse:
            group_id[x] = -1
        elif not assigned[x]:
            group_counter += 1
            y = inv[x]
            group_id[x] = group_counter
            group_id[y] = group_counter
            assigned[x] = True
            assigned[y] = True
    
    freq = {}
    left = 1
    best_len = 0
    best_s = 1
    
    for right in range(1, 1009):
        g = group_id[right]
        if g == -1:
            freq.clear()
            left = right + 1
            if left > 1008:
                break
            continue
        else:
            freq[g] = freq.get(g, 0) + 1
            while freq.get(g, 0) > 1:
                lg = group_id[left]
                if lg != -1:
                    freq[lg] -= 1
                    if freq[lg] == 0:
                        del freq[lg]
                left += 1
        
        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_s = left
    
    if best_len > 0:
        s = best_s
        e = s + best_len - 1
        valid = True
        for x in range(s, e + 1):
            if x in self_inverse:
                valid = False
                break
            y = inv[x]
            if s <= y <= e:
                valid = False
                break
        if valid:
            return [s, best_len]
    
    # Absolute fallback
    return [2, 30]
# EVOLVE-BLOCK-END
