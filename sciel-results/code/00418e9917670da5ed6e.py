"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Self-inverse elements: x^2 ≡ 1 (mod p) means x ≡ 1 or x ≡ -1
    # So x = 1 and x = p-1 = 1008
    self_inverse = {1, p - 1}
    
    best_start = 1
    best_length = 0
    
    # For each start s, find the maximum valid end e
    # An interval [s, e] is valid iff:
    # 1. No self-inverse element is in [s, e]
    # 2. For no x in [s, e] is inv[x] also in [s, e]
    
    # For a fixed start s, the constraint is:
    # - If s <= 1 <= e: invalid (contains self-inverse 1)
    # - If s <= 1008 <= e: invalid (contains self-inverse 1008)
    # - For each x in [s, e], inv[x] must not be in [s, e]
    #   i.e., for each x in [s, e], either inv[x] < s or inv[x] > e
    
    # Strategy: for each s, compute max_e = min over all constraints
    # Self-inverse constraints:
    #   If s <= 1: max_e < 1, so no valid interval starting at s unless s > 1
    #   If s <= 1008: max_e < 1008
    
    # For pair constraints: for each x >= s, if inv[x] >= s, 
    #   then we cannot have both x and inv[x] in the interval
    #   So e < max(x, inv[x]) if min(x, inv[x]) >= s
    #   Actually: if both x and inv[x] are >= s, then e < max(x, inv[x])
    
    # More precisely: for interval [s, e] to be valid,
    # for every x in [s, e], inv[x] must be outside [s, e]
    # This means: for every x in [s, e], inv[x] < s OR inv[x] > e
    
    # Equivalent: e < min{max(x, inv[x]) : x in [s, e] and inv[x] >= s}
    # But this is circular. Let me think differently.
    
    # For a fixed s, define blocking_end(x) for x >= s:
    # If inv[x] >= s, then having both x and inv[x] in the interval is bad.
    # The "blocking" value is: we can't extend e to include both x and inv[x].
    # If we're considering interval [s, e], and x is in [s, e], then inv[x] must not be in [s, e].
    
    # Alternative: for fixed s, the maximum valid e is the smallest value such that
    # no pair (x, inv[x]) with both elements >= s has both elements <= e.
    
    # For each unordered pair {a, b} where a*b ≡ 1 (mod p), a ≠ b, a,b in [1,1008]:
    #   Let lo = min(a,b), hi = max(a,b)
    #   For start s <= lo: the interval [s, e] is blocked at e >= hi (can't include both)
    #   So for start s, if s <= lo, then e must be < hi
    #   This gives constraint: e < hi for all pairs with lo >= s
    #   Wait, no: if s <= lo, then lo is in [s, ?] and hi is in [s, ?] if e >= hi
    #   So we need e < hi for all pairs where lo >= s... no.
    #   If s <= lo, then lo >= s, so lo could be in the interval. If e >= hi, then hi is also in the interval.
    #   So constraint: if s <= lo, then e < hi.
    
    # Also self-inverse: if s <= 1, then e < 1 (impossible for nonempty), so s must be > 1
    # If s <= 1008, then e < 1008
    
    # So for each s:
    #   max_e = p - 1  (1008)
    #   if s <= 1: return invalid (but s starts at 1, so handle)
    #   if s <= 1008: max_e = min(max_e, 1007)  # e < 1008
    #   For each pair (lo, hi) with lo >= s: max_e = min(max_e, hi - 1)
    
    # But we need to be careful: the pair constraint applies when BOTH lo and hi are in [s, e].
    # lo is in [s, e] iff lo >= s and lo <= e
    # hi is in [s, e] iff hi >= s and hi <= e
    # Since hi > lo, hi >= s is implied by lo >= s (if lo >= s then hi > lo >= s)
    # So the constraint is: if lo >= s, then we cannot have e >= hi
    # Thus: e < hi for all pairs with lo >= s
    
    # Precompute all pairs
    pairs = []
    seen = set()
    for x in range(1, p):
        if x in self_inverse:
            continue
        y = inv[x]
        if x == y:
            continue
        a, b = min(x, y), max(x, y)
        pair_key = (a, b)
        if pair_key not in seen:
            seen.add(pair_key)
            pairs.append((a, b))
    
    # For each start s, compute max valid e
    # max_e(s) = min(1007 if s <= 1008 else 1008, min{hi - 1 : (lo, hi) in pairs, lo >= s})
    # But also need e >= s for nonempty interval
    
    # We can compute this efficiently by sorting pairs by lo and using a suffix minimum
    # Sort pairs by lo
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    
    # For each possible lo value, store the minimum hi among pairs with that lo or greater
    # Actually, we want for each s: min{hi : (lo, hi) in pairs, lo >= s}
    # Let's create an array where for each position s, we store min_hi_from_s
    # min_hi_from_s = min{hi : (lo, hi) in pairs, lo >= s}
    
    # Initialize with infinity
    INF = float('inf')
    min_hi_from_s = [INF] * (p + 2)
    
    # For each pair (lo, hi), it contributes hi to all s <= lo
    # So min_hi_from_s[s] = min{hi : (lo, hi) in pairs, lo >= s}
    # We can compute this by iterating s from p-1 down to 1
    # min_hi_from_s[s] = min(min_hi_from_s[s+1], min{hi : (lo, hi) in pairs, lo == s})
    
    # Build: for each lo, store min hi for that lo
    min_hi_at_lo = [INF] * (p + 2)
    for lo, hi in pairs:
        if hi < min_hi_at_lo[lo]:
            min_hi_at_lo[lo] = hi
    
    # Suffix minimum
    for s in range(p - 1, 0, -1):
        min_hi_from_s[s] = min(min_hi_from_s[s + 1], min_hi_at_lo[s])
    
    # Now for each start s:
    #   constraint_from_pairs = min_hi_from_s[s] - 1 if min_hi_from_s[s] < INF else p - 1
    #   constraint_from_self_inv = 1007 if s <= 1008 else 1008
    #   If s <= 1: no valid interval (since 1 is self-inverse and 1 is in [s, e] for any e >= 1)
    #   Actually if s = 1, then 1 is in the interval, and 1*1 = 1 mod 1009, so invalid.
    #   So s must be >= 2.
    #   max_e = min(constraint_from_pairs, constraint_from_self_inv)
    #   If max_e >= s: valid interval [s, max_e] with length max_e - s + 1
    
    for s in range(2, p):  # s from 2 to 1008
        # Self-inverse constraint: 1008 is self-inverse
        # If s <= 1008, then e must be < 1008, so e <= 1007
        # Since s >= 2, s <= 1008 is always true
        max_e = 1007
        
        # Pair constraint
        if min_hi_from_s[s] < INF:
            max_e = min(max_e, min_hi_from_s[s] - 1)
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
