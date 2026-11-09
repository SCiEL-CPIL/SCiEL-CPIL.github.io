"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    MOD = 1009
    N = 1008  # values are 1..1008
    
    # Compute modular inverse for each x in 1..1008
    # x * inv[x] ≡ 1 (mod 1009)
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Forbidden single values: 1 and 1008 (since 1*1%1009=1 and 1008*1008%1009=1)
    # So interval must not include 1 or 1008.
    # Therefore search in [2, 1007]
    
    lo_bound = 2
    hi_bound = 1007
    
    # We want the longest interval [s, e] such that for all x, y in [s, e], x*y % MOD != 1.
    # This is equivalent to: the set [s, e] does not contain any pair {x, inv[x]}.
    # Note: if x == inv[x], then x is a self-inverse. For prime p, x^2 ≡ 1 (mod p) 
    # means x ≡ 1 or x ≡ -1 (mod p), so x=1 and x=p-1=1008 are the only self-inverses.
    # These are already excluded.
    
    # For a pair {x, inv[x]} where x != inv[x], the interval [s,e] must not contain both.
    # So for each such pair, at least one of x or inv[x] must be outside [s,e].
    
    # Strategy: For each possible start s in [2, 1007], find the maximum e such that
    # no forbidden pair is entirely within [s, e].
    
    # A pair {a, b} (with a < b) is "active" in interval [s, e] if s <= a and b <= e.
    # So for a fixed s, the maximum e is the minimum over all pairs {a,b} with a >= s
    # of (b - 1), i.e., e must be < b for every pair with a >= s.
    # Wait, that's not quite right. Let me reconsider.
    
    # For fixed s, we need e such that for every pair {a, b} with a >= s, we don't have b <= e.
    # So e < min{b : {a,b} is a forbidden pair and a >= s}.
    # But also e >= s (nonempty interval).
    
    # Let's precompute all forbidden pairs where both elements are in [2, 1007].
    pairs = []
    seen = set()
    for x in range(2, 1008):
        y = inv[x]
        if y == x:
            # self-inverse, already excluded (only 1 and 1008)
            continue
        if y < 2 or y > 1007:
            continue
        # Create canonical pair
        a, b = min(x, y), max(x, y)
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))
    
    # Sort pairs by a
    pairs.sort()
    
    # For each start s, find the max e
    best_start = 2
    best_len = 0
    best_end = 2
    
    # For efficiency, for each s, we need the minimum b among pairs with a >= s.
    # Let's precompute: for each possible a value, what's the min b for pairs starting at a?
    # Then we can do a suffix minimum.
    
    min_b_for_a = [1009] * (MOD + 1)  # 1009 means no pair
    for a, b in pairs:
        if b < min_b_for_a[a]:
            min_b_for_a[a] = b
    
    # Suffix minimum: suffix_min[s] = min{min_b_for_a[a] : a >= s}
    suffix_min = [1009] * (MOD + 1)
    suffix_min[MOD] = 1009
    for i in range(MOD - 1, 1, -1):
        suffix_min[i] = min(min_b_for_a[i], suffix_min[i + 1])
    
    # For start s, the max end e is suffix_min[s] - 1 (since e < min_b_for any pair with a >= s)
    # But we also need e <= 1007 and e >= s.
    for s in range(2, 1008):
        max_e = suffix_min[s] - 1
        if max_e > 1007:
            max_e = 1007
        if max_e < s:
            continue
        length = max_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
            best_end = max_e
    
    # Verify the best solution
    s, e = best_start, best_end
    for x in range(s, e + 1):
        for y in range(s, e + 1):
            if (x * y) % MOD == 1:
                # Should not happen, but just in case
                break
        else:
            continue
        break
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return whatever we have
        pass
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
