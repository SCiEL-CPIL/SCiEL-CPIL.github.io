"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Precompute modular inverses for 1 to N
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # An interval [s, e] is valid iff for all x in [s, e], inv[x] < s or inv[x] > e.
    # Equivalently, for all x in [s, e], x and inv[x] are NOT both in [s, e].
    
    # Key insight: For a valid interval [s, e], if x is in [s, e] and inv[x] is also in [s, e],
    # that's a conflict. So we need: for all x in [s, e], inv[x] is outside [s, e].
    
    # This means: for all x in [s, e], inv[x] < s or inv[x] > e.
    
    # Sweep-line approach:
    # For each position s, find the maximum e such that [s, e] is valid.
    # 
    # The constraint is: no x in [s, e] has inv[x] in [s, e].
    # Equivalently: for all x in [s, e], min(x, inv[x]) < s or max(x, inv[x]) > e.
    
    # For a fixed s, the valid range of e is constrained by all x >= s where inv[x] >= s.
    # For such x, we need e < inv[x] (since inv[x] >= s, the condition inv[x] > e must hold).
    # So e_max(s) = min over all x in [s, N] with inv[x] >= s of (inv[x] - 1).
    # But we also need e >= s, and we need to check that all x in [s, e] satisfy the condition.
    
    # Actually, let me think more carefully. For interval [s, e] to be valid:
    # For every x in [s, e]: inv[x] < s or inv[x] > e.
    
    # Define for each x: if inv[x] >= x, then for any interval containing x, we need e < inv[x] 
    #   (if s <= x) OR s > inv[x] (impossible since inv[x] >= x >= s).
    #   Wait, that's not right either.
    
    # Let me use a different approach. For each x, define:
    #   a[x] = min(x, inv[x])
    #   b[x] = max(x, inv[x])
    # If a[x] == b[x], then x is self-inverse (x^2 ≡ 1 mod p), so x is "forbidden".
    # Otherwise, for interval [s, e] to be valid, we need: NOT (s <= a[x] AND b[x] <= e).
    # That is, for each non-self-inverse x, the interval [s, e] must NOT contain both a[x] and b[x].
    
    # So the problem reduces to: find the longest interval [s, e] in [1, N] that does not 
    # contain both endpoints of any "forbidden pair" (a[x], b[x]) where a[x] != b[x],
    # and does not contain any self-inverse element.
    
    # Self-inverse elements: x^2 ≡ 1 mod 1009. Since 1009 is prime, solutions are x ≡ 1 and x ≡ -1 ≡ 1008.
    # So positions 1 and 1008 are forbidden.
    
    # For the forbidden pairs: for each x, the pair (min(x,inv[x]), max(x,inv[x])) must not both be in [s,e].
    
    # Sweep-line: for a given s, what is the maximum valid e?
    # e must satisfy: for all x with a[x] >= s, we need e < b[x] (if b[x] > a[x]).
    # Wait, more precisely: for all x, if s <= a[x] and b[x] <= e, that's invalid.
    # So for a fixed s, e must be less than b[x] for all x where a[x] >= s and b[x] > a[x].
    # Actually: for all x where a[x] >= s, we need e < b[x].
    # And for x where a[x] < s, the pair can't both be in [s,e] since a[x] < s, so no constraint.
    
    # So: e_max(s) = min{ b[x] - 1 : x in [1,N], a[x] >= s, a[x] != b[x] }
    # But we also need to exclude self-inverse elements from [s, e].
    # Self-inverse elements are at positions 1 and 1008.
    # So if s <= 1 <= e, invalid. If s <= 1008 <= e, invalid.
    
    # Let me precompute: for each s, what is the minimum b[x] - 1 over all x with a[x] >= s?
    # We can do this by processing x in decreasing order of a[x].
    
    # Create list of (a[x], b[x]) for non-self-inverse x.
    pairs = []
    for x in range(1, N + 1):
        if inv[x] == x:
            continue  # self-inverse, forbidden
        a = min(x, inv[x])
        b = max(x, inv[x])
        pairs.append((a, b))
    
    # Sort by a in decreasing order
    pairs.sort(key=lambda p: -p[0])
    
    # For each s, e_max(s) = min{ b - 1 : (a, b) in pairs, a >= s }
    # Process s from N down to 1, maintaining the minimum b-1 for pairs with a >= s.
    
    e_max = [0] * (N + 2)  # e_max[s] = maximum valid e for start s
    min_b_minus_1 = N + 1  # infinity sentinel
    
    # Pointers into sorted pairs
    pair_idx = 0
    for s in range(N, 0, -1):
        # Add all pairs with a == s (since we're going decreasing, add pairs with a >= s)
        while pair_idx < len(pairs) and pairs[pair_idx][0] >= s:
            min_b_minus_1 = min(min_b_minus_1, pairs[pair_idx][1] - 1)
            pair_idx += 1
        
        if min_b_minus_1 == N + 1:
            # No constraint from pairs
            e_max[s] = N
        else:
            e_max[s] = min_b_minus_1
    
    # Now handle self-inverse elements (1 and 1008):
    # If s <= 1, then e must be < 1, i.e., e < s, so no valid interval starting at s <= 1.
    # Actually, if s <= 1 <= e, invalid. Since s >= 1, this means if s == 1, e must be < 1, impossible.
    # So s = 1 is always invalid.
    # If s <= 1008 <= e, invalid. Since e <= N = 1008, this means if s <= 1008, e must be < 1008.
    # So for all s, e must be < 1008, i.e., e <= 1007.
    
    # Wait, let me reconsider. The self-inverse elements are 1 and 1008.
    # For interval [s, e] to be valid, it must NOT contain 1 and must NOT contain 1008.
    # So: e < 1 OR s > 1 (for excluding 1), and e < 1008 OR s > 1008 (for excluding 1008).
    # Since s >= 1 and e <= 1008:
    #   To exclude 1: need s > 1 (since e >= s >= 1, so e < 1 is impossible).
    #   To exclude 1008: need e < 1008 (since s <= 1008, so s > 1008 is impossible).
    
    # Therefore: s must be >= 2, and e must be <= 1007.
    
    best_start = 2
    best_length = 0
    
    for s in range(2, N):  # s from 2 to 1007
        e = e_max[s]
        # Also cap e at 1007 to exclude self-inverse 1008
        e = min(e, N - 1)  # N-1 = 1007
        if e >= s:
            length = e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
    
    if best_length == 0:
        # Fallback: try all valid intervals by brute force (shouldn't happen for N=1008)
        for s in range(2, N):
            for e in range(N - 1, s - 1, -1):
                valid = True
                for x in range(s, e + 1):
                    if inv[x] == x or (s <= inv[x] <= e):
                        valid = False
                        break
                if valid:
                    best_length = e - s + 1
                    best_start = s
                    break
            if best_length > 0:
                break
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
