"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Find the longest interval [start, start+length-1] contained in [1, 1008]
    such that for ALL x, y in the interval (including x=y), x*y % 1009 != 1.
    """
    p = 1009
    N = p - 1  # 1008

    # Step 1: Compute modular inverses for all x in [1, 1008]
    # Since p is prime, inv[x] = x^(p-2) mod p
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)

    # Step 2: Build conflict pairs
    # For each x, the conflict is with inv[x].
    # An interval is invalid if it contains both x and inv[x] for any x.
    # Note: if x == inv[x], then x^2 ≡ 1 (mod p), so x = 1 or x = p-1 = 1008.
    # These self-inverse elements cannot be in any valid interval.
    
    # For each possible left endpoint L, find the minimum right endpoint R
    # such that (L, R) is a conflict pair with L <= R.
    # If an interval starts at s and ends at e, it's invalid if there exists
    # a conflict pair (L, R) with s <= L and R <= e.
    
    # So for a given start s, the maximum valid end is:
    #   e_max(s) = min{ R | (L, R) is a conflict pair, L >= s } - 1
    # (capped at N)
    
    INF = float('inf')
    min_R_at_L = [INF] * (p + 2)  # index 0..1009, we use 1..1008

    for x in range(1, p):
        y = inv[x]
        L = min(x, y)
        R = max(x, y)
        # L is in [1, 1008], R is in [1, 1008]
        if R < min_R_at_L[L]:
            min_R_at_L[L] = R

    # Step 3: Suffix minimum to get min_R_from[s] = min{ R | L >= s }
    min_R_from = [INF] * (p + 2)
    min_R_from[p] = INF  # sentinel
    for s in range(p - 1, 0, -1):
        min_R_from[s] = min(min_R_at_L[s], min_R_from[s + 1])

    # Step 4: Find the longest valid interval
    best_len = 0
    best_start = 1

    for s in range(1, p):
        # Maximum valid end for start s
        max_e = min_R_from[s] - 1
        if max_e > N:
            max_e = N
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s

    # If no valid interval found (shouldn't happen), return a single element
    if best_len == 0:
        # Find any x in [1, 1008] that is not self-inverse
        for x in range(2, p - 1):
            if x != inv[x]:
                best_start = x
                best_len = 1
                break
        else:
            best_start = 1
            best_len = 1

    return [best_start, best_len]

# EVOLVE-BLOCK-END
