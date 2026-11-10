"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Finds the longest contiguous interval [start, start+length-1] within [1, 1008]
    such that for all x, y in the interval, x*y % 1009 != 1.
    
    This is equivalent to finding the longest interval that does not contain
    any pair (x, inv(x)) where inv(x) is the modular inverse of x mod 1009.
    Note that x=1 and x=1008 are self-inverses, so they cannot be in any valid interval.
    
    Algorithm:
    1. Precompute modular inverses for 1..1008.
    2. Use a two-pointer (sweep-line) technique.
       Let L be the start of the interval.
       Let R be the maximum end of the interval starting at L.
       As L increases, R is non-decreasing.
       For a fixed L, we extend R as far as possible such that
       for all x in [L, R], inv(x) is not in [L, R].
       
       The condition for adding a new element x = R+1 to the interval [L, R] is:
       inv(x) must not be in [L, x].
       Since x is the largest element, this simplifies to:
       inv(x) < L or inv(x) > x.
    """
    p = 1009
    n = p - 1  # 1008
    
    # Precompute inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
        
    best_len = 0
    best_start = 1
    R = 0
    
    for L in range(1, n + 1):
        if R < L:
            R = L - 1
            
        # Extend R as far as possible
        while R + 1 <= n:
            x = R + 1
            ix = inv[x]
            # Check if ix is inside [L, x]
            if ix < L or ix > x:
                R = x
            else:
                break
                
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
            
    return [best_start, best_len]
# EVOLVE-BLOCK-END
