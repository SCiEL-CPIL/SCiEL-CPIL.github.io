# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    p_minus_1 = p - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Conflict pairs (a, b) with a < b, a*b = 1 mod p
    # a, b in [2, 1007]
    # Since x != x^-1 for x in [2, 1007], all x form pairs.
    
    # min_b_for_a[a] = min b such that (a, b) is a conflict pair
    min_b_for_a = [p + 1] * (p + 1) # Use p+1 as infinity
    
    for x in range(2, p_minus_1): # 2 to 1007
        ix = inv[x]
        if ix < 2 or ix > p_minus_1:
            continue
        a, b = min(x, ix), max(x, ix)
        if b < min_b_for_a[a]:
            min_b_for_a[a] = b
            
    # Suffix minimum of min_b_for_a
    # suf_min[i] = min(min_b_for_a[i], min_b_for_a[i+1], ...)
    suf_min = [p + 1] * (p + 2)
    for i in range(p_minus_1, 0, -1):
        suf_min[i] = min(min_b_for_a[i], suf_min[i + 1])
        
    best_len = 0
    best_start = 2
    
    # Iterate possible starts
    for s in range(2, p_minus_1):
        # min b among pairs with a >= s
        min_b = suf_min[s]
        
        # Max e such that no pair (a,b) with a>=s and b<=e exists
        # We need b > e for all such pairs.
        # So e < min_b.
        # Max e is min_b - 1.
        # Also e <= 1007.
        max_e = min(p_minus_1, min_b - 1)
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                
    return [best_start, best_len]
# EVOLVE-BLOCK-END
