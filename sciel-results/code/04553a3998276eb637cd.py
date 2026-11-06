"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START

def construct():
    p = 1009
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse values
    self_inv = {1, p - 1}
    
    best_start = 2
    best_length = 1
    
    # Iterate over all possible starts
    for start in range(1, p):
        # Extend the interval as far as possible
        end = start
        while end < p:
            x = end
            # Check if x is self-inverse
            if x in self_inv:
                break
            # Check if the inverse of x is already in the interval [start, end]
            # If inv[x] is in [start, end], then the pair (x, inv[x]) is in the interval.
            # Note: if inv[x] == x, it is caught by self_inv check.
            if inv[x] >= start and inv[x] <= end:
                break
            end += 1
        
        length = end - start
        if length > best_length:
            best_length = length
            best_start = start
            
    return [best_start, best_length]

# EVOLVE-BLOCK-END
