"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse residues: x such that x^2 % p == 1
    # For prime p, these are x = 1 and x = p-1 = 1008
    self_inverse = set([1, p - 1])
    
    best_len = 0
    best_start = 1
    L = 1
    
    for R in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Self-inverses 1 and 1008 cannot be in any valid interval
        if R in self_inverse:
            L = R + 1
            continue
        
        invR = inv[R]
        
        # Check if invR is in the current window [L, R-1]
        # If so, we have a conflict. We must move L past invR.
        if L <= invR < R:
            L = invR + 1
        
        # The window [L, R] is valid by construction:
        # - R is not a self-inverse
        # - inv(R) is not in [L, R-1]
        # - By induction, all pairs in [L, R-1] were valid
        # - For any x in [L, R-1], inv(x) cannot be in [L, R]:
        #   If inv(x) < R, it was excluded when x was added (or L moved past it)
        #   If inv(x) = R, then x = inv(R), but we just ensured L > inv(R) = x,
        #   contradicting x >= L
        
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
    
    if best_len == 0:
        # Fallback: find any valid single element
        for x in range(2, n):
            if x not in self_inverse:
                return [x, 1]
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
