"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    p = 1009
    
    # Precompute modular inverses for 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse: x^2 ≡ 1 mod p => x=1 or x=p-1=1008
    # So 1 and 1008 cannot be in any valid interval
    
    best_start = 1
    best_len = 0
    
    n = 1008  # max value in range [1, 1008]
    
    for s in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # If s is 1 or 1008, skip (self-inverse)
        if s == 1 or s == 1008:
            continue
        
        # Extend e from s upward
        valid = True
        for e in range(s, n + 1):
            if time.time() > deadline:
                break
            
            # Check if e itself is a self-inverse
            if e == 1 or e == 1008:
                valid = False
                break
            
            # Check if inv[e] is in [s, e-1] (i.e., some earlier element in the interval)
            ie = inv[e]
            if s <= ie < e:
                valid = False
                break
            
            if e - s + 1 > best_len:
                best_len = e - s + 1
                best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
