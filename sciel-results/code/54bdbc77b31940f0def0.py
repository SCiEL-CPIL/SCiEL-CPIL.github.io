"""
Find the least run-start s for 5 consecutive non-zero quadratic residues mod 1009.
Uses a bitmask for efficient consecutive residue checking.
"""
import time

_P = 1009
_M = 5

def construct():
    start_time = time.time()
    
    # 1. Precompute quadratic residues using a bitmask
    # bit i is 1 if i is a non-zero quadratic residue mod _P
    mask = 0
    # Store one root for each residue (smallest positive root)
    root_of = [0] * _P
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            mask |= (1 << r)
            if root_of[r] == 0:
                root_of[r] = x

    # 2. Search for the minimum s
    # We need 5 consecutive bits set, starting at s.
    # mask has bits 0..1008.
    # s ranges from 1 to 1004.
    # Check: ((mask >> s) & 0b11111) == 0b11111
    
    # Precompute the mask for 5 consecutive 1s
    check_mask = (1 << _M) - 1  # 0b11111
    
    for s in range(1, _P - _M + 1):
        if ((mask >> s) & check_mask) == check_mask:
            # Found the minimal s
            roots = [root_of[s + i] for i in range(_M)]
            return {'start': s, 'roots': roots}
            
        # Safety check for time
        if time.time() - start_time > 80:
            break

    raise RuntimeError("No valid run found")
