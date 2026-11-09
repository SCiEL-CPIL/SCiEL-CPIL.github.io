"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    p = 1009
    # Deadline guard
    deadline = time.time() + 55.0
    
    # 1. Compute all quadratic residues and their roots
    # xs: 1 to p-1
    xs = np.arange(1, p, dtype=np.int64)
    # vs: x^2 mod p
    vs = (xs * xs) % p
    
    # root_map[v] = x such that x^2 == v mod p
    # If multiple x map to same v, the last one in xs wins.
    # Since we just need ANY valid root, this is fine.
    root_map = np.zeros(p, dtype=np.int64)
    root_map[vs] = xs
    
    # is_qr[v] = True if v is a non-zero QR
    is_qr = np.zeros(p, dtype=bool)
    is_qr[vs] = True
    
    # 2. Find the minimum s in [1, p-5] such that s..s+4 are all QRs
    # p-5 = 1004
    min_s = -1
    for s in range(1, p - 4):
        # Check if s, s+1, s+2, s+3, s+4 are all QR
        # is_qr[s:s+5] is a boolean array of length 5
        if is_qr[s:s+5].all():
            min_s = s
            break
            
    if min_s == -1:
        # Fallback: should not happen, but if it does, raise or return best effort
        # For now, try descending search just in case
        for s in range(p - 5, 0, -1):
            if is_qr[s:s+5].all():
                min_s = s
                break
                
    if min_s == -1:
        raise RuntimeError("No valid run of 5 consecutive QRs found")
        
    # 3. Retrieve roots
    roots = [int(root_map[min_s + i]) for i in range(5)]
    
    return {'start': min_s, 'roots': roots}
# EVOLVE-BLOCK-END
