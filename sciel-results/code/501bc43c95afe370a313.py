"""Find the minimal start s for 5 consecutive quadratic residues modulo 1009."""
import time

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    p = 1009
    m = 5
    
    # Precompute a list where root_of[r] is a root x such that x*x % p == r
    # 0 indicates r is not a non-zero quadratic residue.
    root_of = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_of[r] == 0:
            root_of[r] = x

    # Create boolean array for QRs
    is_qr = np.zeros(p, dtype=bool)
    for r in range(1, p):
        if root_of[r] != 0:
            is_qr[r] = True
            
    # Find minimal s such that s..s+m-1 are all QRs
    # Use convolution to compute sliding window sums
    conv = np.convolve(is_qr, np.ones(m), mode='valid')
    
    # conv[s] is sum of is_qr[s : s+m]
    # We want the first s >= 1 such that conv[s] == m
    found_s = -1
    for s in range(1, p - m + 1):
        if conv[s] == m:
            found_s = s
            break
            
    if found_s == -1:
        # Fallback linear scan
        for s in range(1, p - m + 1):
            if all(root_of[s+i] != 0 for i in range(m)):
                found_s = s
                break
                
    if found_s == -1:
        raise RuntimeError('No run of 5 consecutive QRs found')
        
    roots = [root_of[found_s + i] for i in range(m)]
    
    return {'start': found_s, 'roots': roots}

# EVOLVE-BLOCK-END
