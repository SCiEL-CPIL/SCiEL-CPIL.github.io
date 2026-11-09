"""Model-and-optimize approach for finding the earliest run of 5 consecutive quadratic residues mod 1009."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    p = 1009
    m = 5

    # Precompute all quadratic residues with their minimal positive root
    # root[r] = smallest x in [1, p-1] such that x^2 mod p == r
    root = {}
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        if r not in root or x < root[r]:
            root[r] = x

    # Build a boolean array: is_residue[i] is True iff i is a nonzero QR mod p
    # Using numpy for vectorized operations
    is_residue = np.zeros(p, dtype=bool)
    for r in root:
        is_residue[r] = True

    # We need the earliest s in [1, p-m] such that s, s+1, ..., s+m-1 are all nonzero QRs.
    # Using vectorized AND of shifted arrays
    # valid_starts[s] is True iff is_residue[s] & is_residue[s+1] & ... & is_residue[s+4] is True
    
    # Slicing to align indices:
    # is_residue[0:p-m+1] corresponds to s=0..p-m
    # is_residue[1:p-m+2] corresponds to s=0..p-m (shifted by 1)
    # ...
    # is_residue[4:p-m+5] corresponds to s=0..p-m (shifted by 4)
    
    # p-m+1 = 1009-5+1 = 1005
    # We are interested in s from 1 to 1004.
    # valid_starts will have length 1005 (indices 0..1004).
    
    valid_starts = (
        is_residue[0:p-m+1] & 
        is_residue[1:p-m+2] & 
        is_residue[2:p-m+3] & 
        is_residue[3:p-m+4] & 
        is_residue[4:p-m+5]
    )
    
    # Find the first index s >= 1 where valid_starts[s] is True
    # np.nonzero returns indices where the condition is True
    indices = np.nonzero(valid_starts)[0]
    
    best_s = None
    best_roots = None
    
    for s in indices:
        if s >= 1:
            best_s = s
            best_roots = [root[s + i] for i in range(m)]
            break
    
    if best_s is None:
        # Fallback: shouldn't happen, but if we hit the deadline or something went wrong,
        # do a quick descending search as in the original
        for s in range(p - 1, 0, -1):
            if all(is_residue[s + i] for i in range(m)):
                best_s = s
                best_roots = [root[s + i] for i in range(m)]
                break

    if best_s is None:
        raise RuntimeError('No run of 5 consecutive nonzero quadratic residues found')

    return {'start': best_s, 'roots': best_roots}

# EVOLVE-BLOCK-END
