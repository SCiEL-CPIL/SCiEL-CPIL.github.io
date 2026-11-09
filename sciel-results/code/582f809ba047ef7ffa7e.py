import time
import sys
import hashlib
import struct

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill

def priority(vector, n):
    """
    Prioritizes vectors based on the balance of their digit distribution.
    
    Heuristic:
    Vectors where the counts of 0s, 1s, and 2s are as close to n/3 as possible
    are considered 'balanced' or 'uniform'. Prioritizing these vectors helps
    avoid the early formation of 3-term arithmetic progressions (cap set violations),
    leading to larger final cap sets.
    
    This is combined with a deterministic hash to break ties and ensure diversity
    among vectors with similar balance scores.
    """
    # Calculate counts of each digit
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    # Calculate the squared deviation from the ideal uniform distribution (n/3 for each)
    # This measures how "unbalanced" the vector is.
    # Lower deviation is better.
    dev0 = c0 - n / 3.0
    dev1 = c1 - n / 3.0
    dev2 = c2 - n / 3.0
    
    balance_score = dev0*dev0 + dev1*dev1 + dev2*dev2
    
    # We want to minimize balance_score, so we use negative score as primary key.
    # Add a small deterministic hash to break ties.
    # The hash is scaled down so it only affects vectors with nearly identical balance.
    
    # Use MD5 for a fast, stable hash
    # Pack the vector into bytes
    # Note: vector values are 0, 1, 2, which fit in a byte.
    try:
        b = struct.pack(f'{n}B', *vector)
        h = int(hashlib.md5(b).hexdigest(), 16)
        # Normalize hash to [0, 1)
        norm_h = (h % 1000000) / 1000000.0
    except Exception:
        # Fallback if packing fails (e.g., very large n or weird types, though unlikely)
        h = 0
        for i, val in enumerate(vector):
            h = (h * 31 + val * (i + 1)) % 1000000007
        norm_h = (h % 1000000) / 1000000.0

    # Primary: -balance_score
    # Secondary: small fraction of hash
    # The factor 0.01 ensures that a difference in balance_score of > 0.01 
    # determines the order, while the hash breaks exact ties or very close calls.
    return -balance_score + norm_h * 0.01
