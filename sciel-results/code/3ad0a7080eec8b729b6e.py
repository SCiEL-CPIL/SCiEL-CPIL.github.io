import time
from collections import defaultdict

def construct():
    """
    Constructs 101 nonempty lists of distinct denominators from 1..10
    such that the sum of their modular inverses mod 101 equals the row index.
    Minimizes the maximum length of any list.
    
    Strategy:
    1. Precompute all 2^10 - 1 nonempty subsets of {1..10}.
    2. For each subset, calculate its sum of inverses mod 101 and its size (popcount).
    3. For each residue r in 0..100, identify the subset(s) that sum to r.
    4. Select the subset with the minimum size for each residue r.
    5. The maximum size among these selected subsets is the theoretical minimum
       possible for the "largest row length" metric, because each row r must have
       a length at least the minimum size required to represent r.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    n = 10
    p = 101
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, -1, p)
        
    # best_terms[r] will store the list of denominators for residue r
    # with the minimum possible length.
    # best_len[r] will store that minimum length.
    best_terms = [None] * p
    best_len = [float('inf')] * p
    
    # Iterate through all non-empty masks of 10 bits
    # mask bit i (0-indexed) corresponds to denominator i+1
    for mask in range(1, 1 << n):
        # Early exit if deadline is approaching (unlikely for 1023 iterations)
        if time.time() > DEADLINE:
            break
            
        res = 0
        terms = []
        pc = 0
        m = mask
        # Unroll loop for speed, or just iterate bits
        for x in range(1, n + 1):
            if mask & (1 << (x - 1)):
                terms.append(x)
                res = (res + inv[x]) % p
                pc += 1
        
        # If this mask's popcount is smaller than the current best for this residue, update
        if pc < best_len[res]:
            best_len[res] = pc
            best_terms[res] = terms
            
    # Construct the final result
    # All residues 0..100 should be covered. 
    # If any are not (which shouldn't happen given 1023 subsets > 101 residues), 
    # we fall back to a simple representation.
    result = []
    for r in range(p):
        if best_terms[r] is not None:
            result.append(best_terms[r])
        else:
            # Fallback: find any representation
            # This path is theoretically unreachable for p=101, n=10
            found = False
            for mask in range(1, 1 << n):
                res = 0
                terms = []
                for x in range(1, n + 1):
                    if mask & (1 << (x - 1)):
                        terms.append(x)
                        res = (res + inv[x]) % p
                if res == r:
                    result.append(terms)
                    found = True
                    break
            if not found:
                # Absolute fallback, should never happen
                result.append([1])
                
    # Verification step (optional but good for debugging during development, 
    # removed in final tight loop if performance is critical, but here it's cheap)
    # We trust the logic above.
    
    return result
