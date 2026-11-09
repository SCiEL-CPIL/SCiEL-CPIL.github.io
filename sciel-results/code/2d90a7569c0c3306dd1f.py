"""Combined program: MODEL-AND-OPTIMIZE approach with clean implementation.
We model the answer as an assignment of one subset (denominator mask) to each residue r mod 101.
Objective: minimize the maximum row length (i.e., the max popcount over all 101 rows).
We precompute all 1023 nonempty masks with their residues and popcounts.
Then for each residue, we pick the mask with minimum popcount that achieves it.
The maximum of these minimums is the theoretical optimum for this problem.
"""
import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill

    n = 10
    p = 101
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, -1, p)
    
    # best_terms[r] stores the list of denominators for residue r with minimum length
    # best_len[r] stores that minimum length
    best_terms = [None] * p
    best_len = [float('inf')] * p
    
    # Iterate through all non-empty masks of 10 bits
    # mask bit i (0-indexed) corresponds to denominator i+1
    for mask in range(1, 1 << n):
        if time.time() > DEADLINE:
            break
            
        res = 0
        terms = []
        pc = 0
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
    result = []
    for r in range(p):
        if best_terms[r] is not None:
            result.append(best_terms[r])
        else:
            # Fallback: find any representation (shouldn't happen for p=101, n=10)
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
    
    return result
