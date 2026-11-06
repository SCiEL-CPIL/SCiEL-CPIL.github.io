"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 58  # Safety margin
    
    # Modular inverses of 1..10 mod 101
    inverses = [0] * 11
    for i in range(1, 11):
        inverses[i] = pow(i, -1, 101)
        
    # best_terms[r] will store the list of denominators for residue r
    # best_size[r] will store the minimum size found so far
    best_terms = [None] * 101
    best_size = [11] * 101  # 11 is effectively infinity since max size is 10
    
    # Iterate through all non-empty subsets of {1, ..., 10}
    # We can iterate by mask
    for mask in range(1, 1 << 10):
        if time.time() > deadline:
            break
            
        terms = []
        s = 0
        size = 0
        # Extract bits
        m = mask
        idx = 1
        while m:
            if m & 1:
                terms.append(idx)
                s += inverses[idx]
                size += 1
            m >>= 1
            idx += 1
            
        r = s % 101
        
        if size < best_size[r]:
            best_size[r] = size
            best_terms[r] = terms
    
    # Construct the result list
    # If any residue is not covered (best_size is still 11), 
    # we must have a bug or the problem is unsolvable for some K < 11.
    # But we know it's solvable.
    result = []
    for r in range(101):
        if best_terms[r] is None:
            # Fallback: should not happen
            # Find any subset
            for mask in range(1, 1 << 10):
                terms = []
                s = 0
                m = mask
                idx = 1
                while m:
                    if m & 1:
                        terms.append(idx)
                        s += inverses[idx]
                    m >>= 1
                    idx += 1
                if s % 101 == r:
                    best_terms[r] = terms
                    break
        result.append(best_terms[r])
        
    return result
# EVOLVE-BLOCK-END
