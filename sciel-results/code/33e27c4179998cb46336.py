# EVOLVE-BLOCK-START
import time

def construct():
    # Precompute modular inverses
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
    
    # min_size[r] will store the minimum size of a subset summing to r
    # terms[r] will store one such subset
    min_size = [11] * 101
    best_terms = [None] * 101
    
    # Iterate through all non-empty subsets
    for mask in range(1, 1 << 10):
        # Calculate sum and size
        s = 0
        size = 0
        terms = []
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                s = (s + inv[x]) % 101
                size += 1
                terms.append(x)
        
        r = s
        if size < min_size[r]:
            min_size[r] = size
            best_terms[r] = terms
            
    # The maximum row length is the max of min_size
    max_len = max(min_size)
    
    # Construct result
    result = []
    for r in range(101):
        result.append(best_terms[r])
        
    return result
# EVOLVE-BLOCK-END
