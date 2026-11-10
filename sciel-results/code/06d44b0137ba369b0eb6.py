import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
    
    # For each residue 0..100, find the minimum-size non-empty subset of {1..10}
    # whose inverses sum to that residue mod 101.
    # Since rows are independent (denominators may recur across rows), each row
    # independently takes the shortest valid subset.
    
    best_size = [11] * 101
    best_subset = [None] * 101
    
    # Enumerate all non-empty subsets via bitmask (1 to 1023)
    # Process in order of increasing popcount to find minimum size first
    # Group masks by popcount
    masks_by_size = [[] for _ in range(11)]
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        masks_by_size[size].append(mask)
    
    # For each size from 1 to 10, check if residue is not yet covered
    for size in range(1, 11):
        if time.time() > deadline:
            break
        for mask in masks_by_size[size]:
            # Compute sum of inverses for this subset
            s = 0
            terms = []
            for i in range(10):
                if mask & (1 << i):
                    denom = i + 1
                    terms.append(denom)
                    s = (s + inv[denom]) % 101
            
            # Only update if this is the first (smallest size) representation
            if best_size[s] > size:
                best_size[s] = size
                best_subset[s] = terms
    
    # Build result: for each residue, use the minimum-size subset found
    result = []
    for r in range(101):
        if best_subset[r] is not None:
            result.append(best_subset[r])
        else:
            # Fallback: should not happen, but use full set as safety
            result.append(list(range(1, 11)))
    
    return result
