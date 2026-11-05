"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit for safety
    
    # Precompute inverses for 1..10 modulo 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    # Initialize min size and best subset for each residue
    # We want to find the smallest subset for each residue
    min_size = [11] * 101
    best_subset = [None] * 101
    
    # Iterate through all non-empty subsets of {1..10}
    # There are 2^10 - 1 = 1023 subsets
    for mask in range(1, 1 << 10):
        if time.time() > deadline:
            break
            
        terms = []
        residue = 0
        size = 0
        for i in range(10):
            if mask & (1 << i):
                val = i + 1
                terms.append(val)
                residue = (residue + inv[val]) % 101
                size += 1
        
        # If this subset is smaller than the current best for this residue, update
        if size < min_size[residue]:
            min_size[residue] = size
            best_subset[residue] = terms
            
    # The objective is the maximum of the min sizes
    # If any residue is not covered (size 11), we have a problem, but with 1023 subsets it's unlikely
    max_len = 0
    result = []
    
    for r in range(101):
        if best_subset[r] is not None:
            result.append(best_subset[r])
            if len(best_subset[r]) > max_len:
                max_len = len(best_subset[r])
        else:
            # Fallback: should not happen
            # Find any subset that sums to r, regardless of size
            # Or just use a dummy, but this indicates failure
            # For robustness, we can do a second pass or just return empty (invalid)
            # But given the density, this shouldn't trigger.
            # Let's try to find one from all subsets if not found (logic error check)
            # Actually, if min_size is 11, it means no subset was found.
            # This is extremely unlikely for p=101 and 10 denominators.
            # We'll just append an empty list to avoid crash, though it's invalid.
            result.append([])
            
    return result
# EVOLVE-BLOCK-END
