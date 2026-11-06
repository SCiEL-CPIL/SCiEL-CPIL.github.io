import time
from collections import defaultdict

def construct():
    """
    Constructs 101 lists of distinct denominators in 1..10 such that the sum of 
    their modular inverses mod 101 equals the row index.
    Minimizes the maximum length of the lists.
    """
    start_time = time.time()
    DEADLINE = 89.0  # 1 second safety margin before 90s limit
    
    MOD = 101
    DENOMS = list(range(1, 11))
    
    # Precompute modular inverses
    invs = [0] * 11
    for x in DENOMS:
        invs[x] = pow(x, -1, MOD)
        
    # We need to find, for each residue r in 0..100, the subset of DENOMS
    # with the minimum size that sums to r mod 101.
    
    # min_size_for_res[r] will store the minimum size of a subset summing to r.
    # best_subset_for_res[r] will store the actual subset (list of denominators).
    
    min_size_for_res = [11] * MOD  # 11 is effectively infinity (max size is 10)
    best_subset_for_res = [None] * MOD
    
    # Iterate through all non-empty subsets of DENOMS
    # There are 2^10 - 1 = 1023 such subsets.
    # We can iterate by mask from 1 to 1023.
    
    # To potentially find smaller subsets first, we could iterate by size, 
    # but since we just need the global minimum per residue, and 1023 is small,
    # we can just iterate all masks and update if size is smaller.
    
    for mask in range(1, 1 << 10):
        # Calculate sum and size for this subset
        current_sum = 0
        current_size = 0
        terms = []
        
        for i in range(10):
            if mask & (1 << i):
                denom = i + 1
                terms.append(denom)
                current_sum = (current_sum + invs[denom]) % MOD
                current_size += 1
        
        residue = current_sum
        
        # Update if this subset is smaller than the current best for this residue
        if current_size < min_size_for_res[residue]:
            min_size_for_res[residue] = current_size
            best_subset_for_res[residue] = terms
            
        # Deadline check (unlikely to be needed for 1023 iterations, but required)
        if time.time() - start_time > DEADLINE:
            # If we run out of time, return what we have so far.
            # However, since the loop is tiny, this won't happen.
            break

    # Construct the result
    result = []
    for r in range(MOD):
        if best_subset_for_res[r] is None:
            # This should not happen if all residues are reachable.
            # Fallback: try to find any subset, or return empty if truly impossible.
            # Given the problem statement, they are reachable.
            # We'll do a quick search if it's None (shouldn't be).
            # Actually, if it's None, it means no subset was found, which is an error.
            # But let's be safe.
            result.append([])
        else:
            result.append(best_subset_for_res[r])
            
    return result
