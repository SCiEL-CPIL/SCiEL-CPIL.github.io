import time
from itertools import combinations

def construct():
    """
    Constructs 101 lists of distinct denominators in 1..10 such that the sum of 
    their modular inverses mod 101 equals the row index.
    Minimizes the maximum length of the lists.
    """
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
    # We iterate by size to find smaller subsets first, which allows us to 
    # potentially skip larger sizes if all residues are already covered with smaller ones.
    # However, since we want the GLOBAL minimum per residue, and the total search space is tiny,
    # we can just iterate all masks and update if size is smaller.
    # But to be efficient and clear, let's iterate by size.
    
    # Actually, since we need the minimum size for EACH residue, and a residue might be 
    # reachable by a size-2 subset and also a size-3 subset, we must check all sizes 
    # unless we can prove no smaller subset exists.
    # Given the small number of subsets (1023), we can just do a full scan.
    
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
