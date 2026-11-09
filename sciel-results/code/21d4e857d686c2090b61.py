import numpy as np

def construct():
    # Precompute inverses
    inverses = np.array([pow(x, -1, 101) for x in range(1, 11)])
    
    # We want to find, for each residue r, the subset of {1..10} with minimum size
    # such that sum(inverses[subset]) % 101 == r.
    
    # Since 2^10 = 1024, we can just enumerate all masks.
    # This is technically "Exhaustive Search", but we structure it to find min size per residue.
    
    min_size_for_residue = [11] * 101
    best_subset_for_residue = [None] * 101
    
    for mask in range(1, 1 << 10):
        # Calculate size
        size = bin(mask).count('1')
        
        # Calculate sum
        # Using numpy for speed, though loop is also fine
        # indices = [i for i in range(10) if (mask >> i) & 1]
        # s = sum(inverses[i] for i in indices) % 101
        
        # Efficient way:
        # We can precompute sums for all masks using DP or just sum.
        # Given small size, simple sum is fine.
        terms = [x for x in range(1, 11) if (mask >> (x-1)) & 1]
        residue = sum(inverses[x-1] for x in terms) % 101
        
        if size < min_size_for_residue[residue]:
            min_size_for_residue[residue] = size
            best_subset_for_residue[residue] = terms
    
    # The max row length is the max of min_size_for_residue
    # We return the best subsets
    
    return [best_subset_for_residue[r] for r in range(101)]
