import time

def construct():
    start_time = time.time()
    
    # Precompute inverses
    inverses = [0] * 11
    for i in range(1, 11):
        inverses[i] = pow(i, -1, 101)
        
    # Generate all non-empty subsets
    # subset: list of terms
    # residue: sum of inverses mod 101
    # size: len(terms)
    
    # We want to find min K such that all residues 0..100 are covered by subsets of size <= K
    # Store for each residue the minimum size subset that covers it?
    # No, we just need to know if ANY subset of size <= K covers it.
    # Let's store min_size_for_residue[r] = minimum size of any subset that sums to r.
    # If min_size_for_residue[r] is infinity, it's not covered.
    # Then K_min = max(min_size_for_residue[r] for r in 0..100).
    # Then for each r, we need to pick a subset. 
    # Wait, if we pick the subset with min_size, the max size in the result is max(min_size_for_residue[r]).
    # This is exactly K_min.
    
    min_size = [11] * 101  # 11 is impossible, max size is 10
    # We also need to store the actual subset to return it.
    # best_subset[r] = the subset with the minimum size. If multiple, pick any.
    best_subset = [None] * 101
    
    # Iterate through all masks
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        # Extract bits
        # Optimization: precompute terms and sum for all masks?
        # 1023 masks is small.
        
        # Efficient bit iteration
        m = mask
        t_list = []
        sum_inv = 0
        while m:
            # find least significant set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            x = idx + 1
            t_list.append(x)
            sum_inv += inverses[x]
            m ^= lsb
            
        res = sum_inv % 101
        size = len(t_list)
        
        if size < min_size[res]:
            min_size[res] = size
            best_subset[res] = t_list
            
    # Check if all residues are covered
    # If any min_size[r] is 11, it means no subset covers r.
    # But with 1023 subsets and 101 residues, it's extremely likely all are covered.
    # If not, we would need to handle it, but the problem implies a solution exists.
    
    # The max row length will be max(min_size)
    max_len = max(min_size)
    
    # Construct result
    result = [best_subset[r] for r in range(101)]
    
    return result
