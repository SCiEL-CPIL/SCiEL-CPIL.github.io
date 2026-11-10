import time

def construct():
    # Deadline guard
    start_time = time.time()
    TIME_LIMIT = 90.0
    SAFETY_MARGIN = 5.0

    # Precompute modular inverses for denominators 1..10
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)

    # We need to find, for each residue r in 0..100, a non-empty subset of {1..10}
    # such that the sum of inverses modulo 101 equals r.
    # We want to minimize the maximum size of these subsets.
    # This is equivalent to finding, for each r, the minimum size subset that sums to r.
    
    # Precompute sum and size for every mask from 1 to 1023.
    # mask_info[mask] = (sum_mod_101, size)
    mask_info = [None] * 1024
    mask_info[0] = (0, 0)
    
    for mask in range(1, 1024):
        lsb = mask & -mask
        i = (lsb.bit_length() - 1)  # index 0..9 corresponds to denominator i+1
        prev_mask = mask ^ lsb
        prev_sum, prev_size = mask_info[prev_mask]
        denom = i + 1
        new_sum = (prev_sum + inv[denom]) % 101
        new_size = prev_size + 1
        mask_info[mask] = (new_sum, new_size)
        
    # For each residue, find the minimum size subset that maps to it.
    min_size_for_residue = [float('inf')] * 101
    best_mask_for_residue = [0] * 101
    
    for mask in range(1, 1024):
        s, sz = mask_info[mask]
        if sz < min_size_for_residue[s]:
            min_size_for_residue[s] = sz
            best_mask_for_residue[s] = mask
            
    # Check time limit (though this should be extremely fast)
    if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
        # If we ran out of time (unlikely), we return what we have so far
        # But since the loop is small, this is just a safety net.
        pass

    # Construct the lists.
    result = [[] for _ in range(101)]
    for r in range(101):
        mask = best_mask_for_residue[r]
        terms = []
        for i in range(10):
            if mask & (1 << i):
                terms.append(i + 1)
        result[r] = terms
        
    return result
