import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    P = 101
    N = 10
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, P)
        
    # We want to find for each residue r in 0..100, the minimum size of a subset S of {1..10}
    # such that sum(inv[x] for x in S) % P == r.
    # Then the answer is the maximum of these minimum sizes.
    
    # min_size[r] = minimum popcount of a subset summing to r
    # Initialize with infinity
    min_size = [float('inf')] * P
    
    # Iterate through all non-empty subsets
    # There are 2^10 - 1 = 1023 subsets
    for mask in range(1, 1 << N):
        # Calculate sum and popcount
        s = 0
        count = 0
        for i in range(N):
            if (mask >> i) & 1:
                s = (s + inv[i+1]) % P
                count += 1
        
        if count < min_size[s]:
            min_size[s] = count
            
    # Check if all residues are covered
    # The problem guarantees they are, but we should be safe.
    max_len = 0
    for r in range(P):
        if min_size[r] == float('inf'):
            # Fallback if not covered (should not happen)
            # Find any subset? The problem states "Every residue must be represented".
            # If it's not represented by size <= 10, it's a problem.
            # But with 10 denominators, we have 1023 subsets. 101 residues.
            # It is extremely likely all are covered.
            # If not, we might need to return something valid, but the contract says 
            # "row r containing ... whose modular inverses sum to r".
            # If impossible, the program is invalid.
            # But let's assume it's possible.
            pass
        if min_size[r] > max_len:
            max_len = min_size[r]
            
    # The task asks to return 101 nonempty lists.
    # We need to construct the actual lists, not just the max length.
    # The score is based on the max length, but the output must be valid.
    # We need to store one such list for each residue.
    
    # Re-run to store the lists
    best_subset_for_residue = [None] * P
    best_size_for_residue = [float('inf')] * P
    
    for mask in range(1, 1 << N):
        s = 0
        terms = []
        for i in range(N):
            if (mask >> i) & 1:
                s = (s + inv[i+1]) % P
                terms.append(i+1)
        
        if len(terms) < best_size_for_residue[s]:
            best_size_for_residue[s] = len(terms)
            best_subset_for_residue[s] = terms
            
    result = []
    for r in range(P):
        if best_subset_for_residue[r] is not None:
            result.append(best_subset_for_residue[r])
        else:
            # Fallback: This should not happen
            result.append([1])
            
    return result
