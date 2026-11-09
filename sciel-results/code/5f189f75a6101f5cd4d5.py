"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Precompute inverses for 1..10
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
    
    # Generate all non-empty subsets of {1..10}
    # We want to find the minimal K such that all residues are covered by subsets of size <= K
    # Then assign one such subset to each residue.
    
    # To be efficient, we can just generate all subsets, compute their sums and sizes,
    # and for each residue, keep the subset with the smallest size.
    # If there's a tie in size, we can keep the first one found or any.
    
    best_subset_for_residue = [None] * 101
    
    # Iterate through all masks
    for mask in range(1, 1 << 10):
        # Time check
        if time.time() > deadline:
            # If we run out of time, return what we have if valid, else fallback
            # But we should finish quickly. 1024 iterations is trivial.
            pass
            
        # Calculate sum and size
        s = 0
        size = 0
        m = mask
        # We can precompute popcount and sum for all masks to be faster, 
        # but 1024 is small enough.
        # Let's just compute.
        terms = []
        for i in range(1, 11):
            if mask & (1 << (i - 1)):
                s = (s + inv[i]) % 101
                size += 1
                terms.append(i)
        
        residue = s
        # We want to minimize the size.
        # If we haven't seen this residue, or we found a smaller size, update.
        if best_subset_for_residue[residue] is None:
            best_subset_for_residue[residue] = terms
        else:
            current_size = len(best_subset_for_residue[residue])
            if size < current_size:
                best_subset_for_residue[residue] = terms

    # Check if all residues are covered
    # If some are None, it means no subset covers them? 
    # But the problem implies a solution exists.
    # With 1023 subsets, it's extremely likely all are covered.
    # If not, we might need to relax "minimize size" to just "find any".
    # But the seed found all, so we will too.
    
    # Construct result
    result = []
    for r in range(101):
        if best_subset_for_residue[r] is None:
            # Fallback: should not happen
            # Find any subset that sums to r
            for mask in range(1, 1 << 10):
                s = sum(inv[i] for i in range(1, 11) if mask & (1 << (i - 1))) % 101
                if s == r:
                    terms = [i for i in range(1, 11) if mask & (1 << (i - 1))]
                    best_subset_for_residue[r] = terms
                    break
        result.append(best_subset_for_residue[r])
        
    return result
# EVOLVE-BLOCK-END
