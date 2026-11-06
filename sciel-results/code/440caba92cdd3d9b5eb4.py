import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 145.0 # Safety margin

    points = range(12)
    all_5subsets = list(combinations(points, 5))
    num_5subsets = len(all_5subsets)
    
    # Map each 5-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    # Generate all 6-subsets (blocks)
    all_6subsets = list(combinations(points, 6))
    num_6subsets = len(all_6subsets)
    
    # Pre-compute bitsets
    # block_bits[b] = bitmask of 5-subsets covered by block b
    block_bits = [0] * num_6subsets
    for b_idx, b in enumerate(all_6subsets):
        mask = 0
        for s in combinations(b, 5):
            mask |= (1 << subset_to_idx[s])
        block_bits[b_idx] = mask
        
    # candidates_bits[s] = bitmask of blocks covering 5-subset s
    candidates_bits = [0] * num_5subsets
    for b_idx in range(num_6subsets):
        mask = block_bits[b_idx]
        # Iterate through set bits in mask
        m = mask
        while m:
            lsb = m & -m
            s_idx = lsb.bit_length() - 1
            candidates_bits[s_idx] |= (1 << b_idx)
            m ^= lsb

    # conflict_bits[b] = bitmask of blocks conflicting with block b
    # Two blocks conflict if they share a 5-subset.
    # conflict[b] includes b itself? Let's include it, we will mask it out.
    conflict_bits = [0] * num_6subsets
    for b_idx in range(num_6subsets):
        mask = block_bits[b_idx]
        m = mask
        while m:
            lsb = m & -m
            s_idx = lsb.bit_length() - 1
            # All blocks covering s_idx conflict with b_idx
            conflict_bits[b_idx] |= candidates_bits[s_idx]
            m ^= lsb

    # Symmetry breaking: Fix the first block to be all_6subsets[0]
    # This is valid because the automorphism group of S(5,6,12) is transitive on blocks.
    first_block_idx = 0
    initial_uncovered = ((1 << num_5subsets) - 1) & ~block_bits[first_block_idx]
    initial_valid_blocks = ((1 << num_6subsets) - 1) & ~conflict_bits[first_block_idx]
    
    # Ensure the first block is removed from valid blocks (it is, via conflict_bits)
    # But we must also ensure we don't pick it again.
    # conflict_bits[first_block_idx] contains first_block_idx.
    
    best_solution = [all_6subsets[first_block_idx]]
    
    def backtrack(uncovered, valid_blocks, current_blocks):
        nonlocal best_solution
        if time.time() - start_time > timeout:
            return False
            
        # Update best solution
        if len(current_blocks) > len(best_solution):
            best_solution = current_blocks[:]
            
        if uncovered == 0:
            best_solution = current_blocks[:]
            return True
            
        # Heuristic: Pick uncovered 5-subset with min valid candidates
        min_subset = -1
        min_count = float('inf')
        
        # Iterate through set bits in uncovered
        m = uncovered
        while m:
            lsb = m & -m
            s_idx = lsb.bit_length() - 1
            # Count valid blocks covering s_idx
            count = (candidates_bits[s_idx] & valid_blocks).bit_count()
            if count < min_count:
                min_count = count
                min_subset = s_idx
                if count == 0:
                    return False
            m ^= lsb
            
        # Try each valid block covering min_subset
        # We need to iterate through the set bits of (candidates_bits[min_subset] & valid_blocks)
        candidates_mask = candidates_bits[min_subset] & valid_blocks
        m = candidates_mask
        while m:
            lsb = m & -m
            b_idx = lsb.bit_length() - 1
            
            new_uncovered = uncovered & ~block_bits[b_idx]
            new_valid_blocks = valid_blocks & ~conflict_bits[b_idx]
            
            if backtrack(new_uncovered, new_valid_blocks, current_blocks + [all_6subsets[b_idx]]):
                return True
                
            m ^= lsb
            
        return False

    initial_current_blocks = [all_6subsets[first_block_idx]]
    
    if backtrack(initial_uncovered, initial_valid_blocks, initial_current_blocks):
        return [list(b) for b in best_solution]
    else:
        # Return best found
        return [list(b) for b in best_solution]
