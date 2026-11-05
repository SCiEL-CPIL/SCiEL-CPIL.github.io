import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n_points = 12
    block_size = 6
    subset_size = 5
    
    # Generate all 5-subsets and map them to indices
    # There are C(12, 5) = 792 subsets
    all_5_subsets = list(combinations(range(n_points), subset_size))
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # Generate all 6-subsets (blocks) and their corresponding 5-subset masks
    all_blocks = list(combinations(range(n_points), block_size))
    block_masks = []
    for b in all_blocks:
        mask = 0
        for s in combinations(b, subset_size):
            idx = subset_to_idx[s]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    # Target mask: all 5-subsets covered
    target_mask = (1 << len(all_5_subsets)) - 1
    
    # Heuristic: Sort blocks by popcount? All have popcount 6.
    # Maybe sort by the index of the first bit? Or just use the order.
    # To improve performance, we can sort blocks such that those with 'rarer' combinations come first?
    # Actually, the exact cover problem is symmetric. The choice of the first bit (LSB) drives the search.
    
    # Precompute for each bit (5-subset), the list of block indices that cover it
    bit_to_blocks = [[] for _ in range(len(all_5_subsets))]
    for i, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            bit_to_blocks[bit_idx].append(i)
            m ^= lsb
            
    # Backtracking search
    # We need to find 132 blocks
    max_blocks = 132
    
    def search(covered_mask, depth, current_blocks):
        if time.time() > deadline:
            return None
        
        if covered_mask == target_mask:
            if depth == max_blocks:
                return current_blocks[:]
            else:
                return None # Should not happen if valid
        
        if depth == max_blocks:
            return None
            
        # Find an uncovered bit
        uncovered = target_mask ^ covered_mask
        if uncovered == 0:
            return None if depth != max_blocks else current_blocks[:]
            
        # Pick the least significant set bit in uncovered
        lsb = uncovered & -uncovered
        bit_idx = lsb.bit_length() - 1
        
        # Get candidates for this bit
        candidates = bit_to_blocks[bit_idx]
        
        # Heuristic: Sort candidates by how many *other* uncovered bits they cover?
        # This is expensive to compute dynamically.
        # Simple heuristic: Iterate in order.
        
        for b_idx in candidates:
            b_mask = block_masks[b_idx]
            # Check if block conflicts with already covered bits
            if (b_mask & covered_mask) == 0:
                current_blocks.append(b_idx)
                res = search(covered_mask | b_mask, depth + 1, current_blocks)
                if res is not None:
                    return res
                current_blocks.pop()
                
        return None

    solution_indices = search(0, 0, [])
    
    if solution_indices is None:
        # Fallback: Return a greedy result if search fails or times out
        # But the search should succeed.
        # For safety, if it fails, we can return a smaller valid packing.
        # However, the problem states 132 is attainable.
        # Let's just return an empty list or a greedy one if it fails.
        # Given the constraints, it should find it.
        
        # Fallback greedy
        covered = 0
        greedy_blocks = []
        # Sort blocks by some heuristic?
        for i in range(len(block_masks)):
            if (block_masks[i] & covered) == 0:
                greedy_blocks.append(i)
                covered |= block_masks[i]
        solution_indices = greedy_blocks

    # Convert indices back to lists of points
    result = [list(all_blocks[i]) for i in solution_indices]
    return result
