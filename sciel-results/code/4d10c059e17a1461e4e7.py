import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = range(12)
    blocks = list(combinations(points, 6))
    n_blocks = len(blocks)
    
    # Map 5-subsets to indices
    all_5subs = list(combinations(points, 5))
    n_5subs = len(all_5subs)
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    # Build matrix for DLX
    # Columns are 5-subsets (constraints)
    # Rows are blocks (choices)
    # We need to link rows to columns they satisfy.
    
    # DLX Implementation
    # Node structure: [L, R, U, D, C, R_idx]
    # We use lists for speed.
    
    # Headers for columns
    # 0 is the header of the header
    # 1..n_5subs are column headers
    
    # We will store nodes in a flat list or arrays.
    # Max nodes: n_5subs (headers) + n_blocks * 6 (entries)
    
    max_nodes = n_5subs + n_blocks * 6 + 1
    
    L = [0] * max_nodes
    R = [0] * max_nodes
    U = [0] * max_nodes
    D = [0] * max_nodes
    C = [0] * max_nodes # Column index
    Row = [0] * max_nodes # Block index
    
    # Initialize headers
    # 0 is root
    # 1..n_5subs are columns
    
    # Link headers in a circular list
    for i in range(n_5subs + 1):
        L[i] = i - 1
        R[i] = (i + 1) % (n_5subs + 1)
        U[i] = i
        D[i] = i
        C[i] = i
        Row[i] = -1
        
    node_count = n_5subs + 1
    
    # Add entries
    # For each block, find the 5-subsets it covers
    for b_idx, b in enumerate(blocks):
        # Get 5-subsets of b
        # b is a tuple of 6 points
        # Generate all 5-subsets
        for s in combinations(b, 5):
            c_idx = sub_to_idx[s]
            # Add node to column c_idx
            # Insert at bottom of column c_idx
            new_node = node_count
            node_count += 1
            
            C[new_node] = c_idx
            Row[new_node] = b_idx
            
            # Insert into column list
            # U[new_node] = U[c_idx]
            # D[new_node] = c_idx
            # D[U[c_idx]] = new_node
            # U[c_idx] = new_node
            # Actually, standard DLX inserts at the bottom.
            
            U[new_node] = U[c_idx]
            D[new_node] = c_idx
            D[U[c_idx]] = new_node
            U[c_idx] = new_node
            
            # Insert into row list
            # We need to link nodes in the same row.
            # We can do this later or keep track.
            # Let's keep a list of nodes for each row.
            
    # This manual node management is error-prone and slow in Python.
    # Let's use a simpler representation for DLX or a different algorithm.
    
    # Alternative: Use scipy? No.
    # Alternative: Use a recursive backtracker with bitmasks.
    
    # Bitmask Backtracker for Exact Cover
    # Constraints: 792 5-subsets.
    # We represent the set of covered 5-subsets as a set of integers or a big int.
    # Since 792 bits, big int is good.
    
    # Precompute mask for each block
    block_masks = []
    for b in blocks:
        mask = 0
        for s in combinations(b, 5):
            idx = sub_to_idx[s]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    # Search for a set of blocks that covers all bits exactly once.
    # This is equivalent to finding a set of masks that OR to (1<<792)-1 and are pairwise disjoint.
    
    best_blocks = []
    best_count = 0
    
    # Greedy fallback
    indices = list(range(n_blocks))
    random.shuffle(indices)
    covered = 0
    current_blocks = []
    for i in indices:
        m = block_masks[i]
        if not (covered & m):
            current_blocks.append(i)
            covered |= m
            
    best_blocks = current_blocks
    best_count = len(current_blocks)
    
    # Try to find Exact Cover using Backtracking
    # We want to cover all 792 bits.
    # State: covered_mask, list of selected blocks
    
    # To make it faster, we can use a heuristic:
    # Pick the bit that is covered by the fewest remaining blocks.
    
    # Precompute for each bit (5-subset) which blocks cover it.
    bit_to_blocks = [[] for _ in range(n_5subs)]
    for b_idx, mask in enumerate(block_masks):
        # Extract set bits
        # Since mask has only 6 bits set, this is fast
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            bit_to_blocks[idx].append(b_idx)
            m ^= lsb
            
    # Backtracking function
    def backtrack(covered_mask, selected_blocks, depth):
        nonlocal best_blocks, best_count
        
        if time.time() > deadline:
            return False
            
        # Check if all covered
        if covered_mask == (1 << n_5subs) - 1:
            if len(selected_blocks) > best_count:
                best_blocks = selected_blocks[:]
                best_count = len(selected_blocks)
            return True
            
        # Heuristic: Find the uncovered bit with the fewest options
        # Uncovered bits are those where covered_mask has 0.
        # We need to find an uncovered bit.
        # Since we need to cover ALL bits, we can just pick any uncovered bit.
        # To optimize, pick the one with min options.
        
        # Finding min options efficiently:
        # We can iterate through uncovered bits.
        # There are 792 bits.
        # We can maintain a count of available blocks for each bit?
        # That's expensive to update.
        
        # Simple heuristic: Pick the first uncovered bit.
        # Or, precompute the "degree" of each bit (number of blocks covering it).
        # All 5-subsets are covered by exactly 6 blocks in the full design?
        # In S(5,6,12), each 5-subset is in exactly 1 block.
        # In the set of all 924 blocks, how many cover a specific 5-subset?
        # A 5-subset can be extended to a 6-subset by adding 1 of the remaining 7 points.
        # So there are 7 blocks containing any given 5-subset.
        # So every bit has exactly 7 options initially.
        
        # So the heuristic "min options" is not very discriminating initially.
        # But as we select blocks, options decrease.
        
        # Let's try a simple recursive search with randomization.
        
        # Find an uncovered bit
        # We can just pick the lowest set bit of ~covered_mask
        # But ~covered_mask is infinite in Python.
        # We can use (1 << n_5subs) - 1 ^ covered_mask
        
        remaining_mask = ((1 << n_5subs) - 1) ^ covered_mask
        if remaining_mask == 0:
             # All covered
             if len(selected_blocks) > best_count:
                 best_blocks = selected_blocks[:]
                 best_count = len(selected_blocks)
             return True

        # Pick a bit from remaining_mask
        # To be smart, pick a bit that has fewest available blocks.
        # We can approximate by picking a random bit from remaining_mask?
        # Or the first one?
        
        # Let's pick the first uncovered bit.
        lsb = remaining_mask & -remaining_mask
        bit_idx = lsb.bit_length() - 1
        
        # Get blocks covering this bit
        candidates = bit_to_blocks[bit_idx]
        
        # Shuffle candidates for randomization
        random.shuffle(candidates)
        
        for b_idx in candidates:
            m = block_masks[b_idx]
            # Check if this block conflicts with covered_mask
            if not (covered_mask & m):
                # Select this block
                selected_blocks.append(b_idx)
                # Recurse
                if backtrack(covered_mask | m, selected_blocks, depth + 1):
                    return True
                selected_blocks.pop()
                
        return False
        
    # Run backtrack
    # We might need to restart with different random shuffles if it fails quickly
    
    # Initial call
    backtrack(0, [], 0)
    
    if best_count >= 132:
        return [list(blocks[i]) for i in best_blocks]
        
    # If we didn't find 132, return the best greedy found
    # Note: The backtrack might have found a better greedy-like solution
    # But if it didn't reach 132, it might have timed out.
    
    return [list(blocks[i]) for i in best_blocks]
