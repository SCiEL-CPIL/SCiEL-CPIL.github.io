import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Precomputation
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    t_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # Map each k-subset to a frozenset of t-subset indices it covers
    k_subsets = list(itertools.combinations(points, k))
    block_masks = []
    for block in k_subsets:
        mask = frozenset(t_to_idx[s] for s in itertools.combinations(block, t))
        block_masks.append(mask)
        
    # 2. Greedy Initialization
    # counts[i] is the number of blocks covering t-subset i
    counts = [0] * num_t
    blocks = []
    # Set of currently uncovered t-subsets
    uncovered = set(range(num_t))
    
    while uncovered and time.time() < deadline - 5.0:
        best_block_idx = -1
        best_gain = 0
        # Find block with max gain
        # Optimization: If we have many blocks, this loop is the bottleneck.
        # But num blocks is ~50k, and intersection is fast.
        for i, mask in enumerate(block_masks):
            # gain = number of subsets in mask that are in uncovered
            # Using set intersection
            gain = len(mask & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                if gain == len(mask): # Perfect coverage of this block's subsets
                    break
        
        if best_block_idx == -1:
            break
            
        # Add the block
        blocks.append(best_block_idx)
        mask = block_masks[best_block_idx]
        for idx in mask:
            counts[idx] += 1
            if counts[idx] == 1:
                uncovered.discard(idx)
        
        # Early exit if all covered
        if not uncovered:
            break

    # 3. Local Search (Minimization)
    # We will try to reduce the number of blocks.
    # Strategy: Add a block, then remove as many redundant blocks as possible.
    # If net change is negative, keep. Else revert.
    
    # Helper to check if a block is redundant
    # A block is redundant if for all s in mask, counts[s] > 1
    def is_redundant(block_idx):
        mask = block_masks[block_idx]
        for s in mask:
            if counts[s] == 1:
                return False
        return True

    def try_remove_redundant():
        """Remove any block that is redundant. Returns number removed."""
        removed = 0
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if is_redundant(blocks[i]):
                    # Remove block i
                    mask = block_masks[blocks[i]]
                    for s in mask:
                        counts[s] -= 1
                    blocks.pop(i)
                    removed += 1
                    changed = True
        return removed

    # Initial cleanup
    try_remove_redundant()

    best_size = len(blocks)
    
    # Local Search Loop
    while time.time() < deadline:
        # Choose a block to add
        # Heuristic: Add a block that covers the most "critical" subsets (count=1)
        # This increases their count to 2, potentially making the block covering them redundant.
        # If no critical subsets, add a random block.
        
        critical_counts = [0] * len(block_masks)
        # This calculation is O(N_blocks * N_mask). 
        # We can optimize by just picking a random block if we want speed, 
        # but targeting critical subsets is better.
        
        # To save time, we might just pick a random block from the universe of blocks.
        # Or, pick a block that shares subsets with existing blocks.
        
        # Let's try: Pick a random block index from 0..len(block_masks)-1
        add_idx = random.randint(0, len(block_masks) - 1)
        
        # State before
        old_blocks = blocks[:]
        old_counts = counts[:]
        
        # Add the block
        blocks.append(add_idx)
        mask = block_masks[add_idx]
        for s in mask:
            counts[s] += 1
            
        # Try to remove redundant blocks
        # We specifically want to remove the block that was "forcing" the critical subsets
        # But the generic is_redundant check handles this.
        try_remove_redundant()
        
        if len(blocks) < len(old_blocks):
            best_size = len(blocks)
            # Keep the change
            # We can also try to be even more aggressive:
            # If we removed more than 1, great.
        else:
            # Revert
            blocks = old_blocks
            counts = old_counts

    # Final cleanup just in case
    try_remove_redundant()
    
    # Convert block indices back to lists of points
    result = [sorted(k_subsets[i]) for i in blocks]
    
    return result
