import itertools
import time
import random
import sys

# Precomputation for specific v, k, t
# The task says v, k, t are passed. We assume v=19, k=7, t=4 based on the prompt, 
# but we should make it generic enough or handle the specific case efficiently.
# Given the 120s limit, precomputing for v=19 is fast.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-subsets (blocks)
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    
    # 3. Precompute mask for each block
    # mask[i] is an integer with bits set for t-subsets contained in block i
    # We also store the list of t-subset indices for each block for faster iteration
    block_masks = [0] * n_k
    block_t_indices = [[] for _ in range(n_k)]
    
    # Optimization: For each k-subset, find which t-subsets it contains
    # A block contains a t-subset if t_sub is a subset of k_sub
    # This is O(n_k * n_t * t) which is 50388 * 3876 * 4 ~ 780M ops. Too slow in Python.
    
    # Better: Iterate through t-subsets and add them to all k-subsets that contain them?
    # Or: For each k-subset, generate its t-subsets.
    # Number of t-subsets in a k-subset is C(k, t) = C(7, 4) = 35.
    # Total operations: n_k * C(k, t) = 50388 * 35 = 1.7M. This is very fast.
    
    for i, k_sub in enumerate(k_subs):
        k_set = set(k_sub)
        mask_val = 0
        indices = []
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_index[t_sub]
            mask_val |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask_val
        block_t_indices[i] = indices

    # 4. Greedy Initialization
    cover_count = [0] * n_t
    current_blocks = []
    # To speed up greedy, we can pick the block covering the most uncovered subsets.
    # We maintain a list of uncovered subsets.
    
    # Simple Greedy:
    # While uncovered exists, pick block with max gain.
    # Gain of block b = number of j in block_t_indices[b] where cover_count[j] == 0.
    # This is O(n_k * 35) per step. Number of steps ~ 150. 150 * 50388 * 35 = 260M ops. Might be slow in Python.
    
    # Optimized Greedy:
    # Start with all subsets uncovered.
    # Pick a random uncovered subset s.
    # Among all blocks containing s, pick the one that covers the most *other* uncovered subsets.
    
    uncovered_set = set(range(n_t))
    
    while uncovered_set:
        if time.time() > deadline:
            break
            
        # Pick a random uncovered subset to anchor
        anchor = random.choice(list(uncovered_set))
        
        # Find all blocks that contain this anchor subset
        # We need a mapping from t-subset index to list of block indices containing it.
        # Let's precompute this.
        # t_to_blocks = [[] for _ in range(n_t)]
        # for i in range(n_k):
        #     for idx in block_t_indices[i]:
        #         t_to_blocks[idx].append(i)
        
        # Doing this on the fly for the first time is expensive, but we can do it once.
        # Let's do it now if not done.
        if not hasattr(construct, 't_to_blocks'):
            construct.t_to_blocks = [[] for _ in range(n_t)]
            for i in range(n_k):
                for idx in block_t_indices[i]:
                    construct.t_to_blocks[idx].append(i)
        
        candidates = construct.t_to_blocks[anchor]
        
        best_block = -1
        max_gain = -1
        
        for b_idx in candidates:
            gain = 0
            for idx in block_t_indices[b_idx]:
                if idx in uncovered_set:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = b_idx
        
        if best_block == -1:
            # Should not happen if uncovered_set is not empty
            break
            
        # Add best_block
        current_blocks.append(best_block)
        for idx in block_t_indices[best_block]:
            cover_count[idx] += 1
            if cover_count[idx] == 1:
                uncovered_set.discard(idx)
                
    # 5. Local Search / Optimization
    # Objective: Minimize len(current_blocks).
    # We will try to delete blocks and swap blocks to reduce redundancy.
    
    # Helper to check if a block can be deleted
    def can_delete(b_idx):
        for idx in block_t_indices[b_idx]:
            if cover_count[idx] == 1:
                return False
        return True

    # Helper to check if a swap is valid (old -> new)
    # Valid if all subsets uniquely covered by 'old' are also covered by 'new'
    def is_valid_swap(old_idx, new_idx):
        # Subsets covered by old
        for idx in block_t_indices[old_idx]:
            if cover_count[idx] == 1:
                # This subset is only covered by old. New must cover it.
                # Check if new covers idx
                # We can check bit in mask or just check if idx in block_t_indices[new_idx]
                # Checking bit in mask is faster: (block_masks[new_idx] >> idx) & 1
                if not (block_masks[new_idx] >> idx) & 1:
                    return False
        return True

    # Main optimization loop
    # We use a random walk with deletion attempts.
    
    # Initial cleanup: try to delete any block that can be deleted
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        # Shuffle current blocks to randomize deletion order
        random.shuffle(current_blocks)
        for b_idx in current_blocks[:]:
            if can_delete(b_idx):
                # Remove b_idx
                current_blocks.remove(b_idx)
                for idx in block_t_indices[b_idx]:
                    cover_count[idx] -= 1
                changed = True
                break # Restart loop after a deletion
    
    # Now try swaps to reduce redundancy, then delete again
    # We define a "redundancy score" as sum of max(0, count[i] - 1)
    # We want to minimize block count primarily.
    
    # We will iterate:
    # 1. Try random swaps. Accept if valid.
    #    To guide swaps, we prefer swaps that increase the number of subsets with count >= 2.
    # 2. Try deletions.
    
    # Precompute "unique coverage" count? No, cover_count is dynamic.
    
    # Let's run a loop until time runs out or no improvements for a while.
    # Improvement: reduction in block count.
    
    # We will use a simple random search:
    # Pick a random block in current_blocks.
    # Try to swap it with a random block not in current_blocks.
    # If valid, perform swap.
    # Then try to delete the new block? No, swap keeps count same.
    # We need to create opportunities for deletion.
    
    # Alternative: 
    # While time remains:
    #   1. Attempt deletion of a random block.
    #   2. If failed, attempt a swap that reduces "criticality".
    #      Criticality of a block = number of subsets it uniquely covers.
    #      We want to replace a block with high criticality with one with low criticality,
    #      provided the swap is valid (i.e., the new block covers all the old block's unique subsets).
    #      Wait, if new block covers old block's unique subsets, the swap is valid.
    #      Does it reduce criticality?
    #      Old criticality = C_old.
    #      New criticality = C_new.
    #      If C_new < C_old, we have reduced the number of "locks" on subsets.
    #      This might allow a subsequent deletion of *another* block.
    
    # Let's implement this.
    
    # To make it faster, we can maintain a list of blocks in current_blocks.
    # We need a set for O(1) lookup of what's in current_blocks.
    current_blocks_set = set(current_blocks)
    
    # Optimization loop
    last_improvement = time.time()
    
    while time.time() < deadline:
        # 1. Try to delete a random block
        if current_blocks:
            b_idx = random.choice(current_blocks)
            if can_delete(b_idx):
                current_blocks.remove(b_idx)
                current_blocks_set.remove(b_idx)
                for idx in block_t_indices[b_idx]:
                    cover_count[idx] -= 1
                last_improvement = time.time()
                continue
        
        # 2. Try a swap
        # Pick a random block to replace
        if not current_blocks:
            break
            
        old_idx = random.choice(current_blocks)
        
        # Calculate current criticality of old_idx
        old_crit = 0
        for idx in block_t_indices[old_idx]:
            if cover_count[idx] == 1:
                old_crit += 1
        
        # If old_crit is 0, we can just delete it (handled above).
        # So old_crit > 0.
        
        # We want to find a new_idx such that:
        # a) new_idx not in current_blocks_set
        # b) is_valid_swap(old_idx, new_idx) is True
        # c) new_crit < old_crit (preferable) or new_crit == old_crit (neutral)
        
        # We don't want to scan all 50k blocks. We can sample.
        # Sample 100 random blocks.
        
        found_swap = False
        for _ in range(100):
            new_idx = random.randint(0, n_k - 1)
            if new_idx in current_blocks_set:
                continue
            if new_idx == old_idx:
                continue
            
            if is_valid_swap(old_idx, new_idx):
                # Check new criticality
                new_crit = 0
                for idx in block_t_indices[new_idx]:
                    # Simulate the swap to count new_crit?
                    # No, we can just count how many subsets are uniquely covered by new_idx *after* the swap.
                    # But that requires updating counts.
                    # Simpler: Just accept the swap if it's valid. 
                    # The reduction in criticality is a heuristic benefit.
                    # Actually, if valid, the subsets uniquely covered by old are covered by new.
                    # The new criticality of new_idx is the number of subsets j in new_idx where cover_count[j] == 1 
                    # AND j is NOT in old_idx (or if j is in old_idx, cover_count[j] becomes 1 after removing old, 
                    # so if new covers it, it's not unique to new).
                    # This is getting complex. Let's just accept valid swaps.
                    
                    # Perform Swap
                    current_blocks.remove(old_idx)
                    current_blocks_set.remove(old_idx)
                    for idx in block_t_indices[old_idx]:
                        cover_count[idx] -= 1
                        
                    current_blocks.append(new_idx)
                    current_blocks_set.add(new_idx)
                    for idx in block_t_indices[new_idx]:
                        cover_count[idx] += 1
                        
                    found_swap = True
                    break
        
        if not found_swap:
            # Try a different old_idx
            pass
            # To avoid infinite loop on a bad block, we can just continue.
            # If we can't swap, we might be stuck.
            # We can try to "kick" by adding a random block and deleting another?
            # For now, just continue.
            # If we are stuck, we might need to restart or use a more complex move.
            # Given the time limit, simple random swaps might be enough to find a deletion.
            
    # Final cleanup
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        random.shuffle(current_blocks)
        for b_idx in current_blocks[:]:
            if can_delete(b_idx):
                current_blocks.remove(b_idx)
                for idx in block_t_indices[b_idx]:
                    cover_count[idx] -= 1
                changed = True
                break

    # Convert block indices to lists of points
    result_blocks = [list(k_subs[i]) for i in current_blocks]
    
    # Ensure distinct and sorted
    # k_subs are already distinct and sorted.
    
    return result_blocks

