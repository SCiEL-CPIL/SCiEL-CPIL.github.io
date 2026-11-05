import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    # Parameters for this task: v=19, k=8, t=5
    # But we should handle general v,k,t if possible, though the task is specific.
    # The prompt says "The evaluator passes this task's parameters v, k and t."
    # So we assume v=19, k=8, t=5.
    
    # Precomputation
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    k_blocks = list(itertools.combinations(points, k))
    
    n_t = len(t_subsets)
    n_k = len(k_blocks)
    
    # Map t-subsets to indices
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # For each k-block, list of t-subset indices it covers
    block_t_indices = []
    for b in k_blocks:
        idxs = []
        for sub in itertools.combinations(b, t):
            idxs.append(t_subset_to_idx[sub])
        block_t_indices.append(idxs)
        
    # For each t-subset, list of k-block indices that cover it
    t_subset_blocks = [[] for _ in range(n_t)]
    for b_idx, idxs in enumerate(block_t_indices):
        for s_idx in idxs:
            t_subset_blocks[s_idx].append(b_idx)
            
    # Greedy Construction
    # uncovered_mask: bit i is 1 if t-subset i is uncovered
    # Python int can hold arbitrary size bits
    full_mask = (1 << n_t) - 1
    uncovered = full_mask
    
    cover_blocks = []
    
    # To speed up greedy, we can just iterate.
    # Optimization: If we use numpy for counts later, we need to be careful.
    # For greedy, bitmasks are good.
    
    while uncovered:
        best_block = -1
        best_gain = 0
        # Iterate all blocks
        for b_idx in range(n_k):
            # Calculate gain: how many uncovered subsets this block covers
            # We need the bitmask of the block's subsets.
            # Precomputing bitmasks for all 75k blocks might use memory but is fast.
            # Let's precompute block_bitmasks.
            pass
        # This loop inside while is too slow if we recompute bitmasks.
        # We must precompute block_bitmasks.
        
    # Precompute block_bitmasks
    block_bitmasks = [0] * n_k
    for b_idx, idxs in enumerate(block_t_indices):
        mask = 0
        for idx in idxs:
            mask |= (1 << idx)
        block_bitmasks[b_idx] = mask
            
    while uncovered:
        best_block = -1
        best_gain = 0
        for b_idx in range(n_k):
            # gain = popcount(uncovered & block_bitmasks[b_idx])
            common = uncovered & block_bitmasks[b_idx]
            gain = common.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
        # If best_gain is 0, we are stuck (should not happen if cover exists)
        if best_gain == 0:
            break
        cover_blocks.append(best_block)
        uncovered &= ~block_bitmasks[best_block]
        
    # Reduction Phase
    # Use numpy array for counts
    counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in cover_blocks:
        for s_idx in block_t_indices[b_idx]:
            counts[s_idx] += 1
            
    # Remove redundant blocks
    # We iterate until no changes
    changed = True
    while changed:
        changed = False
        # Shuffle to avoid bias? Or just iterate.
        # Iterate backwards to safely remove? No, we check validity first.
        for i in range(len(cover_blocks) - 1, -1, -1):
            b_idx = cover_blocks[i]
            # Check if all subsets in this block have count > 1
            # min(counts[block_t_indices[b_idx]]) > 1
            # Using numpy min on a small list is fast
            if np.min(counts[block_t_indices[b_idx]]) > 1:
                # Remove
                for s_idx in block_t_indices[b_idx]:
                    counts[s_idx] -= 1
                cover_blocks.pop(i)
                changed = True
                
    # Local Search / Optimization
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Helper to check if a block can be removed
    def can_remove(b_idx):
        # Returns True if removing b_idx doesn't uncover any subset
        # We check counts BEFORE removal.
        # If min(counts[block_t_indices[b_idx]]) > 1, it can be removed.
        return np.min(counts[block_t_indices[b_idx]]) > 1

    # Helper to find replacement
    def find_replacement(b_idx):
        # Identify subsets that would become uncovered if b_idx is removed
        # These are subsets s in block_t_indices[b_idx] where counts[s] == 1
        critical_subsets = [s for s in block_t_indices[b_idx] if counts[s] == 1]
        if not critical_subsets:
            return None # Can just remove
        
        # Find blocks that cover ALL critical subsets
        # Intersection of t_subset_blocks[s] for s in critical_subsets
        # Start with the list of blocks for the first critical subset
        candidates = set(t_subset_blocks[critical_subsets[0]])
        for s in critical_subsets[1:]:
            candidates.intersection_update(t_subset_blocks[s])
            if not candidates:
                return None
        
        # Filter out blocks already in cover
        cover_set = set(cover_blocks)
        candidates = candidates - cover_set
        
        if not candidates:
            return None
            
        # Pick a candidate
        # We might want to pick one that covers more *other* subsets to help future reductions?
        # Or just random.
        return random.choice(list(candidates))

    while time.time() < time_limit:
        if not cover_blocks:
            break
            
        # Pick a random block to try to remove or replace
        i = random.randint(0, len(cover_blocks) - 1)
        b_idx = cover_blocks[i]
        
        # Tentatively remove
        for s_idx in block_t_indices[b_idx]:
            counts[s_idx] -= 1
            
        # Check validity
        # Find subsets with count 0
        lost = [s for s in block_t_indices[b_idx] if counts[s] == 0]
        
        if not lost:
            # Success, block removed
            cover_blocks.pop(i)
        else:
            # Need replacement
            # Find block covering all 'lost'
            # Note: 'lost' is subset of critical_subsets logic, but here counts are already decremented.
            # So we need blocks that contain all s in 'lost'.
            candidates = set(t_subset_blocks[lost[0]])
            for s in lost[1:]:
                candidates.intersection_update(t_subset_blocks[s])
                if not candidates:
                    break
            
            if candidates:
                # Exclude current cover
                cover_set = set(cover_blocks)
                candidates = candidates - cover_set
                if candidates:
                    new_b = random.choice(list(candidates))
                    # Replace b_idx with new_b
                    cover_blocks[i] = new_b
                    # Update counts: add new_b, b_idx is already "removed" from counts
                    for s_idx in block_t_indices[new_b]:
                        counts[s_idx] += 1
                    # Try to remove redundant blocks again (quick pass)
                    # We can do a limited reduction here
                    for j in range(len(cover_blocks) - 1, -1, -1):
                         if j == i: continue
                         bj = cover_blocks[j]
                         if np.min(counts[block_t_indices[bj]]) > 1:
                             for s_idx in block_t_indices[bj]:
                                 counts[s_idx] -= 1
                             cover_blocks.pop(j)
            else:
                # No replacement found, revert
                for s_idx in block_t_indices[b_idx]:
                    counts[s_idx] += 1
                cover_blocks[i] = b_idx # It was already there, effectively no change

    # Convert indices back to lists of points
    result = [list(k_blocks[b_idx]) for b_idx in cover_blocks]
    return result
