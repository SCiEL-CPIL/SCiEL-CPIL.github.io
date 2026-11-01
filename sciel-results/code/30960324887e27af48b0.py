import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Pre-compute all blocks and their covered t-subsets
    points = range(v)
    t_subs = list(itertools.combinations(points, t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    blocks = list(itertools.combinations(points, k))
    num_blocks = len(blocks)
    
    # M: (num_blocks, num_t_subs) boolean matrix
    # M[i, j] = 1 if block i covers t-subset j
    # Using uint8 to save memory (1 byte per element)
    M = np.zeros((num_blocks, num_t_subs), dtype=np.uint8)
    
    # Pre-compute indices of covered subsets for each block
    block_indices = []
    for i, block in enumerate(blocks):
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            M[i, idx] = 1
        block_indices.append(M[i].nonzero()[0])
        
    # Helper: For each t-subset, which blocks cover it?
    sub_to_blocks = [[] for _ in range(num_t_subs)]
    for i in range(num_blocks):
        for idx in block_indices[i]:
            sub_to_blocks[idx].append(i)

    def greedy_init():
        counts = np.zeros(num_t_subs, dtype=np.int32)
        used_blocks = []
        used_mask = np.zeros(num_blocks, dtype=bool)
        
        while True:
            uncovered_mask = (counts == 0)
            if not uncovered_mask.any():
                break
            
            # Calculate gains: number of uncovered subsets covered by each block
            # (M & uncovered_mask[:, None]).sum(axis=1)
            # This creates a large temporary array. 
            # Optimization: Only compute for unused blocks? 
            # M is dense in memory, so slicing doesn't save computation much, but we can mask.
            gains = (M & uncovered_mask[:, None]).sum(axis=1)
            
            # Mask out used blocks
            gains[used_mask] = -1
            
            best_block = np.argmax(gains)
            if gains[best_block] == 0:
                # Should not happen if valid covering exists and we haven't used all
                break
                
            used_mask[best_block] = True
            used_blocks.append(best_block)
            counts[block_indices[best_block]] += 1
            
        return used_blocks, counts

    def reduce_blocks(used_blocks, counts):
        # Remove redundant blocks
        changed = True
        while changed:
            changed = False
            new_used = []
            for b in used_blocks:
                # Check if all subsets in b have count > 1
                if np.all(counts[block_indices[b]] > 1):
                    counts[block_indices[b]] -= 1
                    changed = True
                else:
                    new_used.append(b)
            used_blocks = new_used
        return used_blocks, counts

    def find_replacement(used_blocks, counts, b_old):
        # Remove b_old
        counts[block_indices[b_old]] -= 1
        uncovered = np.where(counts == 0)[0]
        
        if len(uncovered) == 0:
            return None, counts, True # Success, no replacement needed
        
        # Find intersection of blocks covering all uncovered subsets
        candidates = set(sub_to_blocks[uncovered[0]])
        
        for u in uncovered[1:]:
            candidates.intersection_update(sub_to_blocks[u])
            if not candidates:
                break
        
        if not candidates:
            # Revert
            counts[block_indices[b_old]] += 1
            return None, counts, False
            
        # Pick a candidate
        # Heuristic: pick one that covers the most *other* subsets?
        # For speed, just pick the first one.
        b_new = next(iter(candidates))
        
        # Add b_new
        counts[block_indices[b_new]] += 1
        
        return b_new, counts, True

    # 1. Greedy Initialization
    used_blocks, counts = greedy_init()
    best_blocks, best_counts = reduce_blocks(used_blocks, counts.copy())
    best_size = len(best_blocks)
    
    # 2. Local Search
    while time.time() < deadline:
        if not best_blocks:
            break
        idx = np.random.randint(len(best_blocks))
        b_old = best_blocks[idx]
        
        # Try replacement
        b_new, new_counts, success = find_replacement(best_blocks, best_counts.copy(), b_old)
        
        if success:
            # Construct new list of blocks
            new_blocks = [b for i, b in enumerate(best_blocks) if i != idx]
            if b_new is not None:
                # Ensure b_new is not already in new_blocks
                if b_new not in new_blocks:
                    new_blocks.append(b_new)
                else:
                    # If b_new is already there, we just removed b_old and added b_new (same as existing)?
                    # No, if b_new is in new_blocks, it means we are duplicating.
                    # This case is invalid.
                    continue
            
            # Reduce
            new_blocks, new_counts = reduce_blocks(new_blocks, new_counts)
            
            if len(new_blocks) < best_size:
                best_blocks = new_blocks
                best_counts = new_counts
                best_size = len(new_blocks)

    # Convert block IDs back to point lists
    return [list(blocks[b]) for b in best_blocks]
