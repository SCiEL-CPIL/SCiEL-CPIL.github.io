import itertools
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, find the indices of the t-subsets it covers
    t_subsets_per_block = len(list(itertools.combinations(range(k), t)))
    BlockTsets = np.zeros((num_blocks, t_subsets_per_block), dtype=np.int32)
    
    # Inverted index: tset index -> list of block indices that cover it
    # This is crucial for fast local search (finding blocks that cover a set of t-sets)
    tset_to_blocks = [np.array([], dtype=np.int32) for _ in range(num_tsets)]
    
    for i, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            idx = tset_to_idx[sub]
            BlockTsets[i, len(list(itertools.combinations(block, t))) - 1] = idx # Placeholder, we fill below properly
            # Actually, let's fill BlockTsets and tset_to_blocks in one pass correctly
            pass
            
    # Re-do the filling correctly
    BlockTsets = np.zeros((num_blocks, t_subsets_per_block), dtype=np.int32)
    # Initialize lists for inverted index
    # We can't easily append to numpy arrays, so we use lists first or pre-allocate if we know counts.
    # Count how many blocks cover each tset?
    # Each tset is covered by C(v-t, k-t) blocks.
    # C(19-5, 8-5) = C(14, 3) = 364.
    # So we can pre-allocate.
    tset_cover_count = [0] * num_tsets
    for i in range(num_blocks):
        for j in range(t_subsets_per_block):
             # We need to fill BlockTsets first to know which tsets
             pass
             
    # Let's just build lists and convert to arrays at the end
    tset_to_blocks_list = [[] for _ in range(num_tsets)]
    
    for i, block in enumerate(all_blocks):
        count = 0
        for sub in itertools.combinations(block, t):
            idx = tset_to_idx[sub]
            BlockTsets[i, count] = idx
            tset_to_blocks_list[idx].append(i)
            count += 1
            
    # Convert to numpy arrays for faster intersection
    tset_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in tset_to_blocks_list]

    # 2. Polish function: Remove redundant blocks
    def polish_blocks(blocks, cover_count=None):
        if not blocks:
            return blocks, np.zeros(num_tsets, dtype=np.int32)
        
        if cover_count is None:
            cover_count = np.zeros(num_tsets, dtype=np.int32)
            for b in blocks:
                cover_count[BlockTsets[b]] += 1
        else:
            # Ensure cover_count is up to date
            pass
            
        # Iterate until no more removals
        changed = True
        while changed:
            changed = False
            i = len(blocks) - 1
            while i >= 0:
                b = blocks[i]
                can_remove = True
                for ts in BlockTsets[b]:
                    if cover_count[ts] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts in BlockTsets[b]:
                        cover_count[ts] -= 1
                    blocks.pop(i)
                    changed = True
                else:
                    i -= 1
                    
        return blocks, cover_count

    # 3. Local Search: 1-1 Swap
    def local_search(blocks, cover_count, max_iters=1000):
        """
        Tries to swap blocks to enable further reductions.
        """
        current_size = len(blocks)
        block_set = set(blocks)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Check for removals first (Polish)
            blocks, cover_count = polish_blocks(blocks, cover_count)
            if len(blocks) < current_size:
                current_size = len(blocks)
                block_set = set(blocks)
                continue
            
            # If no removals, try a swap
            if not blocks:
                break
                
            # Pick a random block
            idx = np.random.randint(len(blocks))
            b_i = blocks[idx]
            
            # Find unique t-sets for b_i
            unique_ts = []
            for ts in BlockTsets[b_i]:
                if cover_count[ts] == 1:
                    unique_ts.append(ts)
                    
            if not unique_ts:
                # Should have been removed by polish
                continue
                
            # Find candidates that cover all unique_ts
            # Intersect tset_to_blocks_arr[ts] for all ts in unique_ts
            candidates = tset_to_blocks_arr[unique_ts[0]]
            for ts in unique_ts[1:]:
                candidates = np.intersect1d(candidates, tset_to_blocks_arr[ts])
                
            if len(candidates) == 0:
                continue
                
            # Filter out blocks already in the solution
            # We need to check if candidate is in block_set
            # Since block_set is a set of ints, we can iterate candidates
            valid_candidates = []
            for c in candidates:
                if c not in block_set:
                    valid_candidates.append(c)
                    
            if not valid_candidates:
                continue
                
            # Pick a random valid candidate
            b_j = valid_candidates[np.random.randint(len(valid_candidates))]
            
            # Perform Swap: remove b_i, add b_j
            blocks[idx] = b_j
            block_set.remove(b_i)
            block_set.add(b_j)
            
            # Update cover_count
            # Decrement for b_i
            for ts in BlockTsets[b_i]:
                cover_count[ts] -= 1
            # Increment for b_j
            for ts in BlockTsets[b_j]:
                cover_count[ts] += 1
                
        return blocks, cover_count

    # 4. Main Loop: Greedy + Local Search
    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline:
        # Greedy Construction
        uncovered = np.ones(num_tsets, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Calculate gains
            gains = uncovered[BlockTsets].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_block_idx = candidates[np.random.randint(len(candidates))]
            blocks.append(best_block_idx)
            uncovered[BlockTsets[best_block_idx]] = False
            
        # Initial Polish
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1
            
        blocks, cover_count = polish_blocks(blocks, cover_count)
        
        # Local Search
        blocks, cover_count = local_search(blocks, cover_count, max_iters=200)
        
        # Final Polish
        blocks, cover_count = polish_blocks(blocks, cover_count)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            
    if best_blocks is None:
        return []
        
    result = [list(all_blocks[b]) for b in best_blocks]
    return result
