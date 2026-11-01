import time
import random
import itertools
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-subsets (blocks) and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map block index to its t-subsets indices
    block_tsubsets = []
    # Map t-subset index to list of block indices containing it
    tsubset_to_blocks = defaultdict(list)
    
    for bi, block in enumerate(all_blocks):
        ts_indices = []
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[ts]
            ts_indices.append(idx)
            tsubset_to_blocks[idx].append(bi)
        block_tsubsets.append(ts_indices)
        
    best_blocks = None
    best_count = num_tsubsets
    
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = set(range(num_tsubsets))
        blocks = []
        
        while uncovered and time.time() < deadline - 5:
            # Pick a random uncovered t-subset
            start_idx = rng.choice(list(uncovered))
            
            # Find the best block containing this t-subset
            best_block_idx = -1
            best_gain = -1
            
            for bi in tsubset_to_blocks[start_idx]:
                gain = 0
                for ts_idx in block_tsubsets[bi]:
                    if ts_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    if gain == len(uncovered): # Can't get better
                        break
            
            if best_block_idx < 0:
                break
                
            block = all_blocks[best_block_idx]
            blocks.append(block)
            
            for ts_idx in block_tsubsets[best_block_idx]:
                uncovered.discard(ts_idx)
                
        if not uncovered:
            return [list(b) for b in blocks]
        return None

    def remove_redundant(blocks):
        if not blocks:
            return blocks
            
        # Count coverage
        cover_count = [0] * num_tsubsets
        for block in blocks:
            bi = all_blocks.index(block) # This is slow O(N). Better to map block->idx
            # We need a fast way to get block index.
            # Let's create a set of block indices for the current solution.
            
        # Optimization: Use a set of block indices in the solution.
        # But remove_redundant takes a list of blocks (lists).
        # Let's convert to indices first.
        
        block_indices = []
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        for b in blocks:
            block_indices.append(block_to_idx[tuple(b)])
            
        cover_count = [0] * num_tsubsets
        for bi in block_indices:
            for ts_idx in block_tsubsets[bi]:
                cover_count[ts_idx] += 1
                
        result_indices = list(block_indices)
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(result_indices)):
                bi = result_indices[i]
                can_remove = True
                for ts_idx in block_tsubsets[bi]:
                    if cover_count[ts_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts_idx in block_tsubsets[bi]:
                        cover_count[ts_idx] -= 1
                    result_indices.pop(i)
                    changed = True
                    break # Restart
                    
        return [list(all_blocks[bi]) for bi in result_indices]

    def local_search(blocks, time_limit):
        # Convert to indices
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        current_indices = set(block_to_idx[tuple(b)] for b in blocks)
        
        # Initial cover count
        cover_count = [0] * num_tsubsets
        for bi in current_indices:
            for ts_idx in block_tsubsets[bi]:
                cover_count[ts_idx] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to remove redundant
            # (Same logic as remove_redundant but on indices)
            changed = True
            while changed and time.time() < time_limit:
                changed = False
                for bi in list(current_indices):
                    can_remove = True
                    for ts_idx in block_tsubsets[bi]:
                        if cover_count[ts_idx] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for ts_idx in block_tsubsets[bi]:
                            cover_count[ts_idx] -= 1
                        current_indices.remove(bi)
                        changed = True
                        improved = True
                        break # Restart scan
            
            if time.time() >= time_limit:
                break
                
            # Try swaps
            for bi in list(current_indices):
                if time.time() >= time_limit:
                    break
                # Find critical triples
                criticals = []
                for ts_idx in block_tsubsets[bi]:
                    if cover_count[ts_idx] == 1:
                        criticals.append(ts_idx)
                
                if not criticals:
                    # Should have been removed
                    continue
                    
                # Find a block that covers all criticals
                # Candidate blocks must contain all criticals.
                # Start with blocks containing the first critical
                candidates = tsubset_to_blocks[criticals[0]]
                
                best_swap = -1
                for cand_bi in candidates:
                    if cand_bi == bi:
                        continue
                    if cand_bi in current_indices:
                        continue # Swapping with existing block is just removal if it covers criticals? 
                                 # If cand_bi is in current_indices, and it covers criticals of bi, 
                                 # then bi is redundant. This is handled by the removal loop above.
                                 # So we only look for NEW blocks.
                    
                    # Check if cand_bi covers all criticals
                    cand_set = set(block_tsubsets[cand_bi])
                    if all(c in cand_set for c in criticals):
                        # Check if this swap allows further reduction?
                        # For now, just accept any swap that maintains validity.
                        # But we want to reduce count. A swap keeps count same.
                        # It only helps if it enables future removals.
                        # Heuristic: Prefer swaps that increase redundancy elsewhere?
                        # Simple: Just try it and see if removal reduces count.
                        
                        # Tentative swap
                        current_indices.remove(bi)
                        for ts_idx in block_tsubsets[bi]:
                            cover_count[ts_idx] -= 1
                            
                        current_indices.add(cand_bi)
                        for ts_idx in block_tsubsets[cand_bi]:
                            cover_count[ts_idx] += 1
                            
                        # Try to remove redundant blocks
                        old_size = len(current_indices)
                        changed = True
                        while changed and time.time() < time_limit:
                            changed = False
                            for rbi in list(current_indices):
                                can_remove = True
                                for ts_idx in block_tsubsets[rbi]:
                                    if cover_count[ts_idx] == 1:
                                        can_remove = False
                                        break
                                if can_remove:
                                    for ts_idx in block_tsubsets[rbi]:
                                        cover_count[ts_idx] -= 1
                                    current_indices.remove(rbi)
                                    changed = True
                                    break # Restart
                        
                        if len(current_indices) < old_size:
                            improved = True
                            break # Restart main loop
                        else:
                            # Revert swap
                            current_indices.remove(cand_bi)
                            for ts_idx in block_tsubsets[cand_bi]:
                                cover_count[ts_idx] -= 1
                            
                            current_indices.add(bi)
                            for ts_idx in block_tsubsets[bi]:
                                cover_count[ts_idx] += 1
                            
                        break # Only try one swap per bi to save time?
                        
        return [list(all_blocks[bi]) for bi in current_indices]

    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 10:
            break
            
        # Greedy
        blocks = greedy_cover(seed)
        if blocks is not None:
            # Redundancy
            blocks = remove_redundant(blocks)
            
            # Local Search
            time_left_for_ls = min(remaining - 5, 30)
            if time_left_for_ls > 5:
                blocks = local_search(blocks, time.time() + time_left_for_ls)
                blocks = remove_redundant(blocks) # Final cleanup
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                if best_count <= 76:
                    break # Found optimal or near optimal
            
        seed += 1
        
    if best_blocks is None:
        # Fallback
        return []
        
    return best_blocks
