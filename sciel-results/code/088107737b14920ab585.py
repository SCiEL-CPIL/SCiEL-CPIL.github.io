import time
import random
import itertools

def construct(v, k, t):
    start_time = time.time()
    # Use 115 seconds to be safe against the 120s limit
    deadline = start_time + 115.0
    
    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Coverage array
    coverage = [0] * num_tsubs
    blocks = []
    
    def add_block(block):
        for sub in itertools.combinations(block, t):
            coverage[tsub_to_idx[sub]] += 1
            
    def remove_block(block):
        for sub in itertools.combinations(block, t):
            coverage[tsub_to_idx[sub]] -= 1

    # 1. Greedy Initialization
    # We maintain a set of uncovered subset indices for efficiency
    uncovered_indices = set(range(num_tsubs))
    
    while uncovered_indices:
        if time.time() > deadline:
            break
            
        # Pick an uncovered subset to start a new block
        # Random selection helps avoid deterministic traps
        target_idx = random.choice(list(uncovered_indices))
        target_sub = all_tsubs[target_idx]
        
        block = list(target_sub)
        
        # Greedily extend the block to size k
        while len(block) < k:
            best_gain = -1
            best_point = -1
            
            # Pre-compute (t-1) subsets of current block for gain calculation
            # This is small, so generating on the fly is fine
            block_t_minus_1 = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                for rest in block_t_minus_1:
                    # Form the t-subset
                    sub_tuple = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(sub_tuple)
                    if idx is not None and idx in uncovered_indices:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                # Should not happen if v >= k and we have uncovered subsets
                break
                
            block.append(best_point)
            
            # Update uncovered_indices for the newly added point's contributions
            # We only need to update if we added a point.
            # The subsets formed by the new point and (t-1) old points might be covered.
            # We don't need to update the set immediately for the *next* point selection
            # if we recalculate gain based on the *current* uncovered set.
            # But `uncovered_indices` is used in the gain calc.
            # So we MUST update it after adding the point to the block?
            # No, the point is not in the block yet.
            # Once we add it, the subsets it forms are now "covered" by this block (potentially).
            # But we are still building the block.
            # Actually, the gain calculation assumes the point is NOT in the block.
            # After we pick the best point, we add it.
            # Then we should update `uncovered_indices` to reflect that these subsets are now covered
            # BY THIS BLOCK.
            # However, other blocks might also cover them.
            # `uncovered_indices` means "not covered by ANY block".
            # Since we are constructing a NEW block, and it's not added to `blocks` yet,
            # we can't just remove from `uncovered_indices` unless we know this block is the first to cover it.
            # But we don't know that until we check `coverage`.
            
            # Simpler approach: Don't update `uncovered_indices` during block construction.
            # Just use it to find the *next* block.
            # The gain calculation `if idx in uncovered_indices` is slightly inaccurate
            # because it might count a subset as "gain" even if it's already covered by a previous block
            # (in which case it's not in `uncovered_indices`).
            # Wait, `uncovered_indices` contains indices with `coverage == 0`.
            # If a subset is covered by a previous block, it's not in `uncovered_indices`.
            # So `gain` correctly counts only subsets that are currently UNCOVERED.
            # This is exactly what we want: maximize coverage of currently uncovered subsets.
            
            # So we do NOT update `uncovered_indices` inside the `while len(block) < k` loop.
            # We only update it after the block is complete and added.
            
        # Add the completed block
        blocks.append(block)
        add_block(block)
        
        # Update uncovered_indices
        # We need to remove from `uncovered_indices` any subset that is now covered (count >= 1)
        # Since we only add to coverage, we can just check the subsets of this new block.
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            if coverage[idx] == 1: # Just became covered
                uncovered_indices.discard(idx)

    # 2. Local Search / Optimization
    # Try to reduce the number of blocks
    
    while time.time() < deadline:
        improved = False
        
        # Pass 1: Remove redundant blocks
        # We do this in a loop to catch cascading redundancies
        removed_any = True
        while removed_any:
            removed_any = False
            # Iterate in reverse to make popping easier, or random
            # Random is better for diversity
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                if i >= len(blocks):
                    continue
                
                block = blocks[i]
                is_redundant = True
                for sub in itertools.combinations(block, t):
                    if coverage[tsub_to_idx[sub]] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    remove_block(block)
                    blocks.pop(i)
                    improved = True
                    removed_any = True
                    break # Restart the scan with new list
                
        if improved:
            continue
            
        # Pass 2: Try to replace a block to enable future removals
        # Pick a random block
        if not blocks:
            break
            
        i = random.randint(0, len(blocks) - 1)
        block = blocks[i]
        
        # Temporarily remove
        remove_block(block)
        
        # Find subsets that are now uncovered (were critical)
        critical_subs = []
        for sub in itertools.combinations(block, t):
            if coverage[tsub_to_idx[sub]] == 0:
                critical_subs.append(sub)
        
        if not critical_subs:
            # This means the block was redundant, but Pass 1 missed it?
            # Or maybe it became redundant due to a previous swap?
            # In that case, we successfully removed it.
            improved = True
            break # Restart to check for more removals
            
        # Check if we can cover critical_subs with a single new block
        union_set = set()
        for sub in critical_subs:
            union_set.update(sub)
            
        if len(union_set) > k:
            # Cannot cover with one block
            add_block(block)
            continue
            
        # We have a core set `union_set` of size <= k
        core = list(union_set)
        candidates = [x for x in range(v) if x not in union_set]
        
        if len(candidates) < k - len(core):
            add_block(block)
            continue
            
        # Find best extension
        best_block = None
        best_score = -1
        
        # Generate extensions
        # C(len(candidates), k - len(core))
        # Max C(15, 3) = 455. Fast.
        for ext in itertools.combinations(candidates, k - len(core)):
            new_block = core + list(ext)
            
            # Verify it covers all critical_subs
            valid = True
            for sub in critical_subs:
                if not set(sub).issubset(set(new_block)):
                    valid = False
                    break
            if not valid:
                continue
                
            # Score: number of subsets with coverage == 1 that are covered by new_block
            # These are subsets that are critical for OTHER blocks.
            # Covering them makes those other blocks potentially redundant.
            score = 0
            for sub in itertools.combinations(new_block, t):
                idx = tsub_to_idx[sub]
                if coverage[idx] == 1:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_block = new_block
        
        if best_block is not None:
            # Accept replacement
            blocks[i] = best_block
            add_block(best_block)
            improved = True
            # Break to restart the main loop and check for removals
            break
        else:
            # No good replacement found
            add_block(block)
            # Try a different block in the next iteration

    return blocks
