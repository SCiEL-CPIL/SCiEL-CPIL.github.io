import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map to indices
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # 2. Generate all k-blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # 3. Precompute block masks (bitsets) and subset indices
    # block_masks[i] is an integer where bit j is 1 if block i covers subset j
    block_masks = [0] * num_blocks
    block_subsets = [[] for _ in range(num_blocks)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        subs = []
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            mask |= (1 << s_idx)
            subs.append(s_idx)
        block_masks[b_idx] = mask
        block_subsets[b_idx] = subs
        
    # 4. Precompute subset_to_blocks for swap search
    # subset_to_blocks[s_idx] is a list of block indices covering subset s_idx
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx in range(num_blocks):
        for s_idx in block_subsets[b_idx]:
            subset_to_blocks[s_idx].append(b_idx)
            
    # Convert subset_to_blocks to numpy arrays for faster intersection? 
    # Actually, lists are fine for small intersections.
    
    full_mask = (1 << num_subsets) - 1
    
    # Helper: Greedy Set Cover using bitmasks
    def run_greedy():
        covered = 0
        selected = []
        # We iterate through all blocks. To speed up, we could maintain a list of active blocks,
        # but removing from a list is O(N). 
        # Given 75k blocks and ~300 iterations, 22M checks is acceptable.
        # Each check is: (mask & ~covered).bit_count()
        
        active_blocks = list(range(num_blocks))
        
        while covered != full_mask and active_blocks:
            best_block = -1
            best_gain = 0
            # Optimization: If we have many blocks, we can sample? 
            # No, greedy is deterministic. 
            # To speed up, we can skip blocks that are subsets of others? 
            # For now, linear scan.
            for b in active_blocks:
                # If block is already fully covered, gain is 0
                # mask & ~covered gives the new bits
                gain = (block_masks[b] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    
            if best_block == -1 or best_gain == 0:
                break
                
            selected.append(best_block)
            covered |= block_masks[best_block]
            
            # Remove best_block from active_blocks
            # This is O(N). 
            # Alternative: Use a set? Or just mark as inactive?
            # Let's use a list and remove. 
            # To avoid O(N^2) total if we remove many, we can just iterate all blocks 
            # and check if they are in selected? No, that's also O(N).
            # Actually, we can just keep a list of all blocks and filter?
            # Let's stick to removing from list. 300 * 75k / 2 ~ 11M operations. Fine.
            active_blocks.remove(best_block)
            
        return selected

    # Helper: Minimize (remove redundant blocks)
    def minimize(blocks_indices):
        if not blocks_indices:
            return blocks_indices
            
        # Initialize cover_count
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            # Add counts for this block
            cover_count[block_subsets[b_idx]] += 1
            
        current_blocks = list(blocks_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks)):
                b_idx = current_blocks[i]
                # Check if any subset in this block is covered ONLY by this block
                is_redundant = True
                for s_idx in block_subsets[b_idx]:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                        
                if is_redundant:
                    # Remove block
                    cover_count[block_subsets[b_idx]] -= 1
                    current_blocks.pop(i)
                    changed = True
                    break # Restart loop to check new indices
                    
        return current_blocks

    # Helper: Local Search (Swap)
    def local_search_swap(blocks_indices, max_iterations, rng):
        if not blocks_indices:
            return blocks_indices
            
        # Initialize cover_count
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            cover_count[block_subsets[b_idx]] += 1
            
        current_blocks = list(blocks_indices)
        current_set = set(current_blocks)
        
        improved = False
        for _ in range(max_iterations):
            if not current_blocks:
                break
                
            # Pick a random block to potentially swap
            idx = rng.randint(0, len(current_blocks) - 1)
            b_idx = current_blocks[idx]
            
            # Find critical subsets (covered only by b_idx)
            criticals = []
            for s_idx in block_subsets[b_idx]:
                if cover_count[s_idx] == 1:
                    criticals.append(s_idx)
                    
            if len(criticals) == 0:
                # This block is redundant!
                # Remove it
                cover_count[block_subsets[b_idx]] -= 1
                current_blocks.pop(idx)
                current_set.remove(b_idx)
                improved = True
                continue
            
            # Find candidate blocks that cover ALL criticals
            # Start with blocks covering the first critical
            candidates = set(subset_to_blocks[criticals[0]])
            for s_idx in criticals[1:]:
                candidates.intersection_update(subset_to_blocks[s_idx])
                if not candidates:
                    break
                    
            # Remove blocks already in current solution
            candidates.difference_update(current_set)
            
            if not candidates:
                continue
                
            # Pick a random candidate
            candidates_list = list(candidates)
            new_b_idx = candidates_list[rng.randint(0, len(candidates_list) - 1)]
            
            # Perform swap
            # Remove old
            cover_count[block_subsets[b_idx]] -= 1
            current_blocks[idx] = new_b_idx
            current_set.remove(b_idx)
            current_set.add(new_b_idx)
            # Add new
            cover_count[block_subsets[new_b_idx]] += 1
            improved = True
            
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Phase 1: Initial Greedy
    greedy_res = run_greedy()
    min_res = minimize(greedy_res)
    best_blocks = min_res
    best_len = len(min_res)
    
    # Phase 2: Iterative Local Search
    trial = 0
    while time.time() < deadline:
        # Perturb: Remove some random blocks and repair
        # Or just do swaps on the current best?
        # Let's try swaps on the current best first.
        rng = random.Random(trial + 100)
        
        # Perform a batch of swaps
        ls_res = local_search_swap(best_blocks, max_iterations=20, rng=rng)
        
        # Minimize again
        min_res = minimize(ls_res)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            # If we improved, maybe do more swaps?
            
        # Occasionally, do a more aggressive perturb: remove 2-3 blocks and greedy repair
        if trial % 5 == 0:
            num_remove = random.randint(1, 3)
            if len(best_blocks) > num_remove:
                remove_indices = random.sample(range(len(best_blocks)), num_remove)
                blocks_to_remove = set(best_blocks[i] for i in remove_indices)
                remaining = [b for b in best_blocks if b not in blocks_to_remove]
                
                # Repair with greedy
                # We need to find blocks to cover the missing subsets
                # Calculate current coverage
                cover_mask = 0
                for b in remaining:
                    cover_mask |= block_masks[b]
                    
                missing_mask = full_mask & ~cover_mask
                
                if missing_mask != 0:
                    # Greedy repair
                    repair_blocks = []
                    # Find best blocks for missing
                    # We can reuse the greedy logic but restricted to missing?
                    # Simpler: just run greedy on the whole problem but start with 'remaining'
                    # This is expensive. 
                    # Alternative: Just add blocks until covered.
                    # Let's just use the standard greedy but initialize 'covered' with 'cover_mask'
                    # and 'active_blocks' with all blocks not in 'remaining'
                    
                    # To keep it simple and fast, let's just add random blocks until covered?
                    # No, that's bad.
                    
                    # Let's use a simplified greedy repair
                    covered = cover_mask
                    active = [b for b in range(num_blocks) if b not in set(remaining)]
                    while covered != full_mask and active:
                         best_b = -1
                         best_gain = 0
                         for b in active:
                             gain = (block_masks[b] & ~covered).bit_count()
                             if gain > best_gain:
                                 best_gain = gain
                                 best_b = b
                         if best_b == -1: break
                         repair_blocks.append(best_b)
                         covered |= block_masks[best_b]
                         active.remove(best_b)
                         
                    new_blocks = remaining + repair_blocks
                    min_res = minimize(new_blocks)
                    if len(min_res) < best_len:
                        best_len = len(min_res)
                        best_blocks = min_res

        trial += 1

    return [list(all_blocks[i]) for i in best_blocks]
