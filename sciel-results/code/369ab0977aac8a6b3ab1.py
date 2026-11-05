import numpy as np
import itertools
import time
import random
import sys

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets indices
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
    
    # Precompute block masks as Python integers for fast subset checks
    # bit i is 1 if block covers subset i
    block_masks = [0] * num_blocks
    for b_idx in range(num_blocks):
        mask = 0
        for s_idx in block_to_subsets[b_idx]:
            mask |= (1 << s_idx)
        block_masks[b_idx] = mask
        
    # Helper to iterate set bits of an integer mask
    def iter_bits(mask):
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_mask = (1 << num_subsets) - 1
        gains = np.array([bin(m).count('1') for m in block_masks], dtype=np.int32)
        # Actually, initial gain is just number of subsets in block, which is C(k,t)
        # But some might be "already covered" if we start with non-empty? No, start empty.
        # So gains = [len(block_to_subsets[i]) for i in range(num_blocks)]
        
        selected_blocks = []
        
        while uncovered_mask:
            # Add noise to break ties
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_mask = block_masks[best_idx]
            newly_covered = b_mask & uncovered_mask
            
            if newly_covered == 0:
                continue
                
            uncovered_mask ^= newly_covered
            
            # Update gains
            # For each subset s in newly_covered, decrement gain of all blocks covering s
            for s_idx in iter_bits(newly_covered):
                # subset_to_blocks[s_idx] is a list of block indices
                # We need to decrement gains for these blocks
                # Using numpy slicing is faster
                gains[subset_to_blocks[s_idx]] -= 1
                    
        return selected_blocks

    def minimize(blocks_indices, random_seed=None, deadline=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if not current_blocks:
            return current_blocks
            
        # cover_count[s_idx] = number of blocks covering subset s_idx
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_to_subsets[b_idx]:
                cover_count[s_idx] += 1
                
        block_in_set = np.zeros(num_blocks, dtype=bool)
        for b_idx in current_blocks:
            block_in_set[b_idx] = True

        def remove_redundant():
            changed = True
            while changed:
                changed = False
                redundant_blocks = []
                for i, b_idx in enumerate(current_blocks):
                    is_redundant = True
                    for s_idx in block_to_subsets[b_idx]:
                        if cover_count[s_idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        redundant_blocks.append(i)
                    
                if redundant_blocks:
                    # Remove a random redundant block
                    remove_pos = redundant_blocks[rng.randint(0, len(redundant_blocks)-1)]
                    b_idx_to_remove = current_blocks[remove_pos]
                    for s_idx in block_to_subsets[b_idx_to_remove]:
                        cover_count[s_idx] -= 1
                    block_in_set[b_idx_to_remove] = False
                    current_blocks.pop(remove_pos)
                    changed = True
                    return True
            return False

        def try_swap():
            if not current_blocks:
                return False
            # Pick a random block to try to remove/swap
            i = rng.randint(0, len(current_blocks) - 1)
            b_idx = current_blocks[i]
            
            # Find unique subsets for b_idx
            unique_mask = 0
            for s_idx in block_to_subsets[b_idx]:
                if cover_count[s_idx] == 1:
                    unique_mask |= (1 << s_idx)
            
            if unique_mask == 0:
                # Redundant, remove it
                for s_idx in block_to_subsets[b_idx]:
                    cover_count[s_idx] -= 1
                block_in_set[b_idx] = False
                current_blocks.pop(i)
                return True
            
            # Find a block C that covers all unique subsets
            # Get candidates from the first set bit in unique_mask
            first_bit = None
            temp_mask = unique_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                first_bit = lsb.bit_length() - 1
                break
                
            candidates = subset_to_blocks[first_bit]
            
            best_new = -1
            best_overlap = -1
            b_mask = block_masks[b_idx]
            
            for cand in candidates:
                if block_in_set[cand]:
                    continue
                # Check if cand covers all unique subsets
                if (block_masks[cand] & unique_mask) == unique_mask:
                    # Calculate overlap: how many subsets does cand cover that are already covered?
                    # We want to maximize this to create more redundancies
                    # overlap = popcount(block_masks[cand] & covered_mask)
                    # But we don't have covered_mask as a single int easily updated?
                    # We can compute overlap by iterating subsets of cand?
                    # Or just use a heuristic: prefer cand that shares more subsets with other blocks in the set?
                    # Simple heuristic: count subsets of cand that have cover_count > 0
                    overlap = 0
                    for s_idx in block_to_subsets[cand]:
                        if cover_count[s_idx] > 0:
                            overlap += 1
                            
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_new = cand
            
            if best_new != -1:
                # Perform swap
                for s_idx in block_to_subsets[b_idx]:
                    cover_count[s_idx] -= 1
                block_in_set[b_idx] = False
                
                for s_idx in block_to_subsets[best_new]:
                    cover_count[s_idx] += 1
                block_in_set[best_new] = True
                
                current_blocks[i] = best_new
                return True
                
            return False

        # Main minimize loop
        while True:
            if deadline and time.time() > deadline:
                break
                
            # Try to remove redundant
            if remove_redundant():
                continue
                
            # Try swaps
            improved = False
            for _ in range(10): # Try a few swaps
                if deadline and time.time() > deadline:
                    break
                if try_swap():
                    improved = True
                    break
                # If swap failed, try another block
            if not improved:
                break
                
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    trial = 0
    while time.time() < deadline - 2.0:
        # Strategy 1: Full Greedy + Minimize
        greedy_res = run_greedy_full(trial)
        
        # Minimize with tight deadline
        min_deadline = min(deadline - 1.0, time.time() + 5.0) # Give minimize up to 5s or until global deadline
        min_res = minimize(greedy_res, trial, min_deadline)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        # Strategy 2: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 10.0:
            perturbed = list(best_blocks)
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    for s_idx in block_to_subsets[b_idx]:
                        cover_count_temp[s_idx] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                # Repair using greedy on uncovered
                # We can reuse run_greedy_partial logic or just call run_greedy_full and filter?
                # Better: specific repair
                repair_res = []
                # Simple repair: greedy pick blocks covering uncovered
                current_uncovered = uncovered_mask.copy()
                while np.any(current_uncovered):
                    # Find best block for current_uncovered
                    # Gains relative to current_uncovered
                    # This is slow. Let's just use the full greedy result and intersect?
                    # No, let's use a simple greedy repair
                    gains_repair = np.zeros(num_blocks, dtype=np.int32)
                    for s_idx in np.where(current_uncovered)[0]:
                        gains_repair[subset_to_blocks[s_idx]] += 1
                    
                    if gains_repair.max() == 0: break
                    
                    best_idx = int(np.argmax(gains_repair))
                    repair_res.append(best_idx)
                    
                    for s_idx in block_to_subsets[best_idx]:
                        if current_uncovered[s_idx]:
                            current_uncovered[s_idx] = False
                            gains_repair[subset_to_blocks[s_idx]] -= 1
                            if gains_repair[best_idx] > 0:
                                gains_repair[best_idx] -= 1 # Wait, best_idx is one of the blocks in subset_to_blocks[s_idx]
                                # Actually, the loop above decrements gains for ALL blocks covering s_idx.
                                # So gains_repair[best_idx] is already decremented.
                    
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                # Minimize the new set
                min_deadline2 = min(deadline - 1.0, time.time() + 5.0)
                min_res2 = minimize(new_set, trial + 2000, min_deadline2)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0, deadline - 0.1)
        
    return [list(all_blocks[i]) for i in best_blocks]
