import numpy as np
import itertools
import time
import random
import heapq

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
    
    # Helper: Greedy Set Cover using bitmasks with priority queue
    def run_greedy():
        covered = 0
        selected = []
        
        # Use a max-heap (negative gain) to efficiently find best block
        # Initially, all blocks have their full gain
        heap = [(-block_masks[b].bit_count(), b) for b in range(num_blocks)]
        heapq.heapify(heap)
        
        while covered != full_mask and heap:
            # Pop until we find a block with positive gain
            best_block = -1
            best_gain = 0
            
            while heap:
                neg_gain, b = heapq.heappop(heap)
                gain = -neg_gain
                # Recompute actual gain since coverage may have changed
                actual_gain = (block_masks[b] & ~covered).bit_count()
                if actual_gain == 0:
                    continue
                best_block = b
                best_gain = actual_gain
                break
            
            if best_block == -1 or best_gain == 0:
                break
                
            selected.append(best_block)
            covered |= block_masks[best_block]
            
            # Re-heapify: we need to update gains for remaining blocks
            # But this is expensive. Instead, we can lazily recompute.
            # For now, we'll just continue with lazy evaluation.
            
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

    # Helper: Local Search (Swap) with 1-swap and 2-swap moves
    def local_search_swap(blocks_indices, max_iterations, rng):
        if not blocks_indices:
            return blocks_indices
            
        # Initialize cover_count
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            cover_count[block_subsets[b_idx]] += 1
            
        current_blocks = list(blocks_indices)
        current_set = set(current_blocks)
        
        for _ in range(max_iterations):
            if not current_blocks:
                break
            
            move_type = rng.random()
            if move_type < 0.3:
                # 1-swap: replace one block with another
                idx = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[idx]
                
                # Find critical subsets (covered only by b_idx)
                criticals = []
                for s_idx in block_subsets[b_idx]:
                    if cover_count[s_idx] == 1:
                        criticals.append(s_idx)
                        
                if len(criticals) == 0:
                    # This block is redundant!
                    cover_count[block_subsets[b_idx]] -= 1
                    current_blocks.pop(idx)
                    current_set.remove(b_idx)
                    continue
                
                # Find candidate blocks that cover ALL criticals
                candidates = set(subset_to_blocks[criticals[0]])
                for s_idx in criticals[1:]:
                    candidates.intersection_update(subset_to_blocks[s_idx])
                    if not candidates:
                        break
                        
                candidates.difference_update(current_set)
                
                if not candidates:
                    continue
                    
                candidates_list = list(candidates)
                new_b_idx = candidates_list[rng.randint(0, len(candidates_list) - 1)]
                
                # Perform swap
                cover_count[block_subsets[b_idx]] -= 1
                current_blocks[idx] = new_b_idx
                current_set.remove(b_idx)
                current_set.add(new_b_idx)
                cover_count[block_subsets[new_b_idx]] += 1
                
            elif move_type < 0.6 and len(current_blocks) >= 2:
                # 2-swap: replace two blocks with two new blocks
                idx1, idx2 = rng.sample(range(len(current_blocks)), 2)
                b1 = current_blocks[idx1]
                b2 = current_blocks[idx2]
                
                # Find critical subsets for both blocks
                criticals1 = [s for s in block_subsets[b1] if cover_count[s] == 1]
                criticals2 = [s for s in block_subsets[b2] if cover_count[s] == 1]
                
                # If either is empty, it's redundant
                if not criticals1:
                    cover_count[block_subsets[b1]] -= 1
                    current_blocks.pop(idx1)
                    current_set.remove(b1)
                    if idx2 > idx1:
                        idx2 -= 1
                    continue
                if not criticals2:
                    cover_count[block_subsets[b2]] -= 1
                    current_blocks.pop(idx2)
                    current_set.remove(b2)
                    continue
                
                # Find candidates for b1
                cand1 = set(subset_to_blocks[criticals1[0]])
                for s in criticals1[1:]:
                    cand1.intersection_update(subset_to_blocks[s])
                    if not cand1:
                        break
                cand1.difference_update(current_set)
                
                # Find candidates for b2
                cand2 = set(subset_to_blocks[criticals2[0]])
                for s in criticals2[1:]:
                    cand2.intersection_update(subset_to_blocks[s])
                    if not cand2:
                        break
                cand2.difference_update(current_set)
                cand2.difference_update(cand1)  # Ensure different blocks
                
                if not cand1 or not cand2:
                    continue
                    
                new_b1 = list(cand1)[rng.randint(0, len(cand1) - 1)]
                new_b2 = list(cand2)[rng.randint(0, len(cand2) - 1)]
                
                # Perform 2-swap
                cover_count[block_subsets[b1]] -= 1
                cover_count[block_subsets[b2]] -= 1
                current_blocks[idx1] = new_b1
                current_blocks[idx2] = new_b2
                current_set.remove(b1)
                current_set.remove(b2)
                current_set.add(new_b1)
                current_set.add(new_b2)
                cover_count[block_subsets[new_b1]] += 1
                cover_count[block_subsets[new_b2]] += 1
                
            else:
                # Perturb: remove a random block and repair with greedy
                idx = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[idx]
                
                # Check if we can remove it (if it's not the only cover for any subset)
                can_remove = True
                for s_idx in block_subsets[b_idx]:
                    if cover_count[s_idx] == 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    cover_count[block_subsets[b_idx]] -= 1
                    current_blocks.pop(idx)
                    current_set.remove(b_idx)
                    
                    # Repair: add blocks until all subsets covered
                    # Find uncovered subsets
                    uncovered = [s for s in range(num_subsets) if cover_count[s] == 0]
                    if uncovered:
                        # Greedy repair
                        active = [b for b in range(num_blocks) if b not in current_set]
                        while uncovered:
                            best_b = -1
                            best_gain = 0
                            for b in active:
                                gain = sum(1 for s in block_subsets[b] if cover_count[s] == 0)
                                if gain > best_gain:
                                    best_gain = gain
                                    best_b = b
                            if best_b == -1 or best_gain == 0:
                                break
                            cover_count[block_subsets[best_b]] += 1
                            current_blocks.append(best_b)
                            current_set.add(best_b)
                            active.remove(best_b)
                            # Recompute uncovered
                            uncovered = [s for s in range(num_subsets) if cover_count[s] == 0]
                            
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
        rng = random.Random(trial + 100)
        
        # Perform local search with more iterations
        ls_res = local_search_swap(best_blocks, max_iterations=50, rng=rng)
        
        # Minimize again
        min_res = minimize(ls_res)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        # Occasionally, do a more aggressive perturb: remove 2-3 blocks and greedy repair
        if trial % 3 == 0 and len(best_blocks) > 5:
            num_remove = min(3, len(best_blocks) - 1)
            remove_indices = random.sample(range(len(best_blocks)), num_remove)
            blocks_to_remove = set(best_blocks[i] for i in remove_indices)
            remaining = [b for b in best_blocks if b not in blocks_to_remove]
            
            # Repair with greedy
            cover_mask = 0
            for b in remaining:
                cover_mask |= block_masks[b]
                
            if cover_mask != full_mask:
                # Greedy repair
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
                    if best_b == -1:
                        break
                    covered |= block_masks[best_b]
                    active.remove(best_b)
                    remaining.append(best_b)
                    
                min_res = minimize(remaining)
                if len(min_res) < best_len:
                    best_len = len(min_res)
                    best_blocks = min_res

        trial += 1

    return [list(all_blocks[i]) for i in best_blocks]
