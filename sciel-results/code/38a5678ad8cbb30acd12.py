import itertools
import time
import random
import sys
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute masks and subset indices for each block
    block_masks = []
    block_subset_indices = []
    
    for block in all_blocks:
        mask = 0
        subs_idx = []
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
            subs_idx.append(idx)
        block_masks.append(mask)
        block_subset_indices.append(subs_idx)
    
    # Precompute for each t-subset, which blocks cover it
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for s_idx in block_subset_indices[i]:
            subset_to_blocks[s_idx].append(i)
    
    def greedy_repair(uncovered_mask, existing_blocks=None):
        if existing_blocks is None:
            existing_blocks = []
        else:
            existing_blocks = list(existing_blocks)
            
        if not uncovered_mask:
            return existing_blocks

        # Use a heap to speed up finding the best block
        heap = []
        for i in range(num_blocks):
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > 0:
                heapq.heappush(heap, (-gain, i))
        
        while uncovered_mask and heap:
            neg_gain, block_idx = heapq.heappop(heap)
            gain = -neg_gain
            
            actual_gain = (uncovered_mask & block_masks[block_idx]).bit_count()
            
            if actual_gain != gain:
                if actual_gain > 0:
                    heapq.heappush(heap, (-actual_gain, block_idx))
                continue
            
            if actual_gain == 0:
                continue
                
            existing_blocks.append(block_idx)
            uncovered_mask &= ~block_masks[block_idx]
            
        return existing_blocks

    def reduce_blocks(block_indices):
        if not block_indices:
            return []
            
        cover_count = [0] * num_t_subsets
        
        for idx in block_indices:
            for s_idx in block_subset_indices[idx]:
                cover_count[s_idx] += 1
                
        new_blocks = []
        for idx in block_indices:
            subs = block_subset_indices[idx]
            for s_idx in subs:
                cover_count[s_idx] -= 1
                
            is_redundant = True
            for s_idx in subs:
                if cover_count[s_idx] == 0:
                    is_redundant = False
                    break
                    
            if is_redundant:
                pass
            else:
                for s_idx in subs:
                    cover_count[s_idx] += 1
                new_blocks.append(idx)
                
        return new_blocks

    full_mask = (1 << num_t_subsets) - 1
    initial_blocks = greedy_repair(full_mask)
    current_blocks = reduce_blocks(initial_blocks)
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    while time.time() < deadline:
        if not current_blocks:
            break
        
        # Choose between removal-based and swap-based local search
        if random.random() < 0.5:
            # Removal-based: remove one block, repair
            perturbed_blocks = list(current_blocks)
            remove_idx = random.randint(0, len(perturbed_blocks) - 1)
            perturbed_blocks.pop(remove_idx)
            
            covered_mask = 0
            for idx in perturbed_blocks:
                covered_mask |= block_masks[idx]
            
            uncovered_mask = full_mask & ~covered_mask
            
            if uncovered_mask == 0:
                # Already valid without the removed block
                new_blocks = reduce_blocks(perturbed_blocks)
            else:
                repaired_blocks = greedy_repair(uncovered_mask, perturbed_blocks)
                new_blocks = reduce_blocks(repaired_blocks)
        else:
            # Swap-based: remove one block, try to add one block that covers most uncovered
            perturbed_blocks = list(current_blocks)
            remove_idx = random.randint(0, len(perturbed_blocks) - 1)
            removed_block = perturbed_blocks.pop(remove_idx)
            
            covered_mask = 0
            for idx in perturbed_blocks:
                covered_mask |= block_masks[idx]
            
            uncovered_mask = full_mask & ~covered_mask
            
            if uncovered_mask == 0:
                new_blocks = reduce_blocks(perturbed_blocks)
            else:
                # Find the best single block to add
                best_gain = 0
                best_block = -1
                for i in range(num_blocks):
                    if i == removed_block:
                        continue
                    gain = (uncovered_mask & block_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = i
                
                if best_block >= 0 and best_gain > 0:
                    perturbed_blocks.append(best_block)
                    # Check if we need more repairs
                    covered_mask2 = 0
                    for idx in perturbed_blocks:
                        covered_mask2 |= block_masks[idx]
                    uncovered_mask2 = full_mask & ~covered_mask2
                    if uncovered_mask2 != 0:
                        repaired_blocks = greedy_repair(uncovered_mask2, perturbed_blocks)
                        new_blocks = reduce_blocks(repaired_blocks)
                    else:
                        new_blocks = reduce_blocks(perturbed_blocks)
                else:
                    new_blocks = reduce_blocks(perturbed_blocks)
        
        new_size = len(new_blocks)
        
        if new_size < best_size:
            best_blocks = new_blocks
            best_size = new_size
            current_blocks = new_blocks
        else:
            if new_size == best_size:
                if random.random() < 0.5:
                    current_blocks = new_blocks
            else:
                if random.random() < 0.2:
                    current_blocks = new_blocks
                else:
                    current_blocks = best_blocks

    final_blocks = [list(all_blocks[idx]) for idx in best_blocks]
    return final_blocks
