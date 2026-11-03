import itertools
import time
import random
import sys
import heapq
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_masks[i] is a bitset of t-subsets covered by block i
    block_masks = [0] * num_blocks
    # block_subset_indices[i] is a list of t-subset indices covered by block i
    block_subset_indices = [[] for _ in range(num_blocks)]
    
    # block_cover_masks[j] is a bitset of blocks covering t-subset j
    # Use Python ints for bitsets (arbitrary precision)
    block_cover_masks = [0] * num_t_subsets
    
    for i, block in enumerate(all_blocks):
        mask = 0
        subs_idx = []
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
            subs_idx.append(idx)
            block_cover_masks[idx] |= (1 << i)
        block_masks[i] = mask
        block_subset_indices[i] = subs_idx

    full_mask = (1 << num_t_subsets) - 1
    
    # Helper: Reduce redundant blocks
    def reduce_blocks(block_indices):
        if not block_indices:
            return []
        
        # Count coverage for each t-subset
        cover_count = [0] * num_t_subsets
        
        for idx in block_indices:
            for s_idx in block_subset_indices[idx]:
                cover_count[s_idx] += 1
                
        new_blocks = []
        for idx in block_indices:
            subs = block_subset_indices[idx]
            # Check if any subset is covered only by this block
            is_redundant = True
            for s_idx in subs:
                if cover_count[s_idx] == 1:
                    is_redundant = False
                    break
                    
            if is_redundant:
                # It is redundant, do not add to new_blocks
                pass
            else:
                new_blocks.append(idx)
                
        return new_blocks

    # Helper: Greedy Set Cover with heap-based lazy update
    def greedy_cover():
        uncovered = full_mask
        selected = []
        selected_set = set()
        
        # Initialize heap with all blocks
        heap = []
        for i in range(num_blocks):
            gain = block_masks[i].bit_count()
            heapq.heappush(heap, (-gain, i))
        
        while uncovered:
            best_gain = 0
            best_idx = -1
            
            # Pop from heap until we find a valid block
            while heap:
                neg_gain, i = heapq.heappop(heap)
                gain = -neg_gain
                
                if i in selected_set:
                    continue
                
                # Recalculate actual gain
                actual_gain = (uncovered & block_masks[i]).bit_count()
                
                if actual_gain != gain:
                    # Stale entry, push updated entry
                    heapq.heappush(heap, (-actual_gain, i))
                    continue
                
                if actual_gain == 0:
                    continue
                    
                best_gain = actual_gain
                best_idx = i
                break
            
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
            
            # Update gains for remaining blocks in heap
            # We'll let lazy evaluation handle this - no need to update all
            # But we need to push updated gains for blocks that might now have higher relative gain
            # Actually, since we're using lazy deletion, we just continue popping
            
        return selected

    # Helper: Find candidates for a set of t-subsets
    # Returns a list of block indices that cover all subsets in E
    def find_candidates(E):
        if not E:
            return list(range(num_blocks))
        
        # Start with the first element's mask
        current_mask = block_cover_masks[E[0]]
        
        for e in E[1:]:
            current_mask &= block_cover_masks[e]
            if current_mask == 0:
                return []
                
        # Extract indices from current_mask
        indices = []
        m = current_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
            
        return indices

    # Initial Solution
    current_blocks = greedy_cover()
    current_blocks = reduce_blocks(current_blocks)
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    # Let's maintain cover_count dynamically.
    cover_count = [0] * num_t_subsets
    for idx in current_blocks:
        for s_idx in block_subset_indices[idx]:
            cover_count[s_idx] += 1

    def update_cover_count(idx, delta):
        for s_idx in block_subset_indices[idx]:
            cover_count[s_idx] += delta

    def is_redundant(idx):
        for s_idx in block_subset_indices[idx]:
            if cover_count[s_idx] == 1:
                return False
        return True

    def try_reduce():
        # Remove redundant blocks
        new_blocks = []
        for idx in current_blocks:
            if is_redundant(idx):
                update_cover_count(idx, -1)
            else:
                new_blocks.append(idx)
        return new_blocks

    # Local Search Loop with improved 1-swap using candidate sets
    no_improvement_count = 0
    while time.time() < deadline:
        improved = False
        
        # 1. Delete redundant blocks
        new_blocks = try_reduce()
        if len(new_blocks) < len(current_blocks):
            current_blocks = new_blocks
            improved = True
            if len(current_blocks) < best_size:
                best_blocks = list(current_blocks)
                best_size = len(best_blocks)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            continue # Restart loop to re-evaluate
        
        # 2. 1-Swap: Replace a block with another that covers its unique elements
        # Use a more efficient approach: precompute which blocks can replace which
        indices_to_try = list(range(len(current_blocks)))
        random.shuffle(indices_to_try)
        
        for pos in indices_to_try:
            if time.time() > deadline:
                break
                
            b_idx = current_blocks[pos]
            
            # Find unique elements of b_idx
            unique_e = []
            for s_idx in block_subset_indices[b_idx]:
                if cover_count[s_idx] == 1:
                    unique_e.append(s_idx)
                    
            if not unique_e:
                # Redundant, remove it
                update_cover_count(b_idx, -1)
                current_blocks.pop(pos)
                improved = True
                if len(current_blocks) < best_size:
                    best_blocks = list(current_blocks)
                    best_size = len(best_blocks)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                break # Restart loop as indices changed
            
            # Find candidates that cover unique_e using set intersection
            # Start with the most constrained subset
            unique_e_sorted = sorted(unique_e, key=lambda x: bin(block_cover_masks[x]).count('1'))
            
            candidates_set = set()
            first = True
            for s_idx in unique_e_sorted:
                if first:
                    # Extract indices from mask
                    m = block_cover_masks[s_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        candidates_set.add(idx)
                        m ^= lsb
                    first = False
                else:
                    # Intersect with current candidates
                    m = block_cover_masks[s_idx]
                    new_candidates = set()
                    for c in candidates_set:
                        if (block_cover_masks[s_idx] >> c) & 1:
                            new_candidates.add(c)
                    candidates_set = new_candidates
                    if not candidates_set:
                        break
            
            # Filter out b_idx itself
            candidates_set.discard(b_idx)
            
            if candidates_set:
                # Pick a random candidate
                new_idx = random.choice(list(candidates_set))
                
                # Perform swap
                update_cover_count(b_idx, -1)
                update_cover_count(new_idx, 1)
                current_blocks[pos] = new_idx
                
                improved = True
                no_improvement_count += 1
                
        # 3. 2-Exchange: Remove 2, Add 1
        max_2ex_attempts = 50
        attempts = 0
        while attempts < max_2ex_attempts and time.time() < deadline:
            attempts += 1
            if len(current_blocks) < 2:
                break
                
            i1, i2 = random.sample(range(len(current_blocks)), 2)
            b1 = current_blocks[i1]
            b2 = current_blocks[i2]
            
            # Find unique elements of b1 U b2
            e_set = set()
            for s_idx in block_subset_indices[b1]:
                if cover_count[s_idx] <= 2:
                    e_set.add(s_idx)
            for s_idx in block_subset_indices[b2]:
                if cover_count[s_idx] <= 2:
                    e_set.add(s_idx)
                    
            if not e_set:
                continue
                
            unique_e_12 = list(e_set)
            
            # Find candidates that cover unique_e_12
            candidates = find_candidates(unique_e_12)
            
            # Filter out b1 and b2
            candidates = [c for c in candidates if c != b1 and c != b2]
            
            if candidates:
                new_idx = random.choice(candidates)
                
                # Perform 2-Exchange
                update_cover_count(b1, -1)
                update_cover_count(b2, -1)
                update_cover_count(new_idx, 1)
                
                if i1 > i2:
                    i1, i2 = i2, i1
                current_blocks.pop(i2)
                current_blocks.pop(i1)
                current_blocks.append(new_idx)
                
                improved = True
                if len(current_blocks) < best_size:
                    best_blocks = list(current_blocks)
                    best_size = len(best_blocks)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                break # Restart loop after a successful 2-exchange
                
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > 100:
                # Perturbation: pick a random block, remove it, and repair greedily
                if len(current_blocks) > 0:
                    # Pick a random block to remove
                    rem_pos = random.randint(0, len(current_blocks) - 1)
                    rem_idx = current_blocks[rem_pos]
                    
                    # Remove it
                    update_cover_count(rem_idx, -1)
                    current_blocks.pop(rem_pos)
                    
                    # Repair: find uncovered subsets
                    uncovered = []
                    for s_idx in range(num_t_subsets):
                        if cover_count[s_idx] == 0:
                            uncovered.append(s_idx)
                    
                    # Greedily add blocks to cover uncovered using heap
                    while uncovered and time.time() < deadline:
                        best_gain = 0
                        best_idx = -1
                        
                        # Use heap for efficiency
                        heap = []
                        for i in range(num_blocks):
                            if i in current_blocks:
                                continue
                            # Calculate gain: how many uncovered does i cover?
                            gain = 0
                            for s_idx in block_subset_indices[i]:
                                if cover_count[s_idx] == 0:
                                    gain += 1
                            if gain > 0:
                                heapq.heappush(heap, (-gain, i))
                        
                        while heap:
                            neg_gain, i = heapq.heappop(heap)
                            gain = -neg_gain
                            # Verify gain is still valid
                            actual_gain = 0
                            for s_idx in block_subset_indices[i]:
                                if cover_count[s_idx] == 0:
                                    actual_gain += 1
                            if actual_gain != gain:
                                heapq.heappush(heap, (-actual_gain, i))
                                continue
                            if actual_gain > 0:
                                best_gain = actual_gain
                                best_idx = i
                                break
                        
                        if best_idx == -1 or best_gain == 0:
                            break
                            
                        current_blocks.append(best_idx)
                        update_cover_count(best_idx, 1)
                        
                        # Update uncovered list (remove covered ones)
                        # Simple way: rebuild
                        uncovered = [s for s in uncovered if cover_count[s] == 0]
                        
                    new_blocks = try_reduce()
                    if len(new_blocks) < len(best_blocks):
                        best_blocks = list(new_blocks)
                        best_size = len(best_blocks)
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    current_blocks = new_blocks
                else:
                    no_improvement_count += 1

    final_blocks = [list(all_blocks[idx]) for idx in best_blocks]
    
    # Final validation
    covered = [0] * num_t_subsets
    for block in final_blocks:
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            covered[idx] += 1
            
    if any(c == 0 for c in covered):
        # Fallback to greedy if invalid
        fallback = greedy_cover()
        fallback = reduce_blocks(fallback)
        final_blocks = [list(all_blocks[idx]) for idx in fallback]
        
    return final_blocks
