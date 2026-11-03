import itertools
import time
import random
import heapq
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubs = list(itertools.combinations(points, t))
    n_tsubs = len(all_tsubs)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # block_mask[b] is a bitmask of tsubs covered by block b
    block_mask = [0] * n_blocks
    # tsub_blocks[s] is a list of blocks covering tsub s
    tsub_blocks = [[] for _ in range(n_tsubs)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks[idx].append(b_idx)
        block_mask[b_idx] = mask
        
    full_mask = (1 << n_tsubs) - 1
    
    # 2. Greedy Construction
    def greedy_cover():
        uncovered_mask = full_mask
        selected = []
        selected_set = set()
        
        # Initial gains
        gains = [0] * n_blocks
        for b in range(n_blocks):
            gains[b] = block_mask[b].bit_count()
            
        # Heap: (-gain, block_idx)
        heap = [(-gains[b], b) for b in range(n_blocks)]
        heapq.heapify(heap)
        
        # current_gain[b] is the number of *currently uncovered* tsubs covered by b
        current_gain = gains[:]
        
        while uncovered_mask:
            if time.time() > deadline - 10.0:
                break
                
            best_block = -1
            best_gain_val = 0
            
            # Pop stale entries
            while heap:
                neg_g, b_idx = heapq.heappop(heap)
                g = -neg_g
                if g == current_gain[b_idx] and b_idx not in selected_set:
                    best_block = b_idx
                    best_gain_val = g
                    break
                # else stale or already selected, discard
                
            if best_block == -1:
                break
                
            selected.append(best_block)
            selected_set.add(best_block)
            current_gain[best_block] = -1 # Mark as selected/invalid
            
            # Update uncovered mask
            new_covered = block_mask[best_block] & uncovered_mask
            uncovered_mask &= ~block_mask[best_block]
            
            # Update gains for blocks covering newly covered tsubs
            # Iterate through bits in new_covered
            m = new_covered
            while m:
                lsb = m & -m
                ts_idx = lsb.bit_length() - 1
                m ^= lsb
                
                # For each block covering this ts_idx, decrease its gain
                for b_idx in tsub_blocks[ts_idx]:
                    if b_idx not in selected_set:
                        current_gain[b_idx] -= 1
                        heapq.heappush(heap, (-current_gain[b_idx], b_idx))
                        
        return selected

    initial_solution = greedy_cover()
    best_solution = initial_solution
    best_size = len(best_solution)
    
    # 3. Local Search
    def local_search(solution, seed):
        rng = random.Random(seed)
        selected_set = set(solution)
        
        # cover_count[s] = number of selected blocks covering s
        cover_count = [0] * n_tsubs
        for b in selected_set:
            mask = block_mask[b]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Phase 1: Redundancy Removal
            to_remove = []
            for b in selected_set:
                mask = block_mask[b]
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    to_remove.append(b)
                    
            if to_remove:
                for b in to_remove:
                    selected_set.remove(b)
                    mask = block_mask[b]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                improved = True
                continue # Restart loop as removals might make others redundant
                
            # Phase 2: Swaps
            # Iterate over a copy of selected_set to allow modification
            current_blocks = list(selected_set)
            rng.shuffle(current_blocks)
            
            for b in current_blocks:
                if time.time() > deadline - 5.0:
                    break
                if b not in selected_set:
                    continue
                    
                # Find unique tsubs for b
                unique_tsubs = []
                mask = block_mask[b]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        unique_tsubs.append(idx)
                    m ^= lsb
                    
                if not unique_tsubs:
                    # Should have been caught by redundancy, but just in case
                    selected_set.remove(b)
                    mask = block_mask[b]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    improved = True
                    break
                    
                # Find candidates for swap
                # Start with the most constrained tsub (fewest covering blocks)
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates.intersection_update(tsub_blocks[ts_idx])
                    
                if not candidates:
                    continue
                    
                candidates.discard(b)
                if not candidates:
                    continue
                    
                # Filter out already selected blocks
                candidates = candidates - selected_set
                
                if not candidates:
                    continue
                    
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                # Limit candidates to avoid too much work
                if len(cand_list) > 10:
                    cand_list = cand_list[:10]
                    
                swapped = False
                for r in cand_list:
                    # Perform swap: remove b, add r
                    selected_set.remove(b)
                    mask_b = block_mask[b]
                    m = mask_b
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                        
                    selected_set.add(r)
                    mask_r = block_mask[r]
                    m = mask_r
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] += 1
                        m ^= lsb
                        
                    improved = True
                    swapped = True
                    break
                    
                if swapped:
                    break # Restart local search loop
                    
        return list(selected_set)

    # 4. Iterated Local Search
    current_solution = initial_solution
    seed = 42
    
    while time.time() < deadline - 2.0:
        # Local search
        refined = local_search(current_solution, seed)
        
        if len(refined) < best_size:
            best_solution = refined
            best_size = len(refined)
            current_solution = refined
            
        # Perturbation: Randomly remove a few blocks and re-add them via greedy or random
        # Simple perturbation: remove 5% of blocks randomly, then run a few greedy steps
        if len(current_solution) > 0:
            perturbed = current_solution.copy()
            num_to_remove = max(1, len(perturbed) // 20)
            to_remove = random.sample(perturbed, num_to_remove)
            for b in to_remove:
                perturbed.remove(b)
                
            # Re-add blocks using a simple greedy step on the perturbed solution
            # Calculate current coverage
            covered_mask = 0
            for b in perturbed:
                covered_mask |= block_mask[b]
                
            uncovered = full_mask & ~covered_mask
            
            if uncovered:
                # Greedy fill
                gains = [0] * n_blocks
                for b in range(n_blocks):
                    if b not in set(perturbed):
                        gains[b] = (block_mask[b] & uncovered).bit_count()
                        
                # Sort by gain
                sorted_blocks = sorted(range(n_blocks), key=lambda x: gains[x], reverse=True)
                
                for b in sorted_blocks:
                    if time.time() > deadline - 2.0:
                        break
                    if (block_mask[b] & uncovered) == 0:
                        continue
                    if b in set(perturbed):
                        continue
                    perturbed.append(b)
                    uncovered &= ~block_mask[b]
                    if uncovered == 0:
                        break
            current_solution = perturbed
        
        seed += 1

    # Final local search on best solution
    if time.time() < deadline - 1.0:
        final_refined = local_search(best_solution, seed)
        if len(final_refined) < len(best_solution):
            best_solution = final_refined

    # Convert indices back to lists
    result = [list(all_blocks[b_idx]) for b_idx in best_solution]
    
    # Ensure distinct blocks
    seen = set()
    final_result = []
    for b in result:
        b_tuple = tuple(b)
        if b_tuple not in seen:
            seen.add(b_tuple)
            final_result.append(b)
            
    return final_result
