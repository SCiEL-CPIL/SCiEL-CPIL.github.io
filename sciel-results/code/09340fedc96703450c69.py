import itertools
import time
import numpy as np
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # 1. Precomputation
    # All t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    # All k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_to_tsets: list of lists of tset indices
    # tset_to_blocks: list of lists of block indices
    block_to_tsets = [[] for _ in range(num_blocks)]
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)
    
    def get_solution(indices):
        return [list(all_blocks[i]) for i in indices]

    def run_local_search(initial_indices, max_time_per_run=10.0):
        """
        Takes a list of block indices, applies local search improvements.
        Includes 1-for-1 and 1-for-2 swaps.
        Returns the best list of indices found.
        """
        current = list(initial_indices)
        if not current:
            return []
        
        # Calculate initial cover counts
        cover_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in current:
            for ts_idx in block_to_tsets[b_idx]:
                cover_counts[ts_idx] += 1
        
        best_size = len(current)
        best_solution = list(current)
        
        improved = True
        local_start = time.time()
        no_improvement_rounds = 0
        max_no_improvement = 3
        
        while improved and (time.time() - local_start) < max_time_per_run:
            improved = False
            
            # Phase 1: Remove redundant blocks (multiple passes until stable)
            while True:
                found_redundant = False
                for i in range(len(current) - 1, -1, -1):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    b_idx = current[i]
                    is_redundant = True
                    for ts_idx in block_to_tsets[b_idx]:
                        if cover_counts[ts_idx] == 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        for ts_idx in block_to_tsets[b_idx]:
                            cover_counts[ts_idx] -= 1
                        current.pop(i)
                        found_redundant = True
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                
                if not found_redundant or (time.time() - local_start) > max_time_per_run:
                    break
            
            if not current:
                break
                
            # Phase 2: 1-for-1 Swaps
            # (Existing logic, slightly optimized)
            random.shuffle(current)
            for i in range(len(current)):
                if (time.time() - local_start) > max_time_per_run:
                    break
                    
                b_out = current[i]
                unique_tsets = [ts_idx for ts_idx in block_to_tsets[b_out] if cover_counts[ts_idx] == 1]
                if not unique_tsets:
                    continue

                candidates = set(tset_to_blocks[unique_tsets[0]])
                candidates.discard(b_out)
                for ts_idx in unique_tsets[1:]:
                    if not candidates:
                        break
                    candidates.intersection_update(tset_to_blocks[ts_idx])
                
                if not candidates:
                    continue
                
                best_candidate = -1
                best_score = -1
                b_out_tsets_set = set(block_to_tsets[b_out])
                candidate_list = list(candidates)
                if len(candidate_list) > 200:
                    random.shuffle(candidate_list)
                    candidate_list = candidate_list[:200]
                
                for b_in in candidate_list:
                    # Score: number of newly covered tsets (or reduction in redundancy)
                    # Here we just check if it's a valid swap (covers all unique)
                    # We prefer blocks that cover many other tsets with count > 1? 
                    # Actually, we want to minimize total blocks. 
                    # Since it's 1-for-1, size is constant. 
                    # We want to improve "robustness" or prepare for future removals.
                    # A common heuristic: maximize the minimum count of tsets covered by b_in that are not in b_out?
                    # Or simply: if it covers more tsets with count==1 (outside b_out) it might be better?
                    # Let's stick to a simple score: sum of weights of tsets covered by b_in but not b_out.
                    score = 0
                    for ts_idx in block_to_tsets[b_in]:
                        if ts_idx not in b_out_tsets_set:
                            if cover_counts[ts_idx] == 1:
                                score += 10 # Protecting sparse coverage is good? No, we want to reduce blocks.
                            else:
                                score += 1
                    if score > best_score:
                        best_score = score
                        best_candidate = b_in
                        
                if best_candidate != -1:
                    for ts_idx in block_to_tsets[b_out]:
                        cover_counts[ts_idx] -= 1
                    for ts_idx in block_to_tsets[best_candidate]:
                        cover_counts[ts_idx] += 1
                    current[i] = best_candidate
                    improved = True

            # Phase 3: 1-for-2 Swaps (New)
            # Try to remove 1 block and add 2 blocks.
            # This can reduce the "bottleneck" of a single block covering many unique tsets.
            if (time.time() - local_start) < max_time_per_run:
                random.shuffle(current)
                for i in range(len(current)):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    b_out = current[i]
                    unique_tsets = [ts_idx for ts_idx in block_to_tsets[b_out] if cover_counts[ts_idx] == 1]
                    if not unique_tsets:
                        continue
                    
                    # If unique_tsets is too large, 1-for-2 is unlikely to help efficiently
                    if len(unique_tsets) > 10:
                        continue

                    # Find candidates for B_in1 that cover the first unique tset
                    u0 = unique_tsets[0]
                    cands1 = [b for b in tset_to_blocks[u0] if b != b_out]
                    if not cands1:
                        continue
                    
                    # Limit candidates to keep it fast
                    if len(cands1) > 50:
                        random.shuffle(cands1)
                        cands1 = cands1[:50]
                    
                    found_swap = False
                    for b_in1 in cands1:
                        # Determine remaining unique tsets not covered by b_in1
                        remaining = [ts for ts in unique_tsets if ts not in block_to_tsets[b_in1]]
                        if not remaining:
                            # b_in1 covers all unique tsets. This is a 1-for-1 swap effectively, 
                            # but we are in 1-for-2 phase. We can just do 1-for-1 or skip.
                            # If we add a second block, we increase size by 1. We only want this if it leads to further reductions?
                            # For now, let's strictly look for cases where b_in1 does NOT cover all, 
                            # so we need b_in2.
                            continue
                        
                        # Find candidates for B_in2 that cover ALL remaining tsets
                        # Intersection of tset_to_blocks for all in 'remaining'
                        cands2 = set(tset_to_blocks[remaining[0]])
                        for ts in remaining[1:]:
                            cands2.intersection_update(tset_to_blocks[ts])
                        
                        # Exclude b_out and b_in1
                        cands2.discard(b_out)
                        cands2.discard(b_in1)
                        
                        if not cands2:
                            continue
                        
                        # Pick a random candidate for b_in2 (or best score)
                        # To keep it simple, pick the first one
                        b_in2 = next(iter(cands2))
                        
                        # Check if this swap is beneficial?
                        # 1-for-2 increases block count by 1. 
                        # We should only accept if it allows future reductions?
                        # This is hard to predict.
                        # Alternative: Only accept 1-for-2 if it reduces the "max unique tsets" for other blocks?
                        # Or simply: Don't accept 1-for-2 immediately unless it leads to a valid state 
                        # that is "better" in some metric.
                        # Since we minimize blocks, 1-for-2 is a "costly" move.
                        # We should only use it if it enables a 2-for-1 or 2-for-2 later.
                        # This is complex. 
                        
                        # Let's try a different angle: **2-for-1 Swap**
                        # Remove 2 blocks, add 1 block.
                        # This directly reduces size.
                        # Find two blocks B1, B2 such that their unique tsets can be covered by a single new block B_new.
                        # Unique(B1) U Unique(B2) subset T(B_new).
                        
                        # Let's implement 2-for-1 instead. It's more valuable for minimization.
                        pass # Skip 1-for-2 for now, implement 2-for-1 below

            # Phase 4: 2-for-1 Swaps
            if (time.time() - local_start) < max_time_per_run:
                random.shuffle(current)
                # Iterate over pairs? Too slow O(N^2).
                # Heuristic: Pick a block B1. Find blocks B2 that share many tsets with B1?
                # Or just random pairs.
                # Let's try random pairs with a limit.
                pairs_tried = 0
                max_pairs = 100
                while pairs_tried < max_pairs and (time.time() - local_start) < max_time_per_run:
                    pairs_tried += 1
                    if len(current) < 2:
                        break
                    i, j = random.sample(range(len(current)), 2)
                    b1 = current[i]
                    b2 = current[j]
                    
                    # Get unique tsets for b1 and b2
                    # Note: A tset is unique to b1 if count==1 and b1 covers it.
                    # But if b1 and b2 both cover it, and count==2, it's not unique to either.
                    # When we remove b1 and b2, a tset is "lost" if its count becomes 0.
                    # So we need to find all tsets that are covered ONLY by b1 or ONLY by b2 (or both, but no one else).
                    # Let U1 = {ts | ts in B1 and count[ts]==1}
                    # Let U2 = {ts | ts in B2 and count[ts]==1}
                    # If a ts is in B1 and B2, and count[ts]==2, it is in neither U1 nor U2?
                    # No, if count[ts]==2 and B1,B2 cover it, removing both makes count 0.
                    # So we need to be careful.
                    
                    # Let's define "critical" tsets for the pair {b1, b2}.
                    # A tset is critical if count[ts] <= 2 and (b1 in tset_to_blocks[ts] or b2 in tset_to_blocks[ts]).
                    # Actually, simpler:
                    # Remove b1, b2 temporarily.
                    # Check which tsets become 0.
                    # Those must be covered by B_new.
                    
                    # To avoid expensive temporary removal, we can calculate:
                    # Tsets that will be uncovered:
                    # 1. Tsets with count 1 covered by b1 (and not b2)
                    # 2. Tsets with count 1 covered by b2 (and not b1)
                    # 3. Tsets with count 2 covered by both b1 and b2
                    
                    # Let's just do the temporary update for the small number of tsets in b1 and b2.
                    # Since k=8, t=5, each block has 56 tsets.
                    # Intersection is small.
                    
                    tsets_b1 = set(block_to_tsets[b1])
                    tsets_b2 = set(block_to_tsets[b2])
                    
                    # Identify tsets that will drop to 0
                    lost_tsets = []
                    for ts in tsets_b1 | tsets_b2:
                        c = cover_counts[ts]
                        in_b1 = ts in tsets_b1
                        in_b2 = ts in tsets_b2
                        # New count = c - (1 if in_b1 else 0) - (1 if in_b2 else 0)
                        new_c = c
                        if in_b1: new_c -= 1
                        if in_b2: new_c -= 1
                        if new_c < 0:
                            # Should not happen if counts are correct
                            pass
                        elif new_c == 0:
                            lost_tsets.append(ts)
                    
                    if not lost_tsets:
                        # If no tsets are lost, we can just remove one of them?
                        # No, if no tsets are lost, it means all tsets in b1 and b2 are covered by others.
                        # Then we can remove b1 (redundant).
                        # This should have been caught in Phase 1.
                        # But maybe Phase 1 missed it due to ordering?
                        # If so, remove b1.
                        for ts in tsets_b1:
                            cover_counts[ts] -= 1
                        current.pop(i)
                        improved = True
                        break # Restart loop to re-index
                    
                    if len(lost_tsets) > 10:
                        continue # Too many to cover with 1 block likely

                    # Find candidates for B_new that cover ALL lost_tsets
                    cands_new = set(tset_to_blocks[lost_tsets[0]])
                    for ts in lost_tsets[1:]:
                        cands_new.intersection_update(tset_to_blocks[ts])
                    
                    cands_new.discard(b1)
                    cands_new.discard(b2)
                    
                    if not cands_new:
                        continue
                    
                    # Pick a candidate
                    # Prefer one that covers many other tsets?
                    b_new = next(iter(cands_new))
                    
                    # Perform swap
                    for ts in tsets_b1:
                        cover_counts[ts] -= 1
                    for ts in tsets_b2:
                        cover_counts[ts] -= 1
                    for ts in block_to_tsets[b_new]:
                        cover_counts[ts] += 1
                    
                    # Remove b1, b2, add b_new
                    # Be careful with indices
                    if i < j:
                        current.pop(j)
                        current.pop(i)
                    else:
                        current.pop(i)
                        current.pop(j)
                    current.append(b_new)
                    
                    improved = True
                    if len(current) < best_size:
                        best_size = len(current)
                        best_solution = list(current)
                    
                    break # Restart loop to re-index

            if not improved:
                no_improvement_rounds += 1
                if no_improvement_rounds >= max_no_improvement:
                    break
            else:
                no_improvement_rounds = 0
        
        return best_solution

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []
        
        tset_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in range(num_blocks):
            for ts_idx in block_to_tsets[b_idx]:
                tset_counts[ts_idx] += 1
        safe_counts = np.maximum(tset_counts, 1).astype(np.float64)
        tset_weights = 1.0 / safe_counts
        
        gains = np.zeros(num_blocks, dtype=np.float64)
        for b_idx in range(num_blocks):
            if block_to_tsets[b_idx]:
                gains[b_idx] = np.sum(tset_weights[block_to_tsets[b_idx]])
                
        selected_mask = np.zeros(num_blocks, dtype=np.bool_)
        
        while uncovered_count > 0 and time.time() < deadline - 15:
            masked_gains = gains.copy()
            masked_gains[selected_mask] = -1.0
            best_idx = int(np.argmax(masked_gains))
            best_gain = gains[best_idx]
            
            if best_gain <= 0:
                break
                
            selected.append(best_idx)
            selected_mask[best_idx] = True
            
            for ts_idx in block_to_tsets[best_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_count -= 1
                    w = tset_weights[ts_idx]
                    for b in tset_to_blocks[ts_idx]:
                        if not selected_mask[b]:
                            gains[b] -= w
                            
        return selected

    best_global_solution = None
    best_global_size = float('inf')
    
    # Run multiple instances
    seeds = [42, 123, 777, 2024, 555]
    time_budgets = [40, 30, 20, 15, 10]
    
    for seed, budget in zip(seeds, time_budgets):
        if time.time() > deadline - budget - 5:
            continue
        init = greedy_cover(seed)
        if init:
            sol = run_local_search(init, max_time_per_run=budget)
            if sol and len(sol) < best_global_size:
                best_global_size = len(sol)
                best_global_solution = sol
    
    # Final polish
    if best_global_solution and time.time() < deadline - 3:
        remaining_time = deadline - time.time() - 1
        if remaining_time > 3:
            improved_sol = run_local_search(best_global_solution, max_time_per_run=remaining_time)
            if improved_sol and len(improved_sol) < best_global_size:
                best_global_size = len(improved_sol)
                best_global_solution = improved_sol
            
    if not best_global_solution:
        if init:
            return get_solution(init)
        return []
        
    return get_solution(best_global_solution)

if __name__ == "__main__":
    v, k, t = 19, 8, 5
    res = construct(v, k, t)
    print(f"Number of blocks: {len(res)}")
    covered = set()
    for block in res:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    total_tsets = len(list(itertools.combinations(range(v), t)))
    print(f"Covered: {len(covered)}/{total_tsets}")
