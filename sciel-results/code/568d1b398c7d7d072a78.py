import itertools
import time
import numpy as np
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_to_tsets = [[] for _ in range(num_blocks)]
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)

    def get_solution(indices):
        return [list(all_blocks[i]) for i in indices]

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []
        
        # Weights: prioritize rare t-sets
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
        
        while uncovered_count > 0 and time.time() < deadline - 20:
            best_idx = -1
            best_gain = -1.0
            
            # Optimization: Use numpy argmax on gains where mask is False
            # But we need to update gains efficiently.
            # Simple scan is O(N). N=75k. 500 iterations -> 37M ops. ~0.5s. OK.
            for i in range(num_blocks):
                if not selected_mask[i] and gains[i] > best_gain:
                    best_gain = gains[i]
                    best_idx = i
                    
            if best_idx == -1 or best_gain <= 0:
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

    def run_local_search(initial_indices, max_time_per_run=10.0):
        current = list(initial_indices)
        if not current:
            return []
        
        cover_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in current:
            for ts_idx in block_to_tsets[b_idx]:
                cover_counts[ts_idx] += 1
        
        best_size = len(current)
        best_solution = list(current)
        
        local_start = time.time()
        no_improve_time = 0.0
        last_improve_time = local_start
        
        while (time.time() - local_start) < max_time_per_run:
            improved = False
            
            # Phase 1: Remove redundant blocks
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
                    improved = True
                    if len(current) < best_size:
                        best_size = len(current)
                        best_solution = list(current)
                        last_improve_time = time.time()
            
            # Phase 2: 1-for-2 moves
            # Try to remove a block and add two others
            # To be efficient, only try blocks with few unique t-sets
            random.shuffle(current)
            for i in range(len(current)):
                if (time.time() - local_start) > max_time_per_run:
                    break
                
                b_out = current[i]
                unique_tsets = [ts for ts in block_to_tsets[b_out] if cover_counts[ts] == 1]
                
                if len(unique_tsets) > 10: # Heuristic: if too many unique, hard to cover with 2
                    continue
                
                if not unique_tsets:
                    continue # Should have been caught by Phase 1

                # Find candidates for the first block in the pair
                # It must cover at least one unique tset
                cand1 = set(tset_to_blocks[unique_tsets[0]])
                cand1.discard(b_out)
                
                if len(cand1) > 50:
                    cand1 = set(random.sample(list(cand1), 50))
                
                found_pair = False
                for b_a in cand1:
                    # What unique tsets are NOT covered by b_a?
                    remaining_unique = [ts for ts in unique_tsets if ts not in set(block_to_tsets[b_a])]
                    if not remaining_unique:
                        # b_a covers all unique tsets. This is a 1-for-1 swap, not 1-for-2.
                        # We can handle swaps separately or accept this as a valid move (size unchanged)
                        # But we want to reduce size. So we need b_a and b_b.
                        # If b_a covers all, we can just swap. But that doesn't reduce size.
                        # So we need b_a and b_b to cover unique tsets, and we remove b_out.
                        # Size change: -1 + 2 = +1. Wait.
                        # 1-for-2: Remove 1, Add 2. Size increases by 1.
                        # Why would we do that? To enable future 2-for-1 or redundancy removal.
                        # So we accept if it leads to a "better" structure?
                        # Actually, usually we do 1-for-1 (swap) to rearrange, then 1-for-2 (expand) then 2-for-1 (contract).
                        # Or directly 1-for-2 if it allows immediate redundancy removal?
                        # Let's stick to moves that reduce size or maintain size.
                        # 1-for-2 increases size. It's a "detour".
                        # Let's prioritize 2-for-1 and 1-for-1.
                        continue
                    
                    # Find b_b that covers remaining_unique
                    cand2 = set(tset_to_blocks[remaining_unique[0]])
                    cand2.discard(b_out)
                    cand2.discard(b_a)
                    for ts in remaining_unique[1:]:
                        if not cand2:
                            break
                        cand2.intersection_update(tset_to_blocks[ts])
                    
                    if not cand2:
                        continue
                    
                    # Pick best b_b
                    # Heuristic: maximize coverage of non-unique tsets with count 2
                    best_b_b = -1
                    best_score = -1
                    for b_b in list(cand2)[:20]:
                        score = 0
                        for ts in block_to_tsets[b_b]:
                            if cover_counts[ts] == 2:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_b_b = b_b
                    
                    if best_b_b != -1:
                        # Apply move: Remove b_out, Add b_a, b_b
                        # Check if this creates new redundancies?
                        # We just apply it.
                        for ts_idx in block_to_tsets[b_out]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[b_a]:
                            cover_counts[ts_idx] += 1
                        for ts_idx in block_to_tsets[best_b_b]:
                            cover_counts[ts_idx] += 1
                        
                        current[i] = b_a
                        current.append(best_b_b)
                        improved = True
                        found_pair = True
                        break
                
                if found_pair:
                    break

            # Phase 3: 2-for-1 moves
            # Try to remove two blocks and add one
            # This reduces size by 1.
            # We need to find a pair (A, B) such that all t-sets that would be uncovered
            # are covered by a single block C.
            
            # Heuristic: Look for pairs of blocks that share many t-sets?
            # Or just random pairs.
            # To speed up, we can iterate over t-sets that are "critical".
            
            # Let's try random pairs first.
            if len(current) > 2:
                random.shuffle(current)
                for i in range(min(len(current) - 1, 20)):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    
                    b_out1 = current[i]
                    b_out2 = current[i + 1]
                    
                    # Calculate t-sets that would become uncovered
                    # A tset is uncovered if its count becomes 0.
                    # Count becomes 0 if it was 1 and in A or B, or was 2 and in A and B.
                    
                    # Efficiently:
                    # tsets_in_A = set(block_to_tsets[b_out1])
                    # tsets_in_B = set(block_to_tsets[b_out2])
                    # lost_tsets = set()
                    # for ts in tsets_in_A:
                    #     if cover_counts[ts] == 1:
                    #         lost_tsets.add(ts)
                    #     elif cover_counts[ts] == 2 and ts in tsets_in_B:
                    #         lost_tsets.add(ts)
                    # for ts in tsets_in_B:
                    #     if cover_counts[ts] == 1:
                    #         lost_tsets.add(ts)
                    
                    # Optimization:
                    # If A and B are disjoint, lost_tsets are just those with count 1 in A or B.
                    # If they overlap, we need to be careful.
                    
                    tsets_A = set(block_to_tsets[b_out1])
                    tsets_B = set(block_to_tsets[b_out2])
                    lost_tsets = set()
                    
                    for ts in tsets_A:
                        if cover_counts[ts] == 1:
                            lost_tsets.add(ts)
                        elif cover_counts[ts] == 2 and ts in tsets_B:
                            lost_tsets.add(ts)
                    
                    for ts in tsets_B:
                        if cover_counts[ts] == 1:
                            lost_tsets.add(ts)
                        # The case count==2 and in A is already handled if ts in A
                        # Wait, if ts in B, count==2, and ts in A, it's handled in first loop.
                        # If ts in B, count==2, and ts NOT in A, then removing B makes count 1, not 0.
                        # So only count==1 in B is lost.
                    
                    if not lost_tsets:
                        # Both are redundant? Should have been caught.
                        # Remove one?
                        continue

                    # Find a block C that covers all lost_tsets
                    if len(lost_tsets) > 10:
                        continue # Unlikely to find a single block
                    
                    candidates = set(tset_to_blocks[list(lost_tsets)[0]])
                    candidates.discard(b_out1)
                    candidates.discard(b_out2)
                    for ts in list(lost_tsets)[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(tset_to_blocks[ts])
                    
                    if not candidates:
                        continue
                    
                    # Pick best C
                    best_c = -1
                    best_add = -1
                    for c in list(candidates)[:20]:
                        add = 0
                        for ts in block_to_tsets[c]:
                            if cover_counts[ts] == 1:
                                add += 1
                        if add > best_add:
                            best_add = add
                            best_c = c
                    
                    if best_c != -1:
                        # Apply: Remove b_out1, b_out2, Add best_c
                        for ts_idx in block_to_tsets[b_out1]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[b_out2]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[best_c]:
                            cover_counts[ts_idx] += 1
                        
                        current[i] = best_c
                        current.pop(i + 1)
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                            last_improve_time = time.time()
                        break

            if not improved:
                # Try 1-for-1 swaps to escape local minima
                random.shuffle(current)
                for i in range(min(len(current), 50)):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    b_out = current[i]
                    unique_tsets = [ts for ts in block_to_tsets[b_out] if cover_counts[ts] == 1]
                    if not unique_tsets:
                        continue # Redundant, should be removed
                    
                    candidates = set(tset_to_blocks[unique_tsets[0]])
                    candidates.discard(b_out)
                    for ts in unique_tsets[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(tset_to_blocks[ts])
                    
                    if not candidates:
                        continue
                    
                    # Pick a candidate that maximizes "potential" for future reduction
                    # e.g., covers many t-sets with count 2
                    best_cand = -1
                    best_score = -1
                    for cand in list(candidates)[:20]:
                        score = 0
                        for ts in block_to_tsets[cand]:
                            if cover_counts[ts] == 2:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                    
                    if best_cand != -1:
                        for ts_idx in block_to_tsets[b_out]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[best_cand]:
                            cover_counts[ts_idx] += 1
                        current[i] = best_cand
                        improved = True
                        break

            if not improved:
                # No improvement. Check time for restart?
                # For now, just continue.
                pass

        return best_solution

    best_global_solution = None
    best_global_size = float('inf')
    
    # Run multiple instances
    seeds = [42, 123, 777, 1234, 5678]
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        t_start = time.time()
        init = greedy_cover(seed)
        if init:
            ls_time = min(20.0, deadline - t_start - 5)
            if ls_time > 5:
                sol = run_local_search(init, max_time_per_run=ls_time)
                if sol and len(sol) < best_global_size:
                    best_global_size = len(sol)
                    best_global_solution = sol
    
    # Final polish
    if best_global_solution and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 5:
            improved_sol = run_local_search(best_global_solution, max_time_per_run=remaining_time)
            if improved_sol and len(improved_sol) < best_global_size:
                best_global_size = len(improved_sol)
                best_global_solution = improved_sol

    if not best_global_solution:
        return []
        
    return get_solution(best_global_solution)

if __name__ == "__main__":
    v, k, t = 19, 8, 5
    res = construct(v, k, t)
    print(f"Number of blocks: {len(res)}")
