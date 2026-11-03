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
    
    # Precompute tset indices for each block
    block_to_tsets = [[] for _ in range(num_blocks)]
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)

    # Convert to numpy arrays for faster vectorized operations where possible
    # But lists of lists are often faster for sparse updates in Python loops
    # Let's stick to lists for flexibility but use numpy for counts/weights

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
            
            # Scan for best block
            # Using numpy argmax on a masked array is slow due to allocation.
            # Manual loop is often faster for this size if optimized.
            # Let's try a numpy approach: set selected gains to -inf, argmax, restore.
            # Or just loop. 75k * ~400 iters = 30M ops. Python loop is slow.
            # Let's use numpy:
            # Create a copy of gains with selected set to -inf? No, memory copy.
            # Let's stick to the loop but make it tight.
            
            # Optimization: Use a list of active blocks?
            # For now, simple loop.
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

    def run_local_search(initial_indices, max_time_per_run=10.0, seed=42):
        rng = random.Random(seed)
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
        
        while (time.time() - local_start) < max_time_per_run:
            improved = False
            
            # Phase 1: Remove redundant blocks
            # Randomize order to avoid bias
            indices = list(range(len(current)))
            rng.shuffle(indices)
            for i in indices:
                if (time.time() - local_start) > max_time_per_run:
                    break
                if i >= len(current):
                    continue
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
                    # Continue removing redundants in same pass? 
                    # Yes, but indices are now shifted. 
                    # Simplest: break and restart loop to rebuild indices.
                    break
            
            if improved:
                continue

            # Phase 2: 1-for-1 Swaps (Escape local minima)
            # Pick a block, find its unique t-sets, find a replacement that covers them
            # and ideally covers more "critical" (count 2) t-sets or reduces future risk.
            
            # Strategy: Randomly pick a block to swap out.
            # To be efficient, pick a block that has few unique t-sets? 
            # Or just random.
            
            swap_attempts = 0
            max_swap_attempts = 20
            while swap_attempts < max_swap_attempts:
                if (time.time() - local_start) > max_time_per_run:
                    break
                swap_attempts += 1
                
                if not current:
                    break
                    
                i = rng.randint(0, len(current) - 1)
                b_out = current[i]
                
                # Find unique t-sets of b_out
                unique_tsets = [ts for ts in block_to_tsets[b_out] if cover_counts[ts] == 1]
                
                if not unique_tsets:
                    # This block is redundant. Should have been caught in Phase 1.
                    # Remove it.
                    for ts_idx in block_to_tsets[b_out]:
                        cover_counts[ts_idx] -= 1
                    current.pop(i)
                    improved = True
                    if len(current) < best_size:
                        best_size = len(current)
                        best_solution = list(current)
                    break
                
                # Find candidates that cover all unique_tsets
                # Start with the rarest unique tset
                min_count_ts = unique_tsets[0]
                min_count = len(tset_to_blocks[min_count_ts])
                for ts in unique_tsets[1:]:
                    c = len(tset_to_blocks[ts])
                    if c < min_count:
                        min_count = c
                        min_count_ts = ts
                        
                candidates = tset_to_blocks[min_count_ts]
                
                best_cand = -1
                best_score = -1.0
                
                for b_cand in candidates:
                    if b_cand == b_out:
                        continue
                    # Check if it covers all unique tsets
                    valid = True
                    # Optimization: Use a set? 
                    # Since unique_tsets is small (usually 1-3), linear check is fine.
                    cand_tsets = set(block_to_tsets[b_cand])
                    for ts in unique_tsets:
                        if ts not in cand_tsets:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score the candidate
                    # We want to minimize the number of "new" unique tsets created?
                    # Or maximize the number of tsets with count 2 that it covers?
                    # If it covers a tset with count 2, that tset becomes count 3.
                    # If b_out covered a tset with count 2, that tset becomes count 1.
                    # So we want b_cand to cover as many of b_out's "count 2" tsets as possible?
                    # Actually, we want to keep the cover robust.
                    
                    # Let's score by: 
                    # 1. How many tsets with count 2 does b_cand cover? (Increases stability)
                    # 2. How many tsets with count 1 does b_cand cover that are NOT in b_out? (Expands coverage, good)
                    # 3. How many tsets with count 1 does b_cand cover that ARE in b_out? (Redundant, neutral)
                    
                    # Simpler heuristic: Maximize overlap with "count 2" tsets of b_out?
                    # If b_cand covers a tset that b_out also covers, and that tset had count 2,
                    # then after swap, it goes 2 -> 1 (from b_out removal) -> 2 (from b_cand addition). Net 0.
                    # If b_cand covers a tset that b_out does NOT cover, and that tset had count 2,
                    # it goes 2 -> 3. Net +1.
                    
                    # Let's just try to maximize the number of tsets with count >= 2 covered by b_cand.
                    score = 0
                    for ts in block_to_tsets[b_cand]:
                        if cover_counts[ts] >= 2:
                            score += 1
                            
                    # Add a small random noise to break ties
                    score += rng.random() * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_cand = b_cand
                        
                if best_cand != -1:
                    # Perform swap
                    for ts_idx in block_to_tsets[b_out]:
                        cover_counts[ts_idx] -= 1
                    for ts_idx in block_to_tsets[best_cand]:
                        cover_counts[ts_idx] += 1
                    current[i] = best_cand
                    improved = True
                    # We don't update best_size here as size is unchanged,
                    # but we might have improved structure for future reductions.
                    break

            if improved:
                continue

            # Phase 3: 2-for-1 moves (Reduction)
            # Try to find two blocks A, B such that their union of unique t-sets
            # can be covered by a single block C.
            # This reduces size by 1.
            
            # Heuristic: Look for pairs of blocks that share many t-sets?
            # Or just random pairs.
            # To speed up, we can iterate over t-sets that are "critical".
            
            # Let's try random pairs first.
            if len(current) > 2:
                pair_attempts = 0
                max_pair_attempts = 10
                while pair_attempts < max_pair_attempts:
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    pair_attempts += 1
                    
                    # Pick two distinct blocks
                    i1 = rng.randint(0, len(current) - 1)
                    i2 = rng.randint(0, len(current) - 1)
                    if i1 == i2:
                        continue
                    
                    b_out1 = current[i1]
                    b_out2 = current[i2]
                    
                    # Calculate t-sets that would become uncovered if we remove A and B
                    # A tset is lost if its count becomes 0.
                    # Count becomes 0 if:
                    # 1. It was 1 and in A (and not in B)
                    # 2. It was 1 and in B (and not in A)
                    # 3. It was 2 and in both A and B
                    
                    # Efficiently:
                    # lost_tsets = set()
                    # for ts in block_to_tsets[b_out1]:
                    #     if cover_counts[ts] == 1:
                    #         lost_tsets.add(ts)
                    #     elif cover_counts[ts] == 2 and ts in block_to_tsets[b_out2]:
                    #         lost_tsets.add(ts)
                    # for ts in block_to_tsets[b_out2]:
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
                    # Heuristic: Maximize coverage of non-unique tsets with count 2
                    best_c = -1
                    best_add = -1
                    for c in list(candidates)[:20]:
                        add = 0
                        for ts in block_to_tsets[c]:
                            if cover_counts[ts] == 2:
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
                        
                        # Update current list
                        # Remove i1 and i2, add best_c
                        # Be careful with indices
                        if i1 < i2:
                            current.pop(i2)
                            current.pop(i1)
                        else:
                            current.pop(i1)
                            current.pop(i2)
                        current.append(best_c)
                        
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                        break

            if not improved:
                # No improvement in this iteration.
                # We could add a "kick" move: randomly change a block to a random valid swap
                # even if it doesn't improve immediately, to escape local minima.
                # But for now, let's just break out of the inner loop and restart?
                # Or just continue to next time check.
                # Since we have a time limit, we can just loop again.
                # However, if we are stuck, we might want to try a more aggressive restart.
                # For now, simple loop.
                pass

        return best_solution

    best_global_solution = None
    best_global_size = float('inf')
    
    # Run multiple instances
    seeds = [42, 123, 777, 1234, 5678, 9999, 1111, 2222]
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        t_start = time.time()
        init = greedy_cover(seed)
        if init:
            ls_time = min(20.0, deadline - t_start - 5)
            if ls_time > 5:
                sol = run_local_search(init, max_time_per_run=ls_time, seed=seed)
                if sol and len(sol) < best_global_size:
                    best_global_size = len(sol)
                    best_global_solution = sol
    
    # Final polish
    if best_global_solution and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 5:
            improved_sol = run_local_search(best_global_solution, max_time_per_run=remaining_time, seed=12345)
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
