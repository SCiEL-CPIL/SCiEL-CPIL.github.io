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
    
    # Precompute bitmasks for fast intersection/counting in local search
    # 11628 bits fits easily in Python ints
    block_masks = [0] * num_blocks
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)
            mask |= (1 << ts_idx)
        block_masks[b_idx] = mask

    # Helper to get a valid solution from block indices
    def get_solution(indices):
        return [list(all_blocks[i]) for i in indices]

    def check_validity(indices):
        counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in indices:
            for ts_idx in block_to_tsets[b_idx]:
                counts[ts_idx] += 1
        return np.all(counts > 0)

    def run_local_search(initial_indices, max_time_per_run=10.0, seed=0):
        """
        Takes a list of block indices, applies local search improvements.
        Returns the best list of indices found.
        """
        current = list(initial_indices)
        if not current:
            return []
        
        rng = random.Random(seed)
        
        # Calculate initial cover counts
        cover_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in current:
            for ts_idx in block_to_tsets[b_idx]:
                cover_counts[ts_idx] += 1
        
        best_size = len(current)
        best_solution = list(current)
        
        improved = True
        local_start = time.time()
        
        while improved and (time.time() - local_start) < max_time_per_run:
            improved = False
            
            # Phase 1: Remove redundant blocks (multiple passes until stable)
            pass_count = 0
            while True:
                pass_count += 1
                found_redundant = False
                # Shuffle order to avoid bias
                order = list(range(len(current)))
                rng.shuffle(order)
                
                for i in order:
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
                        found_redundant = True
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                        # Restart pass to check for new redundancies
                        break
                
                if not found_redundant or (time.time() - local_start) > max_time_per_run:
                    break
            
            if not current:
                break
                
            # Phase 2: 1-for-1 Swaps with improved scoring
            # Try multiple swap strategies
            for strategy in range(3):  # 3 different swap strategies
                if (time.time() - local_start) > max_time_per_run:
                    break
                    
                # Shuffle current to try different blocks
                current_copy = list(current)
                rng.shuffle(current_copy)
                
                for i in range(len(current_copy)):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                        
                    # Map shuffled index back to original index? 
                    # Actually, we can just operate on indices if we track them, 
                    # but since we pop, let's stick to iterating the list directly 
                    # but shuffle the *order of checks* by shuffling a list of indices 
                    # and then accessing current[idx]. 
                    # Wait, the previous code did `random.shuffle(current)` which 
                    # actually reorders the solution list. This is valid as long as 
                    # we keep track. Let's revert to shuffling the actual list to 
                    # match the "swap" logic which replaces `current[i]`.
                    
                    # Let's restart the shuffle logic properly.
                    # We want to try swapping different blocks.
                    pass # We will handle this below by shuffling current itself
                    
                # Re-doing the shuffle logic correctly
                # Shuffling the actual list changes the order of blocks in the solution, 
                # which is fine.
                rng.shuffle(current)
                
                for i in range(len(current)):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                        
                    b_out = current[i]
                    
                    # Find tsets uniquely covered by b_out
                    unique_tsets = []
                    for ts_idx in block_to_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            unique_tsets.append(ts_idx)
                            
                    if not unique_tsets:
                        continue

                    # Find candidate blocks B_in that cover all unique_tsets
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
                    # Use different limits per strategy
                    limit = [50, 100, 200][strategy]
                    if len(candidate_list) > limit:
                        rng.shuffle(candidate_list)
                        candidate_list = candidate_list[:limit]
                    
                    for b_in in candidate_list:
                        # Different scoring per strategy
                        score = 0
                        for ts_idx in block_to_tsets[b_in]:
                            if ts_idx not in b_out_tsets_set:
                                if strategy == 0:
                                    # Simple: count non-overlapping tsets
                                    score += 1
                                elif strategy == 1:
                                    # Weight by inverse count (prioritize sparse coverage)
                                    if cover_counts[ts_idx] == 1:
                                        score += 10
                                    else:
                                        score += 1
                                else:
                                    # Hybrid: prioritize very sparse, then general
                                    if cover_counts[ts_idx] <= 2:
                                        score += 5
                                    elif cover_counts[ts_idx] == 3:
                                        score += 2
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
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                    
            # Loop back to Phase 1 if swaps were made
        
        return best_solution

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []
        
        # Weights: 1 / frequency of tset
        tset_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in range(num_blocks):
            for ts_idx in block_to_tsets[b_idx]:
                tset_counts[ts_idx] += 1
        safe_counts = np.maximum(tset_counts, 1).astype(np.float64)
        tset_weights = 1.0 / safe_counts
        
        # Initial gains
        gains = np.zeros(num_blocks, dtype=np.float64)
        for b_idx in range(num_blocks):
            if block_to_tsets[b_idx]:
                gains[b_idx] = np.sum(tset_weights[block_to_tsets[b_idx]])
                
        selected_mask = np.zeros(num_blocks, dtype=np.bool_)
        
        while uncovered_count > 0 and time.time() < deadline - 15:
            # Find best block
            best_idx = -1
            best_gain = -1.0
            
            # Optimization: Use a heap? 
            # But updating gains is the bottleneck. 
            # Given N=75k, a simple scan is ~0.01s. 
            # We expect ~300-500 selections. 500 * 0.01 = 5s. Acceptable.
            
            for i in range(num_blocks):
                if not selected_mask[i] and gains[i] > best_gain:
                    best_gain = gains[i]
                    best_idx = i
                    
            if best_idx == -1 or best_gain <= 0:
                break
                
            selected.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update
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
    
    # Strategy: Run multiple greedy+local search instances with different seeds
    # and time limits.
    
    # Run 1: Standard greedy with extended local search
    t_start = time.time()
    init1 = greedy_cover(42)
    if init1:
        # Use more time for local search since it's the primary run
        ls_time = min(45.0, deadline - t_start - 10)
        sol1 = run_local_search(init1, max_time_per_run=ls_time, seed=42)
        if sol1 and len(sol1) < best_global_size:
            best_global_size = len(sol1)
            best_global_solution = sol1
            
    # Run 2: Greedy with different seed if time permits
    t_start = time.time()
    if time.time() < deadline - 35:
        init2 = greedy_cover(123)
        if init2:
            ls_time = min(25.0, deadline - t_start - 10)
            sol2 = run_local_search(init2, max_time_per_run=ls_time, seed=123)
            if sol2 and len(sol2) < best_global_size:
                best_global_size = len(sol2)
                best_global_solution = sol2
    
    # Run 3: Another greedy with different seed if time permits
    t_start = time.time()
    if time.time() < deadline - 20:
        init3 = greedy_cover(777)
        if init3:
            ls_time = min(15.0, deadline - t_start - 5)
            sol3 = run_local_search(init3, max_time_per_run=ls_time, seed=777)
            if sol3 and len(sol3) < best_global_size:
                best_global_size = len(sol3)
                best_global_solution = sol3
    
    # Final polish: Re-run local search on the best solution with remaining time
    if best_global_solution and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 5:
            # Use a different seed for the final polish to explore different moves
            improved_sol = run_local_search(best_global_solution, max_time_per_run=remaining_time, seed=999)
            if improved_sol and len(improved_sol) < best_global_size:
                best_global_size = len(improved_sol)
                best_global_solution = improved_sol
            
    if not best_global_solution:
        # Fallback: Return a valid but potentially large solution
        if init1:
            return get_solution(init1)
        return []
        
    return get_solution(best_global_solution)

if __name__ == "__main__":
    # Test with the specific parameters
    v, k, t = 19, 8, 5
    res = construct(v, k, t)
    print(f"Number of blocks: {len(res)}")
    # Verify
    covered = set()
    for block in res:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    total_tsets = len(list(itertools.combinations(range(v), t)))
    print(f"Covered: {len(covered)}/{total_tsets}")
