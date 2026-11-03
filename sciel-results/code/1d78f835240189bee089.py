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
    
    # Convert to numpy arrays for faster operations if needed, 
    # but lists are often faster for sparse access in Python loops.
    # Let's stick to lists for the core logic but use sets for fast intersection.
    
    # Helper to get a valid solution from block indices
    def get_solution(indices):
        return [list(all_blocks[i]) for i in indices]

    def check_validity(indices):
        # Exhaustive check is O(num_tsets * avg_blocks_per_tset)
        # We maintain counts, so we just check if all counts > 0
        counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in indices:
            for ts_idx in block_to_tsets[b_idx]:
                counts[ts_idx] += 1
        return np.all(counts > 0)

    def run_local_search(initial_indices, max_time_per_run=10.0):
        """
        Takes a list of block indices, applies local search improvements.
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
                
            # Phase 2: 1-for-1 Swaps with improved scoring
            for strategy in range(4):  # 4 different swap strategies
                if (time.time() - local_start) > max_time_per_run:
                    break
                    
                random.shuffle(current)
                
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
                    # Use set intersection for efficiency
                    candidates = set(tset_to_blocks[unique_tsets[0]])
                    candidates.discard(b_out)
                    
                    for ts_idx in unique_tsets[1:]:
                        if not candidates:
                            break
                        candidates = candidates.intersection(tset_to_blocks[ts_idx])
                    
                    if not candidates:
                        continue
                    
                    best_candidate = -1
                    best_score = -1
                    b_out_tsets_set = set(block_to_tsets[b_out])
                    
                    candidate_list = list(candidates)
                    # Use different limits per strategy
                    limit = [100, 200, 400, 700][strategy]
                    if len(candidate_list) > limit:
                        random.shuffle(candidate_list)
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
                                        score += 15
                                    elif cover_counts[ts_idx] == 2:
                                        score += 5
                                    else:
                                        score += 1
                                elif strategy == 2:
                                    # Hybrid: prioritize very sparse, then general
                                    if cover_counts[ts_idx] <= 2:
                                        score += 8
                                    elif cover_counts[ts_idx] == 3:
                                        score += 3
                                    else:
                                        score += 1
                                else:
                                    # Aggressive: heavily weight sparse coverage
                                    if cover_counts[ts_idx] == 1:
                                        score += 20
                                    elif cover_counts[ts_idx] == 2:
                                        score += 8
                                    elif cover_counts[ts_idx] == 3:
                                        score += 3
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
                    
            # Phase 3: 2-for-1 Swaps (remove 2 blocks, add 1) - escape local minima
            if (time.time() - local_start) < max_time_per_run - 2:
                random.shuffle(current)
                n = len(current)
                # Try a limited number of 2-for-1 swaps
                attempts = 0
                max_attempts = 50
                for i in range(n):
                    if (time.time() - local_start) > max_time_per_run or attempts >= max_attempts:
                        break
                    b_out1 = current[i]
                    # Find unique tsets of b_out1
                    unique1 = []
                    for ts_idx in block_to_tsets[b_out1]:
                        if cover_counts[ts_idx] == 1:
                            unique1.append(ts_idx)
                    if not unique1:
                        continue
                    for j in range(i+1, n):
                        if (time.time() - local_start) > max_time_per_run or attempts >= max_attempts:
                            break
                        b_out2 = current[j]
                        attempts += 1
                        # Find unique tsets of b_out2 (considering b_out1 removed)
                        # Temporarily reduce counts
                        for ts_idx in block_to_tsets[b_out1]:
                            cover_counts[ts_idx] -= 1
                        unique2 = []
                        for ts_idx in block_to_tsets[b_out2]:
                            if cover_counts[ts_idx] == 1:
                                unique2.append(ts_idx)
                        # Restore counts
                        for ts_idx in block_to_tsets[b_out1]:
                            cover_counts[ts_idx] += 1
                        
                        all_unique = unique1 + unique2
                        if not all_unique:
                            continue
                        
                        # Find candidate that covers all unique tsets
                        candidates = set(tset_to_blocks[all_unique[0]])
                        candidates.discard(b_out1)
                        candidates.discard(b_out2)
                        for ts_idx in all_unique[1:]:
                            if not candidates:
                                break
                            candidates = candidates.intersection(tset_to_blocks[ts_idx])
                        
                        if not candidates:
                            continue
                        
                        # Score the best candidate
                        best_candidate = -1
                        best_score = -1
                        out_tsets = set(block_to_tsets[b_out1]) | set(block_to_tsets[b_out2])
                        candidate_list = list(candidates)
                        if len(candidate_list) > 100:
                            random.shuffle(candidate_list)
                            candidate_list = candidate_list[:100]
                        
                        for b_in in candidate_list:
                            score = 0
                            for ts_idx in block_to_tsets[b_in]:
                                if ts_idx not in out_tsets:
                                    if cover_counts[ts_idx] == 1:
                                        score += 15
                                    elif cover_counts[ts_idx] == 2:
                                        score += 5
                                    else:
                                        score += 1
                            if score > best_score:
                                best_score = score
                                best_candidate = b_in
                        
                        if best_candidate != -1:
                            # Apply: remove both, add one
                            for ts_idx in block_to_tsets[b_out1]:
                                cover_counts[ts_idx] -= 1
                            for ts_idx in block_to_tsets[b_out2]:
                                cover_counts[ts_idx] -= 1
                            for ts_idx in block_to_tsets[best_candidate]:
                                cover_counts[ts_idx] += 1
                            
                            # Remove from current (j > i)
                            current.pop(j)
                            current.pop(i)
                            current.append(best_candidate)
                            improved = True
                            if len(current) < best_size:
                                best_size = len(current)
                                best_solution = list(current)
                            break  # Restart i loop after modification
            
            # Track no-improvement rounds to break out of local minima
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
            # Find best block using numpy argmax with masking
            masked_gains = gains.copy()
            masked_gains[selected_mask] = -1.0
            best_idx = int(np.argmax(masked_gains))
            best_gain = gains[best_idx]
            
            if best_gain <= 0:
                break
                
            selected.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update gains efficiently
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
        ls_time = min(45.0, deadline - t_start - 8)
        sol1 = run_local_search(init1, max_time_per_run=ls_time)
        if sol1 and len(sol1) < best_global_size:
            best_global_size = len(sol1)
            best_global_solution = sol1
            
    # Run 2: Greedy with different seed if time permits
    t_start = time.time()
    if time.time() < deadline - 28:
        init2 = greedy_cover(123)
        if init2:
            ls_time = min(25.0, deadline - t_start - 8)
            sol2 = run_local_search(init2, max_time_per_run=ls_time)
            if sol2 and len(sol2) < best_global_size:
                best_global_size = len(sol2)
                best_global_solution = sol2
    
    # Run 3: Another greedy with different seed if time permits
    t_start = time.time()
    if time.time() < deadline - 16:
        init3 = greedy_cover(777)
        if init3:
            ls_time = min(15.0, deadline - t_start - 5)
            sol3 = run_local_search(init3, max_time_per_run=ls_time)
            if sol3 and len(sol3) < best_global_size:
                best_global_size = len(sol3)
                best_global_solution = sol3
    
    # Run 4: Another greedy with different seed if time permits
    t_start = time.time()
    if time.time() < deadline - 10:
        init4 = greedy_cover(2024)
        if init4:
            ls_time = min(8.0, deadline - t_start - 3)
            sol4 = run_local_search(init4, max_time_per_run=ls_time)
            if sol4 and len(sol4) < best_global_size:
                best_global_size = len(sol4)
                best_global_solution = sol4
    
    # Final polish: Re-run local search on the best solution with remaining time
    if best_global_solution and time.time() < deadline - 3:
        remaining_time = deadline - time.time() - 1
        if remaining_time > 3:
            improved_sol = run_local_search(best_global_solution, max_time_per_run=remaining_time)
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
