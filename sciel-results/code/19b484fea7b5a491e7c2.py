import itertools
import time
import numpy as np
import random
import sys
from collections import defaultdict
import heapq

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
        
        while improved and (time.time() - local_start) < max_time_per_run:
            improved = False
            
            # Phase 1: Remove redundant blocks (multiple passes until stable)
            pass_count = 0
            while True:
                pass_count += 1
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
            # Try multiple swap strategies
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
                    limit = [80, 150, 300, 500][strategy]
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
                                        score += 10
                                    else:
                                        score += 1
                                elif strategy == 2:
                                    # Hybrid: prioritize very sparse, then general
                                    if cover_counts[ts_idx] <= 2:
                                        score += 5
                                    elif cover_counts[ts_idx] == 3:
                                        score += 2
                                    else:
                                        score += 1
                                else:
                                    # Aggressive: strongly prefer sparse, penalize dense
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
        
        # Use a max-heap with lazy deletion for efficient best-block selection
        # Heap entries: (-gain, block_idx, version)
        # We'll use a version counter to detect stale entries
        block_version = np.zeros(num_blocks, dtype=np.int32)
        heap = []
        for b_idx in range(num_blocks):
            heapq.heappush(heap, (-gains[b_idx], b_idx, 0))
        
        while uncovered_count > 0 and time.time() < deadline - 15:
            # Find best block from heap (with lazy deletion)
            best_idx = -1
            best_gain = -1.0
            
            while heap:
                neg_gain, b_idx, version = heapq.heappop(heap)
                if selected_mask[b_idx]:
                    continue
                if version != block_version[b_idx]:
                    continue
                best_idx = b_idx
                best_gain = -neg_gain
                break
            
            if best_idx == -1 or best_gain <= 0:
                break
                
            selected.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update: for each tset covered by best_idx that was uncovered,
            # subtract its weight from all blocks covering that tset
            for ts_idx in block_to_tsets[best_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_count -= 1
                    w = tset_weights[ts_idx]
                    for b in tset_to_blocks[ts_idx]:
                        if not selected_mask[b]:
                            old_gain = gains[b]
                            gains[b] -= w
                            block_version[b] += 1
                            heapq.heappush(heap, (-gains[b], b, block_version[b]))
                            
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
        ls_time = min(50.0, deadline - t_start - 10)
        sol1 = run_local_search(init1, max_time_per_run=ls_time)
        if sol1 and len(sol1) < best_global_size:
            best_global_size = len(sol1)
            best_global_solution = sol1
            
    # Run 2: Greedy with different seed if time permits
    t_start = time.time()
    if time.time() < deadline - 40:
        init2 = greedy_cover(123)
        if init2:
            ls_time = min(30.0, deadline - t_start - 10)
            sol2 = run_local_search(init2, max_time_per_run=ls_time)
            if sol2 and len(sol2) < best_global_size:
                best_global_size = len(sol2)
                best_global_solution = sol2
    
    # Run 3: Another greedy with different seed if time permits
    t_start = time.time()
    if time.time() < deadline - 25:
        init3 = greedy_cover(777)
        if init3:
            ls_time = min(20.0, deadline - t_start - 5)
            sol3 = run_local_search(init3, max_time_per_run=ls_time)
            if sol3 and len(sol3) < best_global_size:
                best_global_size = len(sol3)
                best_global_solution = sol3
    
    # Final polish: Re-run local search on the best solution with remaining time
    if best_global_solution and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 5:
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
