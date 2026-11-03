import itertools
import time
import numpy as np
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s limit (evaluator kills at 150s)
    
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
    
    # Helper to get a valid solution from block indices
    def get_solution(indices):
        return [list(all_blocks[i]) for i in indices]

    def check_validity(indices):
        counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in indices:
            for ts_idx in block_to_tsets[b_idx]:
                counts[ts_idx] += 1
        return np.all(counts > 0)

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
        
        improved = True
        local_start = time.time()
        no_improvement_rounds = 0
        max_no_improvement = 3
        
        while improved and (time.time() - local_start) < max_time_per_run:
            improved = False
            
            # Phase 1: Remove redundant blocks
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
            for strategy in range(4):
                if (time.time() - local_start) > max_time_per_run:
                    break
                    
                random.shuffle(current)
                
                for i in range(len(current)):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                        
                    b_out = current[i]
                    unique_tsets = []
                    for ts_idx in block_to_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            unique_tsets.append(ts_idx)
                            
                    if not unique_tsets:
                        continue

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
                    limit = [100, 200, 400, 700][strategy]
                    if len(candidate_list) > limit:
                        random.shuffle(candidate_list)
                        candidate_list = candidate_list[:limit]
                    
                    for b_in in candidate_list:
                        score = 0
                        for ts_idx in block_to_tsets[b_in]:
                            if ts_idx not in b_out_tsets_set:
                                if strategy == 0:
                                    score += 1
                                elif strategy == 1:
                                    if cover_counts[ts_idx] == 1:
                                        score += 15
                                    elif cover_counts[ts_idx] == 2:
                                        score += 5
                                    else:
                                        score += 1
                                elif strategy == 2:
                                    if cover_counts[ts_idx] <= 2:
                                        score += 8
                                    elif cover_counts[ts_idx] == 3:
                                        score += 3
                                    else:
                                        score += 1
                                else:
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
                    
            # Phase 3: 2-for-1 Swaps
            if (time.time() - local_start) < max_time_per_run - 2:
                random.shuffle(current)
                n = len(current)
                attempts = 0
                max_attempts = 50
                for i in range(n):
                    if (time.time() - local_start) > max_time_per_run or attempts >= max_attempts:
                        break
                    b_out1 = current[i]
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
                        for ts_idx in block_to_tsets[b_out1]:
                            cover_counts[ts_idx] -= 1
                        unique2 = []
                        for ts_idx in block_to_tsets[b_out2]:
                            if cover_counts[ts_idx] == 1:
                                unique2.append(ts_idx)
                        for ts_idx in block_to_tsets[b_out1]:
                            cover_counts[ts_idx] += 1
                        
                        all_unique = unique1 + unique2
                        if not all_unique:
                            continue
                        
                        candidates = set(tset_to_blocks[all_unique[0]])
                        candidates.discard(b_out1)
                        candidates.discard(b_out2)
                        for ts_idx in all_unique[1:]:
                            if not candidates:
                                break
                            candidates = candidates.intersection(tset_to_blocks[ts_idx])
                        
                        if not candidates:
                            continue
                        
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
                            for ts_idx in block_to_tsets[b_out1]:
                                cover_counts[ts_idx] -= 1
                            for ts_idx in block_to_tsets[b_out2]:
                                cover_counts[ts_idx] -= 1
                            for ts_idx in block_to_tsets[best_candidate]:
                                cover_counts[ts_idx] += 1
                            
                            current.pop(j)
                            current.pop(i)
                            current.append(best_candidate)
                            improved = True
                            if len(current) < best_size:
                                best_size = len(current)
                                best_solution = list(current)
                            break
            
            if not improved:
                no_improvement_rounds += 1
                if no_improvement_rounds >= max_no_improvement:
                    break
            else:
                no_improvement_rounds = 0
        
        return best_solution

    def greedy_cover(seed, use_lagrangian=False):
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
        
        if use_lagrangian:
            # Lagrangian relaxation: initialize with higher weights for rare tsets
            # and update weights based on uncovered status
            lag_weights = tset_weights.copy()
        else:
            lag_weights = tset_weights.copy()
        
        # Initial gains
        gains = np.zeros(num_blocks, dtype=np.float64)
        for b_idx in range(num_blocks):
            if block_to_tsets[b_idx]:
                gains[b_idx] = np.sum(lag_weights[block_to_tsets[b_idx]])
                
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
                    w = lag_weights[ts_idx]
                    for b in tset_to_blocks[ts_idx]:
                        if not selected_mask[b]:
                            gains[b] -= w
            
            # Lagrangian update: increase weights for still-uncovered tsets
            if use_lagrangian and uncovered_count > 0:
                # Subgradient step: increase weight for uncovered tsets
                for ts_idx in range(num_tsets):
                    if uncovered[ts_idx]:
                        lag_weights[ts_idx] *= 1.1
                # Re-normalize to prevent overflow
                max_w = np.max(lag_weights)
                if max_w > 1000:
                    lag_weights /= max_w / 1000.0
                # Update gains for all unselected blocks
                for b_idx in range(num_blocks):
                    if not selected_mask[b_idx]:
                        gains[b_idx] = np.sum(lag_weights[block_to_tsets[b_idx]])
                            
        return selected

    best_global_solution = None
    best_global_size = float('inf')
    
    # Run 1: Standard greedy with extended local search
    t_start = time.time()
    init1 = greedy_cover(42, use_lagrangian=False)
    if init1:
        ls_time = min(40.0, deadline - t_start - 10)
        sol1 = run_local_search(init1, max_time_per_run=ls_time)
        if sol1 and len(sol1) < best_global_size:
            best_global_size = len(sol1)
            best_global_solution = sol1
            
    # Run 2: Lagrangian greedy with local search
    t_start = time.time()
    if time.time() < deadline - 30:
        init2 = greedy_cover(123, use_lagrangian=True)
        if init2:
            ls_time = min(25.0, deadline - t_start - 10)
            sol2 = run_local_search(init2, max_time_per_run=ls_time)
            if sol2 and len(sol2) < best_global_size:
                best_global_size = len(sol2)
                best_global_solution = sol2
    
    # Run 3: Another greedy with different seed
    t_start = time.time()
    if time.time() < deadline - 18:
        init3 = greedy_cover(777, use_lagrangian=False)
        if init3:
            ls_time = min(12.0, deadline - t_start - 8)
            sol3 = run_local_search(init3, max_time_per_run=ls_time)
            if sol3 and len(sol3) < best_global_size:
                best_global_size = len(sol3)
                best_global_solution = sol3
    
    # Run 4: Another lagrangian greedy
    t_start = time.time()
    if time.time() < deadline - 12:
        init4 = greedy_cover(2024, use_lagrangian=True)
        if init4:
            ls_time = min(8.0, deadline - t_start - 5)
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
        if init1:
            return get_solution(init1)
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
