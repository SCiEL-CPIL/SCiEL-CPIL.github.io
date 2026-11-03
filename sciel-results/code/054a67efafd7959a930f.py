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
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Precompute block_to_tsets as numpy arrays for speed
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)
    
    block_to_tsets = np.empty((num_blocks, n_t_in_k), dtype=np.int32)
    for b_idx, block in enumerate(all_blocks):
        for j, lc in enumerate(t_in_k):
            block_to_tsets[b_idx, j] = tset_to_idx[tuple(block[p] for p in lc)]

    # tset_to_blocks: for each tset, which blocks cover it
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx in range(num_blocks):
        for ts_idx in block_to_tsets[b_idx]:
            tset_to_blocks[ts_idx].append(b_idx)
    
    # Convert to sets for fast intersection
    tset_to_blocks_sets = [set(lst) for lst in tset_to_blocks]

    # Helper to get a valid solution from block indices
    def get_solution(indices):
        return [list(all_blocks[i]) for i in indices]

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
            cover_counts[block_to_tsets[b_idx]] += 1
        
        best_size = len(current)
        best_solution = list(current)
        
        improved = True
        local_start = time.time()
        
        # Simulated annealing parameters
        sa_temperature = 50.0
        sa_cooling = 0.95
        min_temperature = 0.1
        
        while improved and (time.time() - local_start) < max_time_per_run:
            improved = False
            
            # Phase 1: Remove redundant blocks
            while True:
                if (time.time() - local_start) > max_time_per_run:
                    break
                
                found_redundant = False
                order = list(range(len(current)))
                rng.shuffle(order)
                
                for i in order:
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    b_idx = current[i]
                    is_redundant = True
                    for ts_idx in block_to_tsets[b_idx]:
                        if cover_counts[ts_idx] == 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        cover_counts[block_to_tsets[b_idx]] -= 1
                        current.pop(i)
                        found_redundant = True
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                        break
                
                if not found_redundant or (time.time() - local_start) > max_time_per_run:
                    break
            
            if not current:
                break
                
            # Phase 2: 1-for-1 Swaps
            for strategy in range(3):
                if (time.time() - local_start) > max_time_per_run:
                    break
                    
                rng.shuffle(current)
                
                for i in range(len(current)):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                        
                    b_out = current[i]
                    
                    unique_tsets = []
                    for ts_idx in block_to_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            unique_tsets.append(ts_idx)
                            
                    if not unique_tsets:
                        # This block is redundant
                        cover_counts[block_to_tsets[b_out]] -= 1
                        current.pop(i)
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                        break

                    candidates = tset_to_blocks_sets[unique_tsets[0]].copy()
                    for ts_idx in unique_tsets[1:]:
                        candidates.intersection_update(tset_to_blocks_sets[ts_idx])
                    candidates.discard(b_out)
                    
                    if not candidates:
                        continue
                    
                    best_candidate = -1
                    best_score = -1
                    b_out_tsets_set = set(block_to_tsets[b_out])
                    
                    candidate_list = list(candidates)
                    limit = [100, 200, 300][strategy]
                    if len(candidate_list) > limit:
                        rng.shuffle(candidate_list)
                        candidate_list = candidate_list[:limit]
                    
                    for b_in in candidate_list:
                        score = 0
                        for ts_idx in block_to_tsets[b_in]:
                            if ts_idx not in b_out_tsets_set:
                                if strategy == 0:
                                    score += 1
                                elif strategy == 1:
                                    if cover_counts[ts_idx] == 1:
                                        score += 10
                                    else:
                                        score += 1
                                else:
                                    # Simulated annealing: accept worse moves with probability
                                    if cover_counts[ts_idx] == 1:
                                        score += 5
                                    elif cover_counts[ts_idx] == 2:
                                        score += 2
                                    else:
                                        score += 1
                        if score > best_score:
                            best_score = score
                            best_candidate = b_in
                        
                        # Simulated annealing acceptance
                        elif sa_temperature > min_temperature and best_candidate != -1:
                            delta_score = score - best_score
                            if delta_score < 0:
                                accept_prob = np.exp(-(-delta_score) / sa_temperature)
                                if rng.random() < accept_prob:
                                    best_score = score
                                    best_candidate = b_in
                    
                    if best_candidate != -1:
                        cover_counts[block_to_tsets[b_out]] -= 1
                        cover_counts[block_to_tsets[best_candidate]] += 1
                        
                        current[i] = best_candidate
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                    
                    # Cool down after each swap
                    sa_temperature *= sa_cooling
            
            # Phase 3: 2-for-1 Merge (replace two blocks with one if possible)
            if (time.time() - local_start) < max_time_per_run and len(current) >= 2:
                num_2merge_tries = 50
                for _ in range(num_2merge_tries):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    
                    i1, i2 = rng.sample(range(len(current)), 2)
                    b1_out = current[i1]
                    b2_out = current[i2]
                    
                    # Find t-sets uniquely covered by b1 or b2
                    unique_tsets = []
                    for ts_idx in block_to_tsets[b1_out]:
                        if cover_counts[ts_idx] == 1:
                            unique_tsets.append(ts_idx)
                    for ts_idx in block_to_tsets[b2_out]:
                        if cover_counts[ts_idx] == 1 and cover_counts[ts_idx] - 1 == 0:
                            unique_tsets.append(ts_idx)
                    
                    if not unique_tsets:
                        continue
                    
                    # Find a single block that covers all unique t-sets
                    cand_set = tset_to_blocks_sets[unique_tsets[0]].copy()
                    cand_set.discard(b1_out)
                    cand_set.discard(b2_out)
                    for ts_idx in unique_tsets[1:]:
                        if not cand_set:
                            break
                        cand_set.intersection_update(tset_to_blocks_sets[ts_idx])
                    
                    if cand_set:
                        b_in = rng.choice(list(cand_set))
                        
                        # Verify all unique t-sets are covered
                        all_ok = True
                        b_in_tsets_set = set(block_to_tsets[b_in])
                        for ts_idx in unique_tsets:
                            if ts_idx not in b_in_tsets_set:
                                all_ok = False
                                break
                        
                        if all_ok:
                            # Remove b1_out and b2_out, add b_in
                            cover_counts[block_to_tsets[b1_out]] -= 1
                            cover_counts[block_to_tsets[b2_out]] -= 1
                            cover_counts[block_to_tsets[b_in]] += 1
                            
                            idxs_to_remove = sorted([i1, i2], reverse=True)
                            for idx in idxs_to_remove:
                                current.pop(idx)
                            current.append(b_in)
                            
                            improved = True
                            if len(current) < best_size:
                                best_size = len(current)
                                best_solution = list(current)
                            break
                
                if improved:
                    break
            
            # Phase 4: 2-for-2 Swaps
            if (time.time() - local_start) < max_time_per_run and len(current) >= 2:
                num_2swap_tries = 80
                for _ in range(num_2swap_tries):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    
                    i1, i2 = rng.sample(range(len(current)), 2)
                    b1_out = current[i1]
                    b2_out = current[i2]
                    
                    unique_tsets = []
                    for ts_idx in block_to_tsets[b1_out]:
                        if cover_counts[ts_idx] == 1:
                            unique_tsets.append(ts_idx)
                    for ts_idx in block_to_tsets[b2_out]:
                        if cover_counts[ts_idx] == 1:
                            unique_tsets.append(ts_idx)
                    
                    if not unique_tsets:
                        continue
                    
                    cand1_list = list(tset_to_blocks_sets[unique_tsets[0]] - {b1_out, b2_out})
                    if len(cand1_list) > 40:
                        rng.shuffle(cand1_list)
                        cand1_list = cand1_list[:40]
                    
                    for b1_in in cand1_list:
                        if (time.time() - local_start) > max_time_per_run:
                            break
                        
                        remaining_unique = []
                        b1_in_tsets_set = set(block_to_tsets[b1_in])
                        for ts_idx in unique_tsets:
                            if ts_idx not in b1_in_tsets_set:
                                if cover_counts[ts_idx] == 1:
                                    remaining_unique.append(ts_idx)
                        
                        if not remaining_unique:
                            # b1_in alone covers all unique t-sets
                            cover_counts[block_to_tsets[b1_out]] -= 1
                            cover_counts[block_to_tsets[b2_out]] -= 1
                            cover_counts[block_to_tsets[b1_in]] += 1
                            
                            idxs_to_remove = sorted([i1, i2], reverse=True)
                            for idx in idxs_to_remove:
                                current.pop(idx)
                            current.append(b1_in)
                            
                            improved = True
                            if len(current) < best_size:
                                best_size = len(current)
                                best_solution = list(current)
                            break
                        
                        cand2_set = tset_to_blocks_sets[remaining_unique[0]].copy()
                        cand2_set.discard(b1_out)
                        cand2_set.discard(b2_out)
                        cand2_set.discard(b1_in)
                        for ts_idx in remaining_unique[1:]:
                            if not cand2_set:
                                break
                            cand2_set.intersection_update(tset_to_blocks_sets[ts_idx])
                        
                        if not cand2_set:
                            continue
                        
                        cand2_list = list(cand2_set)
                        if len(cand2_list) > 25:
                            rng.shuffle(cand2_list)
                            cand2_list = cand2_list[:25]
                        
                        for b2_in in cand2_list:
                            b2_in_tsets_set = set(block_to_tsets[b2_in])
                            all_covered = True
                            for ts_idx in unique_tsets:
                                if ts_idx not in b1_in_tsets_set and ts_idx not in b2_in_tsets_set:
                                    all_covered = False
                                    break
                            
                            if not all_covered:
                                continue
                            
                            ok = True
                            for ts_idx in block_to_tsets[b1_out]:
                                if cover_counts[ts_idx] == 1 and ts_idx not in b1_in_tsets_set and ts_idx not in b2_in_tsets_set:
                                    ok = False
                                    break
                            if ok:
                                for ts_idx in block_to_tsets[b2_out]:
                                    if cover_counts[ts_idx] == 1 and ts_idx not in b1_in_tsets_set and ts_idx not in b2_in_tsets_set:
                                        ok = False
                                        break
                            
                            if not ok:
                                continue
                            
                            cover_counts[block_to_tsets[b1_out]] -= 1
                            cover_counts[block_to_tsets[b2_out]] -= 1
                            cover_counts[block_to_tsets[b1_in]] += 1
                            cover_counts[block_to_tsets[b2_in]] += 1
                            
                            idxs_to_remove = sorted([i1, i2], reverse=True)
                            for idx in idxs_to_remove:
                                current.pop(idx)
                            current.append(b1_in)
                            current.append(b2_in)
                            
                            improved = True
                            if len(current) < best_size:
                                best_size = len(current)
                                best_solution = list(current)
                            break
                        
                        if improved:
                            break
                
                if improved:
                    break
            
            # Reset temperature occasionally to escape local minima
            if rng.random() < 0.1:
                sa_temperature = max(sa_temperature, 20.0)
        
        return best_solution

    def grasp_cover(seed, alpha=0.3, max_time=30.0):
        """
        GRASP: Greedy Randomized Adaptive Search Procedure.
        At each step, build an RCL from top candidates and pick randomly.
        """
        rng = random.Random(seed)
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []
        selected_mask = np.zeros(num_blocks, dtype=np.bool_)
        
        # Initial gains: number of uncovered t-sets each block covers
        gains = np.zeros(num_blocks, dtype=np.int32)
        for b_idx in range(num_blocks):
            gains[b_idx] = np.sum(uncovered[block_to_tsets[b_idx]])
        
        while uncovered_count > 0 and time.time() < start_time + max_time:
            # Find max gain
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break
            
            # Build RCL: blocks with gain >= (1 - alpha) * max_gain
            threshold = (1 - alpha) * max_gain
            candidates = np.where((gains >= threshold) & (~selected_mask))[0]
            
            if len(candidates) == 0:
                break
            
            # Pick randomly from RCL
            best_idx = int(rng.choice(candidates))
            
            selected.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update gains
            for ts_idx in block_to_tsets[best_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_count -= 1
                    for b in tset_to_blocks[ts_idx]:
                        if not selected_mask[b]:
                            gains[b] -= 1
        
        return selected

    def greedy_cover(seed, max_time=30.0):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []
        
        # Weights: 1 / frequency of tset
        tset_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in range(num_blocks):
            tset_counts[block_to_tsets[b_idx]] += 1
        safe_counts = np.maximum(tset_counts, 1).astype(np.float64)
        tset_weights = 1.0 / safe_counts
        
        # Initial gains
        gains = np.zeros(num_blocks, dtype=np.float64)
        for b_idx in range(num_blocks):
            if n_t_in_k > 0:
                gains[b_idx] = np.sum(tset_weights[block_to_tsets[b_idx]])
                
        selected_mask = np.zeros(num_blocks, dtype=np.bool_)
        
        while uncovered_count > 0 and time.time() < start_time + max_time:
            masked_gains = np.where(selected_mask, -1.0, gains)
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

    def lagrangian_greedy_cover(seed, num_lr_iterations=200, max_time=60.0):
        """
        Lagrangian Relaxation with subgradient optimization to guide greedy selection.
        """
        rng = random.Random(seed)
        
        # Initialize penalty weights uniformly
        penalty = np.ones(num_tsets, dtype=np.float64)
        
        best_solution = None
        best_size = float('inf')
        
        lr_rate = 1.0
        best_lb = -1.0
        
        for lr_iter in range(num_lr_iterations):
            if time.time() > start_time + max_time:
                break
            
            # Phase 1: Greedy selection using current penalty weights
            selected = []
            selected_mask = np.zeros(num_blocks, dtype=np.bool_)
            uncovered = np.ones(num_tsets, dtype=np.bool_)
            uncovered_count = num_tsets
            
            # Compute block scores based on penalty weights
            block_scores = np.zeros(num_blocks, dtype=np.float64)
            for b_idx in range(num_blocks):
                if n_t_in_k > 0:
                    block_scores[b_idx] = np.sum(penalty[block_to_tsets[b_idx]])
            
            while uncovered_count > 0 and time.time() < start_time + max_time:
                masked_scores = np.where(selected_mask, -1e10, block_scores)
                best_idx = int(np.argmax(masked_scores))
                best_gain = block_scores[best_idx]
                
                if best_gain <= 0:
                    break
                
                selected.append(best_idx)
                selected_mask[best_idx] = True
                
                for ts_idx in block_to_tsets[best_idx]:
                    if uncovered[ts_idx]:
                        uncovered[ts_idx] = False
                        uncovered_count -= 1
                        w = penalty[ts_idx]
                        for b in tset_to_blocks[ts_idx]:
                            if not selected_mask[b]:
                                block_scores[b] -= w
            
            # Check if we have a valid solution
            if uncovered_count == 0:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = list(selected)
            
            # Compute subgradient
            subgradient = np.zeros(num_tsets, dtype=np.float64)
            for ts_idx in range(num_tsets):
                if uncovered[ts_idx]:
                    subgradient[ts_idx] = 1.0
            
            current_lb = uncovered_count
            
            if current_lb > best_lb:
                best_lb = current_lb
                lr_rate *= 0.95
            
            step = lr_rate / (best_lb + 1.0)
            penalty = np.maximum(0, penalty + step * subgradient)
            
            # Add controlled noise for exploration
            if lr_iter % 15 == 14:
                noise_scale = max(0.05, 0.2 * (1.0 - lr_iter / num_lr_iterations))
                noise_vec = np.array([rng.random() for _ in range(num_tsets)])
                penalty = penalty + noise_scale * noise_vec
                penalty = np.maximum(0.01, penalty)
        
        return best_solution

    best_global_solution = None
    best_global_size = float('inf')
    
    # Strategy 1: GRASP with different alpha values
    if time.time() < deadline - 50:
        for alpha in [0.2, 0.3, 0.4]:
            if time.time() > deadline - 45:
                break
            grasp_sol = grasp_cover(42 + int(alpha * 100), alpha=alpha, max_time=15.0)
            if grasp_sol:
                ls_time = min(20.0, deadline - time.time() - 10)
                if ls_time > 5:
                    polished = run_local_search(grasp_sol, max_time_per_run=ls_time, seed=42 + int(alpha * 100))
                    if polished and len(polished) < best_global_size:
                        best_global_size = len(polished)
                        best_global_solution = polished
    
    # Strategy 2: Lagrangian Relaxation approach
    if time.time() < deadline - 40:
        lr_sol = lagrangian_greedy_cover(42, num_lr_iterations=120, max_time=30.0)
        if lr_sol:
            ls_time = min(20.0, deadline - time.time() - 10)
            if ls_time > 5:
                polished = run_local_search(lr_sol, max_time_per_run=ls_time, seed=42)
                if polished and len(polished) < best_global_size:
                    best_global_size = len(polished)
                    best_global_solution = polished
            elif lr_sol and len(lr_sol) < best_global_size:
                best_global_size = len(lr_sol)
                best_global_solution = lr_sol
    
    # Strategy 3: Standard greedy with local search
    if time.time() < deadline - 25:
        init1 = greedy_cover(42, max_time=15.0)
        if init1:
            ls_time = min(15.0, deadline - time.time() - 8)
            if ls_time > 5:
                sol1 = run_local_search(init1, max_time_per_run=ls_time, seed=42)
                if sol1 and len(sol1) < best_global_size:
                    best_global_size = len(sol1)
                    best_global_solution = sol1
    
    # Strategy 4: Another GRASP with different seed
    if time.time() < deadline - 15:
        grasp_sol2 = grasp_cover(123, alpha=0.35, max_time=10.0)
        if grasp_sol2:
            ls_time = min(10.0, deadline - time.time() - 5)
            if ls_time > 3:
                polished2 = run_local_search(grasp_sol2, max_time_per_run=ls_time, seed=123)
                if polished2 and len(polished2) < best_global_size:
                    best_global_size = len(polished2)
                    best_global_solution = polished2
    
    # Final polish: Re-run local search on the best solution with remaining time
    if best_global_solution and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 5:
            # Multiple polish passes with different seeds for diversity
            for polish_seed in [999, 777, 555, 333]:
                if time.time() > deadline - 3:
                    break
                pass_time = min(remaining_time / 4.0, 15.0)
                if pass_time < 2:
                    break
                improved_sol = run_local_search(best_global_solution, max_time_per_run=pass_time, seed=polish_seed)
                if improved_sol and len(improved_sol) < best_global_size:
                    best_global_size = len(improved_sol)
                    best_global_solution = improved_sol
            
    if not best_global_solution:
        # Fallback
        init_fallback = greedy_cover(42, max_time=30.0)
        if init_fallback:
            return get_solution(init_fallback)
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
