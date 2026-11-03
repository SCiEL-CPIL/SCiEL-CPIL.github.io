import itertools
import numpy as np
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety
    
    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # 2. Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # 3. Precompute block_t_indices: (n_k, C(k, t))
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)
    
    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    
    # Filling the array
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]
            
    # Precompute t_to_blocks: list of lists of block indices
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for j in range(n_t_in_k):
            t_idx = block_t_indices[i, j]
            t_to_blocks[t_idx].append(i)
            
    t_to_blocks_sets = [set(lst) for lst in t_to_blocks]
    
    # 4. Greedy Set Cover with Multiple Restarts and Enhanced Local Search
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    def try_local_search(remaining, remaining_set, coverage_count, rng, deadline):
        """Enhanced local search: redundancy removal, swaps, 2-for-1, and simulated annealing"""
        improved = True
        sa_temperature = 0.1
        sa_min_temp = 0.001
        sa_cooldown = 0
        
        while improved:
            if time.time() > deadline - 2.0:
                break
            improved = False
            
            # 1. Remove redundant blocks
            changed = True
            while changed:
                if time.time() > deadline - 2.0:
                    break
                changed = False
                for i in range(len(remaining)):
                    bi = remaining[i]
                    ts = block_t_indices[bi]
                    if np.all(coverage_count[ts] > 1):
                        coverage_count[ts] -= 1
                        remaining.pop(i)
                        remaining_set.remove(bi)
                        changed = True
                        break
            
            if time.time() > deadline - 2.0:
                break

            # 2. Try 2-for-1: replace two blocks with one if possible
            n_rem = len(remaining)
            if n_rem >= 2:
                # Sample a few pairs to check
                pairs_to_try = min(20, n_rem * (n_rem - 1) // 2)
                for _ in range(pairs_to_try):
                    if time.time() > deadline - 2.0:
                        break
                    i1, i2 = rng.sample(range(n_rem), 2)
                    b1, b2 = remaining[i1], remaining[i2]
                    ts1 = block_t_indices[b1]
                    ts2 = block_t_indices[b2]
                    
                    # Find t-subsets uniquely covered by b1 or b2
                    unique_to_b1 = ts1[(coverage_count[ts1] == 1) & ~np.isin(ts1, ts2)]
                    unique_to_b2 = ts2[(coverage_count[ts2] == 1) & ~np.isin(ts2, ts1)]
                    shared_unique = (ts1[(coverage_count[ts1] == 1) & np.isin(ts1, ts2)])
                    
                    all_unique = np.concatenate([unique_to_b1, unique_to_b2, shared_unique])
                    
                    if len(all_unique) > 0:
                        # Find a block that covers all these unique t-subsets
                        candidates = t_to_blocks_sets[all_unique[0]]
                        for t_idx in all_unique[1:]:
                            candidates = candidates.intersection(t_to_blocks_sets[t_idx])
                        candidates.difference_update(remaining_set)
                        
                        if candidates:
                            new_b = rng.choice(list(candidates))
                            # Check if new_b doesn't create issues
                            coverage_count[block_t_indices[b1]] -= 1
                            coverage_count[block_t_indices[b2]] -= 1
                            remaining_set.remove(b1)
                            remaining_set.remove(b2)
                            remaining.pop(max(i1, i2))
                            remaining.pop(min(i1, i2))
                            
                            coverage_count[block_t_indices[new_b]] += 1
                            remaining_set.add(new_b)
                            remaining.append(new_b)
                            improved = True
                            break
            if improved:
                break

            # 3. Try swaps with multiple candidate attempts
            for i in range(len(remaining)):
                if time.time() > deadline - 2.0:
                    break
                
                bi = remaining[i]
                ts = block_t_indices[bi]
                unique_ts = ts[coverage_count[ts] == 1]
                
                if len(unique_ts) == 0:
                    coverage_count[ts] -= 1
                    remaining.pop(i)
                    remaining_set.remove(bi)
                    improved = True
                    break
                
                # Find candidates
                candidates = t_to_blocks_sets[unique_ts[0]]
                for t_idx in unique_ts[1:]:
                    candidates = candidates.intersection(t_to_blocks_sets[t_idx])
                candidates.difference_update(remaining_set)
                
                if candidates:
                    # Try multiple candidates with simulated annealing
                    cand_list = list(candidates)
                    rng.shuffle(cand_list)
                    max_attempts = min(5, len(cand_list))
                    
                    for new_bi in cand_list[:max_attempts]:
                        # Evaluate: does this swap reduce total "waste"?
                        # Simple heuristic: prefer blocks that cover more already-covered t-subsets
                        old_waste = np.sum(coverage_count[block_t_indices[bi]] > 1)
                        new_waste = np.sum(coverage_count[block_t_indices[new_bi]] > 1)
                        
                        # Accept if better or with probability based on temperature
                        if new_waste < old_waste:
                            # Perform swap
                            coverage_count[block_t_indices[bi]] -= 1
                            remaining_set.remove(bi)
                            remaining.pop(i)
                            
                            coverage_count[block_t_indices[new_bi]] += 1
                            remaining_set.add(new_bi)
                            remaining.append(new_bi)
                            improved = True
                            break
                        elif sa_temperature > sa_min_temp and rng.random() < math.exp((old_waste - new_waste) / sa_temperature):
                            # Accept worse solution with probability
                            coverage_count[block_t_indices[bi]] -= 1
                            remaining_set.remove(bi)
                            remaining.pop(i)
                            
                            coverage_count[block_t_indices[new_bi]] += 1
                            remaining_set.add(new_bi)
                            remaining.append(new_bi)
                            improved = True
                            break
                    if improved:
                        break
            
            # Cool down simulated annealing
            if not improved:
                sa_temperature *= 0.9
                sa_cooldown += 1
                if sa_cooldown > 10:
                    break
            else:
                sa_cooldown = 0
        
        return remaining, remaining_set, coverage_count

    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Reset state
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []
        selected_set = set()
        
        # Greedy loop with some randomization
        while True:
            if time.time() > deadline - 5.0:
                break
            
            # Calculate gains
            gains = uncovered[block_t_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly for diversity
            best_idx = rng.choice(candidates)
            
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            # Update uncovered
            uncovered[block_t_indices[best_idx]] = False
            
        if time.time() > deadline - 5.0:
            break
            
        # Post-optimization: Enhanced Local Search
        if selected_blocks:
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bi in selected_blocks:
                coverage_count[block_t_indices[bi]] += 1
                
            remaining = list(selected_blocks)
            remaining_set = set(remaining)
            
            remaining, remaining_set, coverage_count = try_local_search(
                remaining, remaining_set, coverage_count, rng, deadline
            )
            
            if len(remaining) < best_size:
                best_size = len(remaining)
                best_solution = [k_subsets[i] for i in remaining]
                
    if best_solution is None:
        return []
        
    return [list(b) for b in best_solution]
