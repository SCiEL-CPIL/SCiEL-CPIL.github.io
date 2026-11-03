import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin for safety
    
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
    # t_to_blocks[i] contains all block indices that cover t_subsets[i]
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for j in range(n_t_in_k):
            t_idx = block_t_indices[i, j]
            t_to_blocks[t_idx].append(i)
            
    # 4. Greedy Set Cover with Dynamic Gain Update
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Reset state
        selected_blocks = []
        selected_set = set()
        cover_count = np.zeros(n_t, dtype=np.int32)
        
        # current_gain[i] is the number of t-subsets in block i that are currently uncovered.
        # Initially, all are uncovered, so gain is C(k, t).
        current_gain = np.full(n_k, n_t_in_k, dtype=np.int32)
        
        # Greedy loop
        while True:
            if time.time() > deadline - 5.0:
                break
            
            # Find block with max gain
            # We can use argmax, but we need to handle ties or random selection among maxes.
            # For speed, just take the first max or a random one if we want diversity.
            # Standard greedy: pick max.
            max_gain = np.max(current_gain)
            if max_gain == 0:
                break
            
            # Find all blocks with max gain
            candidates = np.where(current_gain == max_gain)[0]
            best_idx = rng.choice(candidates)
            
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            # Update cover_count and current_gain
            # For each t-subset in best_idx:
            #   If it was uncovered (cover_count == 0), then all other blocks covering it lose 1 gain.
            #   Increment cover_count.
            ts_indices = block_t_indices[best_idx]
            for t_idx in ts_indices:
                if cover_count[t_idx] == 0:
                    # This t-subset is being covered for the first time.
                    # All blocks that cover this t-subset had it as an "uncovered" gain.
                    # So their gain decreases by 1.
                    for j in t_to_blocks[t_idx]:
                        current_gain[j] -= 1
                cover_count[t_idx] += 1
            
            # Mark selected block gain as 0 to avoid re-selection (though we use selected_set)
            current_gain[best_idx] = 0
            
        if time.time() > deadline - 5.0:
            break
            
        # Post-optimization: Local Search
        if selected_blocks:
            remaining = list(selected_blocks)
            remaining_set = set(remaining)
            
            improved = True
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
                        # Check if all t-subsets in bi are covered by > 1 block
                        if np.all(cover_count[ts] > 1):
                            cover_count[ts] -= 1
                            remaining.pop(i)
                            remaining_set.remove(bi)
                            changed = True
                            break # Restart scan
                
                if time.time() > deadline - 2.0:
                    break

                # 2. Try swaps
                # Iterate through a copy of remaining to avoid index issues
                for i in range(len(remaining)):
                    if time.time() > deadline - 2.0:
                        break
                    
                    bi = remaining[i]
                    ts = block_t_indices[bi]
                    
                    # Find unique t-subsets covered by bi (cover_count == 1)
                    unique_mask = cover_count[ts] == 1
                    unique_ts = ts[unique_mask]
                    
                    if len(unique_ts) == 0:
                        # This block is redundant
                        cover_count[ts] -= 1
                        remaining.pop(i)
                        remaining_set.remove(bi)
                        improved = True
                        break # Restart outer loop
                    
                    # Find candidates B that cover all unique_ts
                    # Intersection of t_to_blocks for each t in unique_ts
                    # Start with the one with smallest list to speed up intersection
                    # Sort unique_ts by length of t_to_blocks
                    sorted_unique_ts = sorted(unique_ts, key=lambda x: len(t_to_blocks[x]))
                    
                    candidates = set(t_to_blocks[sorted_unique_ts[0]])
                    for t_idx in sorted_unique_ts[1:]:
                        candidates.intersection_update(t_to_blocks[t_idx])
                        if not candidates:
                            break
                    
                    # Filter out blocks already in solution
                    candidates.difference_update(remaining_set)
                    
                    if candidates:
                        # Pick a random candidate
                        new_bi = rng.choice(list(candidates))
                        
                        # Perform swap
                        # Remove bi
                        cover_count[block_t_indices[bi]] -= 1
                        remaining_set.remove(bi)
                        remaining.pop(i)
                        
                        # Add new_bi
                        cover_count[block_t_indices[new_bi]] += 1
                        remaining_set.add(new_bi)
                        remaining.append(new_bi)
                        
                        improved = True
                        break # Restart outer loop after a swap to re-evaluate redundancies
            
            if len(remaining) < best_size:
                best_size = len(remaining)
                best_solution = [k_subsets[i] for i in remaining]
                
            # 3. Iterated Local Search (ILS) Perturbation
            if time.time() < deadline - 10.0 and best_solution is not None:
                # Start from a copy of the current best
                perturbed = list(best_solution_idx) if 'best_solution_idx' in locals() else [k_subsets.index(b) for b in best_solution]
                perturbed_set = set(perturbed)
                
                # Calculate coverage
                p_coverage = np.zeros(n_t, dtype=np.int32)
                for bi in perturbed:
                    p_coverage[block_t_indices[bi]] += 1
                
                # Delete 5-10 random blocks
                num_del = min(len(perturbed), rng.randint(5, 10))
                for _ in range(num_del):
                    if not perturbed:
                        break
                    idx = rng.randrange(len(perturbed))
                    bi = perturbed.pop(idx)
                    perturbed_set.remove(bi)
                    p_coverage[block_t_indices[bi]] -= 1
                
                # Repair using greedy (simplified: just add blocks until covered)
                while True:
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Find uncovered
                    uncovered_mask = p_coverage == 0
                    if not np.any(uncovered_mask):
                        break
                    
                    # Calculate gains for repair
                    # We can reuse the dynamic gain logic or just pick randomly from those covering uncovered
                    # For speed, pick a random block that covers at least one uncovered t-subset
                    # Or find the one covering the most.
                    # Let's find the one covering the most uncovered.
                    gains = uncovered_mask[block_t_indices].sum(axis=1)
                    max_gain = np.max(gains)
                    if max_gain == 0:
                        break
                        
                    candidates = np.where(gains == max_gain)[0]
                    # Filter out already selected
                    valid_candidates = [c for c in candidates if c not in perturbed_set]
                    if not valid_candidates:
                        break
                    
                    best_idx = rng.choice(valid_candidates)
                    perturbed.append(best_idx)
                    perturbed_set.add(best_idx)
                    p_coverage[block_t_indices[best_idx]] += 1
                
                # If repair succeeded, run local search on the new solution
                if not np.any(p_coverage == 0):
                    # Run local search on perturbed
                    improved_p = True
                    while improved_p:
                        if time.time() > deadline - 2.0:
                            break
                        improved_p = False
                        
                        # 1. Remove redundant blocks
                        changed = True
                        while changed:
                            if time.time() > deadline - 2.0:
                                break
                            changed = False
                            for i in range(len(perturbed)):
                                bi = perturbed[i]
                                ts = block_t_indices[bi]
                                if np.all(p_coverage[ts] > 1):
                                    p_coverage[ts] -= 1
                                    perturbed.pop(i)
                                    perturbed_set.remove(bi)
                                    changed = True
                                    break # Restart scan
                        
                        if time.time() > deadline - 2.0:
                            break

                        # 2. Try swaps
                        for i in range(len(perturbed)):
                            if time.time() > deadline - 2.0:
                                break
                            
                            bi = perturbed[i]
                            ts = block_t_indices[bi]
                            
                            unique_mask = p_coverage[ts] == 1
                            unique_ts = ts[unique_mask]
                            
                            if len(unique_ts) == 0:
                                p_coverage[ts] -= 1
                                perturbed.pop(i)
                                perturbed_set.remove(bi)
                                improved_p = True
                                break # Restart outer loop
                            
                            sorted_unique_ts = sorted(unique_ts, key=lambda x: len(t_to_blocks[x]))
                            candidates = set(t_to_blocks[sorted_unique_ts[0]])
                            for t_idx in sorted_unique_ts[1:]:
                                candidates.intersection_update(t_to_blocks[t_idx])
                                if not candidates:
                                    break
                            
                            candidates.difference_update(perturbed_set)
                            
                            if candidates:
                                new_bi = rng.choice(list(candidates))
                                
                                p_coverage[block_t_indices[bi]] -= 1
                                perturbed_set.remove(bi)
                                perturbed.pop(i)
                                
                                p_coverage[block_t_indices[new_bi]] += 1
                                perturbed_set.add(new_bi)
                                perturbed.append(new_bi)
                                
                                improved_p = True
                                break # Restart outer loop after a swap
                        
                    # Check if improved
                    if len(perturbed) < best_size:
                        best_size = len(perturbed)
                        best_solution = [k_subsets[i] for i in perturbed]
                        best_solution_idx = perturbed # Store indices for next perturbation start
                    else:
                        # If not improved, we might still want to keep the perturbed state for next iteration?
                        # Usually ILS keeps the best. Let's stick to best.
                        pass

    if best_solution is None:
        return []
        
    return [list(b) for b in best_solution]
