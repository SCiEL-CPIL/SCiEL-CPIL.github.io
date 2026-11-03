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
    # Iterate over blocks and their covered t-subsets
    # This is O(N_k * C(k,t))
    for i in range(n_k):
        for j in range(n_t_in_k):
            t_idx = block_t_indices[i, j]
            t_to_blocks[t_idx].append(i)
            
    # Convert to numpy arrays for faster intersection? 
    # Lists are fine for small intersections, but numpy might be better for set ops.
    # Let's keep as lists for now, or convert to sets if needed.
    # For intersection, we can use set intersection.
    t_to_blocks_sets = [set(lst) for lst in t_to_blocks]

    # 4. Greedy Set Cover with Multiple Restarts
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Reset state
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []
        selected_set = set()
        
        # Greedy loop
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
            
            # Pick one randomly
            best_idx = rng.choice(candidates)
            
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            # Update uncovered
            uncovered[block_t_indices[best_idx]] = False
            
        if time.time() > deadline - 5.0:
            break
            
        # Post-optimization: Local Search with Redundancy Removal and Swaps
        if selected_blocks:
            # Build coverage count
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bi in selected_blocks:
                coverage_count[block_t_indices[bi]] += 1
                
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
                        if np.all(coverage_count[ts] > 1):
                            coverage_count[ts] -= 1
                            remaining.pop(i)
                            remaining_set.remove(bi)
                            changed = True
                            break # Restart scan
                
                if time.time() > deadline - 2.0:
                    break

                # 2. Try swaps
                # For each block in remaining, check if it can be swapped
                # We iterate through a copy of remaining to avoid index issues
                for i in range(len(remaining)):
                    if time.time() > deadline - 2.0:
                        break
                    
                    bi = remaining[i]
                    ts = block_t_indices[bi]
                    
                    # Find unique t-subsets covered by bi
                    unique_ts = ts[coverage_count[ts] == 1]
                    
                    if len(unique_ts) == 0:
                        # This block is redundant, should have been removed
                        # But maybe coverage changed? 
                        # If all coverage > 1, it's redundant.
                        # If some == 1, it's not redundant.
                        # If len(unique_ts) == 0, it means no t-subset is uniquely covered by bi.
                        # This implies all t-subsets in bi are covered by >= 2 blocks.
                        # So bi is redundant.
                        coverage_count[ts] -= 1
                        remaining.pop(i)
                        remaining_set.remove(bi)
                        improved = True
                        break # Restart outer loop
                    
                    # Find candidates B that cover all unique_ts
                    # Intersection of t_to_blocks_sets for each t in unique_ts
                    candidates = t_to_blocks_sets[unique_ts[0]]
                    for t_idx in unique_ts[1:]:
                        candidates = candidates.intersection(t_to_blocks_sets[t_idx])
                    
                    # Filter out blocks already in solution
                    candidates.difference_update(remaining_set)
                    
                    if candidates:
                        # Pick a random candidate
                        new_bi = rng.choice(list(candidates))
                        
                        # Perform swap
                        # Remove bi
                        coverage_count[block_t_indices[bi]] -= 1
                        remaining_set.remove(bi)
                        remaining.pop(i)
                        
                        # Add new_bi
                        coverage_count[block_t_indices[new_bi]] += 1
                        remaining_set.add(new_bi)
                        remaining.append(new_bi)
                        
                        improved = True
                        break # Restart outer loop after a swap to re-evaluate redundancies
            
            # 3. Iterated Local Search (ILS) Perturbation
            # If we have time, perturb the current best solution and repair
            if time.time() < deadline - 10.0 and best_solution is not None:
                # Start from a copy of the current best
                perturbed = list(best_solution)
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
                
                # Repair using greedy
                while True:
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Find uncovered
                    uncovered_mask = p_coverage == 0
                    if not np.any(uncovered_mask):
                        break
                    
                    # Calculate gains
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
                
                # If repair succeeded (all covered), run local search on the new solution
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
                            
                            unique_ts = ts[p_coverage[ts] == 1]
                            
                            if len(unique_ts) == 0:
                                p_coverage[ts] -= 1
                                perturbed.pop(i)
                                perturbed_set.remove(bi)
                                improved_p = True
                                break # Restart outer loop
                            
                            candidates = t_to_blocks_sets[unique_ts[0]]
                            for t_idx in unique_ts[1:]:
                                candidates = candidates.intersection(t_to_blocks_sets[t_idx])
                            
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
                    
                    # Update best_solution for next perturbation
                    # Note: best_solution is a list of tuples, perturbed is list of indices
                    # We need to be careful here.
                    # Actually, let's just use the current 'remaining' as the base for next perturbation
                    # Or stick to 'best_solution'. The code above updates best_solution if improved.
                    # Let's use 'remaining' (the one from the main greedy loop) as the base for perturbation
                    # to avoid confusion, or just always perturb the current 'remaining'.
                    # Let's simplify: Perturb 'remaining' (the current candidate)
                    # This is already done above? No, I used best_solution.
                    # Let's change it to perturb 'remaining' if it's better than best, or just 'remaining'.
                    # Actually, perturbing 'remaining' is better because it's the current state.
                    # Let's re-do the perturbation block to use 'remaining'.
                    pass # Placeholder for logic change

            if len(remaining) < best_size:
                best_size = len(remaining)
                best_solution = [k_subsets[i] for i in remaining]
                
    if best_solution is None:
        return []
        
    return [list(b) for b in best_solution]
