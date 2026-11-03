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
    
    # Fill block_t_indices
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]
            
    # Precompute inverse: for each t-subset, which blocks cover it?
    # This is a sparse structure. List of lists.
    # blocks_covering_t[s] = list of block indices
    blocks_covering_t = [[] for _ in range(n_t)]
    for i in range(n_k):
        for s in block_t_indices[i]:
            blocks_covering_t[s].append(i)
            
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Reset state
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = set()
        
        # Greedy loop
        while True:
            if time.time() > deadline - 10.0: # Leave time for local search
                break
            
            # Calculate gains
            gains = uncovered[block_t_indices].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_idx = rng.choice(candidates)
            
            selected_blocks.add(best_idx)
            uncovered[block_t_indices[best_idx]] = False
            
        # If we didn't finish greedy, skip local search for this restart
        if time.time() > deadline - 5.0:
            continue
            
        # Local Search: Removal and Swaps
        if selected_blocks:
            # Build coverage count
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bi in selected_blocks:
                coverage_count[block_t_indices[bi]] += 1
                
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                
                # 1. Try to remove redundant blocks
                to_remove = []
                for bi in list(selected_blocks):
                    if np.all(coverage_count[block_t_indices[bi]] > 1):
                        to_remove.append(bi)
                
                for bi in to_remove:
                    coverage_count[block_t_indices[bi]] -= 1
                    selected_blocks.remove(bi)
                    improved = True
                    if len(selected_blocks) < best_size:
                        best_size = len(selected_blocks)
                        best_solution = [k_subsets[i] for i in selected_blocks]
                
                if not selected_blocks:
                    break

                # 2. Try swaps
                # For each block in solution, check if it can be swapped
                # to allow removal of itself or another block.
                # Heuristic: Find a block B_i that is "critical" (has unique subsets).
                # Find a block B_j not in solution that covers those unique subsets.
                # If swapping B_i with B_j allows removal of B_i (or makes it redundant),
                # we might reduce size if B_j allows removal of another block, or if
                # the swap itself is neutral but enables future removals.
                # Simpler: If swap makes B_i removable, size stays same, but structure changes.
                # We accept neutral swaps only if they lead to immediate removal of *another* block?
                # No, usually we accept neutral swaps if they break symmetry, but that's risky for time.
                # Let's stick to: Swap B_i with B_j. Update coverage. If any block becomes removable, remove it.
                # If size decreased, accept. If size same, we might accept with low probability to escape local minima?
                # Given time limits, let's only accept if size decreases.
                
                # Optimization: Only check swaps for blocks that are NOT already removable (handled above).
                # We need to find B_i such that U_i (unique subsets) is small.
                
                # To speed up, we can pick a random block from selected_blocks to try swapping.
                # Or iterate. Iterating all is O(N_blocks * Cost_Check).
                # Cost_Check: Find U_i. If empty, skip (already handled).
                # Find candidates B_j covering U_i.
                
                # Let's try a few random swaps per iteration to save time
                for _ in range(10):
                    if time.time() > deadline - 5.0:
                        break
                    if not selected_blocks:
                        break
                    
                    # Pick a random block to try swapping
                    bi = rng.choice(list(selected_blocks))
                    
                    # Find unique subsets for bi
                    b_i_subsets = block_t_indices[bi]
                    unique_mask = coverage_count[b_i_subsets] == 1
                    unique_subsets = b_i_subsets[unique_mask]
                    
                    if len(unique_subsets) == 0:
                        # Should have been removed, but maybe coverage changed?
                        # If so, remove it
                        coverage_count[b_i_subsets] -= 1
                        selected_blocks.remove(bi)
                        improved = True
                        if len(selected_blocks) < best_size:
                            best_size = len(selected_blocks)
                            best_solution = [k_subsets[i] for i in selected_blocks]
                        continue
                        
                    if len(unique_subsets) > 5: # Too many unique subsets, hard to cover with 1 block
                        continue

                    # Find a block B_j not in selected_blocks that covers all unique_subsets
                    # Take the smallest list of blocks covering any of the unique subsets
                    # Sort unique_subsets by len(blocks_covering_t[s])
                    unique_subsets_sorted = sorted(unique_subsets, key=lambda s: len(blocks_covering_t[s]))
                    
                    found_swap = False
                    # Check candidates from the most constrained subset
                    for candidate_j in blocks_covering_t[unique_subsets_sorted[0]]:
                        if candidate_j in selected_blocks:
                            continue
                        
                        # Check if candidate_j covers all unique_subsets
                        # unique_subsets is an array of indices
                        # We need to check if coverage_count[unique_subsets] is 1 for all?
                        # No, we just need candidate_j to *contain* them.
                        # candidate_j contains unique_subsets if all s in unique_subsets are in block_t_indices[candidate_j]
                        # This is a set intersection check.
                        # Since unique_subsets is small, we can check:
                        # For each s in unique_subsets, is s in block_t_indices[candidate_j]?
                        
                        # Optimization: Precompute a set for each block? No, memory.
                        # Just check.
                        valid = True
                        for s in unique_subsets:
                            # Check if s is in block_t_indices[candidate_j]
                            # This is O(1) if we use a set, but we have an array.
                            # We can use np.isin? Or just a loop since len is small.
                            if s not in block_t_indices[candidate_j]:
                                valid = False
                                break
                                
                        if valid:
                            # Perform swap
                            # Remove bi
                            coverage_count[block_t_indices[bi]] -= 1
                            selected_blocks.remove(bi)
                            
                            # Add candidate_j
                            coverage_count[block_t_indices[candidate_j]] += 1
                            selected_blocks.add(candidate_j)
                            
                            # Check if this swap allows removal of any block
                            # We don't immediately know if it helps, but we accept the swap
                            # and let the next iteration's removal phase handle it.
                            # To ensure progress, we might want to check if size can be reduced.
                            # But even if not immediate, it might break a local minimum.
                            # However, to be safe and fast, let's only accept if it leads to
                            # a removal in the *next* step?
                            # Actually, just accepting the swap and letting the loop handle
                            # removal is standard.
                            
                            improved = True
                            found_swap = True
                            break
                            
                    if found_swap:
                        # We made a change, break the random swap loop to re-evaluate removals
                        break
                            
        # End Local Search
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = [k_subsets[i] for i in selected_blocks]

    if best_solution is None:
        return []
        
    return [list(b) for b in best_solution]
