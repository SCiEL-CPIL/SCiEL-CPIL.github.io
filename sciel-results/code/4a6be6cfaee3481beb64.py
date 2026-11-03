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
            
    # 4. Calculate Weights for Weighted Greedy
    # Frequency of each t-subset across all k-blocks
    t_counts = np.bincount(block_t_indices.ravel(), minlength=n_t)
    t_counts[t_counts == 0] = 1
    t_weights = 1.0 / t_counts.astype(np.float64)
    
    # 5. Build sub_to_blocks for local search
    # sub_to_blocks[j] is a list of block indices that cover t-subset j
    sub_to_blocks = [[] for _ in range(n_t)]
    # We can build this efficiently from block_t_indices
    for b_idx in range(n_k):
        for s_idx in block_t_indices[b_idx]:
            sub_to_blocks[s_idx].append(b_idx)
            
    # Convert to numpy arrays for faster intersection if needed? 
    # Lists are fine for now, we will intersect them.
    
    # 6. Greedy Set Cover Initialization
    uncovered = np.ones(n_t, dtype=bool)
    selected_blocks = []
    rng = random.Random(42)
    
    while True:
        if time.time() > deadline - 10.0:
            break
        if not uncovered.any():
            break
            
        # Calculate weighted gains
        gains = (uncovered[block_t_indices] * t_weights[block_t_indices]).sum(axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        candidates = np.where(gains == max_gain)[0]
        best_idx = rng.choice(candidates)
        selected_blocks.append(best_idx)
        uncovered[block_t_indices[best_idx]] = False
        
    if not selected_blocks:
        return []
        
    best_solution = [k_subsets[i] for i in selected_blocks]
    best_size = len(selected_blocks)
    
    # 7. Local Search: Swap-based Optimization
    # We maintain a set of selected block indices and their coverage counts
    # Note: selected_blocks is a list of indices into k_subsets
    
    def local_search_improve(selected_indices, deadline):
        # selected_indices is a list of block indices
        if not selected_indices:
            return [], 0
            
        # Build coverage counts
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for bi in selected_indices:
            coverage_count[block_t_indices[bi]] += 1
            
        current_size = len(selected_indices)
        improved = True
        no_improve_streak = 0
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Phase 1: Remove redundant blocks
            # We scan the list. If a block is redundant, we remove it.
            # We need to be careful with indices if we modify the list.
            # Better: iterate through a copy or use a flag.
            i = 0
            while i < len(selected_indices):
                if time.time() > deadline - 2.0:
                    return selected_indices, len(selected_indices)
                    
                bi = selected_indices[i]
                ts = block_t_indices[bi]
                # Check if all covered subsets have count > 1
                if np.all(coverage_count[ts] > 1):
                    # Remove block
                    coverage_count[ts] -= 1
                    selected_indices.pop(i)
                    current_size -= 1
                    improved = True
                    # Do not increment i, check the new element at i
                else:
                    i += 1
                    
            # Phase 2: Try swaps
            # To avoid O(N^2) complexity, we limit the number of swap attempts or sample blocks.
            # We will iterate through blocks, but if it gets too slow, we break.
            
            # Shuffle indices to randomize order? No, we want to be systematic but maybe random restarts help.
            # Let's try to find a block that can be swapped.
            
            # For efficiency, we only attempt swaps if we have time.
            if time.time() > deadline - 5.0:
                break
                
            # Iterate through blocks
            for i in range(len(selected_indices)):
                if time.time() > deadline - 2.0:
                    break
                    
                bi = selected_indices[i]
                ts_b = block_t_indices[bi]
                
                # Find critical subsets: those with count == 1
                critical_mask = (coverage_count[ts_b] == 1)
                critical_subsets = ts_b[critical_mask]
                
                if len(critical_subsets) == 0:
                    # This block is redundant, should have been removed in Phase 1.
                    # But counts might have changed? No, Phase 1 ran to completion.
                    # If it happens, remove it.
                    coverage_count[ts_b] -= 1
                    selected_indices.pop(i)
                    current_size -= 1
                    improved = True
                    break # Restart loop as indices changed
                    
                # If too many critical subsets, finding a replacement is hard.
                # Heuristic: Skip if critical sets > 3?
                # Let's try up to 4.
                if len(critical_subsets) > 4:
                    continue
                    
                # Find candidate blocks covering ALL critical subsets
                # Intersection of sub_to_blocks[s] for s in critical_subsets
                # Sort critical_subsets by the size of their candidate list to optimize intersection
                cands_lists = [sub_to_blocks[s] for s in critical_subsets]
                cands_lists.sort(key=len)
                
                if not cands_lists:
                    continue
                
                # Start with the smallest list
                candidates = set(cands_lists[0])
                
                for lst in cands_lists[1:]:
                    if not candidates:
                        break
                    # Intersect with next list
                    # Optimization: if current candidates is small, iterate it.
                    # else iterate lst.
                    if len(candidates) < len(lst):
                        candidates = candidates.intersection(lst)
                    else:
                        candidates = set(lst).intersection(candidates)
                        
                if not candidates:
                    continue
                
                # Remove the current block bi from candidates (we don't swap with itself)
                candidates.discard(bi)
                
                if not candidates:
                    continue
                
                # Pick a candidate
                # Heuristic: Pick the one that covers the most subsets with count > 1 (to create redundancy)
                # Or just pick one randomly from a small subset to save time.
                # Let's evaluate a few top candidates.
                cand_list = list(candidates)
                
                # If too many candidates, sample
                if len(cand_list) > 50:
                    cand_list = rng.sample(cand_list, 50)
                
                best_cand = None
                best_score = -1
                
                for bc in cand_list:
                    # Score: How many subsets in bc have count > 1?
                    # This indicates potential for redundancy.
                    ts_c = block_t_indices[bc]
                    # We want to maximize the number of "safe" subsets (count > 1)
                    # because if we swap, these subsets will have count >= 2,
                    # while the critical subsets of bi will have count 1 (from bc).
                    # Wait, if we swap bi -> bc:
                    # - Subsets in bi \ bc: count decreases by 1.
                    # - Subsets in bc \ bi: count increases by 1.
                    # - Subsets in bi \cap bc: count unchanged.
                    
                    # The critical subsets of bi MUST be in bc.
                    # So for critical subsets, count goes from 1 (from bi) to 1 (from bc). Net change 0.
                    # For other subsets in bi: count decreases.
                    # For other subsets in bc: count increases.
                    
                    # We want to avoid making any subset count 0.
                    # Since critical subsets are covered by bc, they are safe.
                    # What about non-critical subsets in bi? They had count >= 2.
                    # If count was 2, it becomes 1. Safe.
                    # If count was > 2, it becomes > 1. Safe.
                    # So the swap is ALWAYS valid if bc covers all critical subsets of bi.
                    
                    # So, what is the benefit?
                    # Benefit comes if bc covers subsets that are "over-covered" (count > 2),
                    # allowing us to remove other blocks later?
                    # Or if bc allows bi to be removed? No, we are swapping.
                    
                    # Actually, the main benefit of a swap is often to change the "structure"
                    # to allow for redundancy removal of OTHER blocks.
                    # A good heuristic: Pick bc that shares the most subsets with bi (minimize change)
                    # OR pick bc that has the highest "redundancy potential".
                    
                    # Let's try: Pick bc that minimizes the number of subsets whose count changes.
                    # This is len(bc \ bi) + len(bi \ bc).
                    # Since we want to keep the solution stable, maybe minimize disruption.
                    # However, to escape local minima, we might want to maximize disruption in a "good" way.
                    
                    # Let's just pick the first valid candidate or a random one from the list.
                    # To be slightly smarter, let's pick the one with the highest overlap with bi.
                    overlap = np.sum(coverage_count[block_t_indices[bc]] > 0) # This is just size of bc
                    # Actually, let's just pick random from cand_list.
                    # Randomness helps escape local minima.
                    best_cand = rng.choice(cand_list)
                    break # Just try one swap per block to keep it fast?
                    
                if best_cand is not None:
                    # Perform swap
                    # Remove bi
                    coverage_count[ts_b] -= 1
                    selected_indices[i] = best_cand
                    
                    # Add best_cand
                    ts_bc = block_t_indices[best_cand]
                    coverage_count[ts_bc] += 1
                    
                    improved = True
                    # Break inner loop to re-run Phase 1 (Redundancy Removal)
                    break
                    
        return selected_indices, len(selected_indices)

    # Run local search
    current_indices = list(selected_blocks)
    current_indices, current_size = local_search_improve(current_indices, deadline)
    
    if current_size < best_size:
        best_size = current_size
        best_solution = [k_subsets[i] for i in current_indices]
        
    # If we have time, try a second pass with a different random seed or restart
    if time.time() < deadline - 15.0:
        # Try to improve further with more aggressive swapping?
        # Or just run the local search again on the current best.
        current_indices, current_size = local_search_improve(current_indices, deadline)
        if current_size < best_size:
            best_size = current_size
            best_solution = [k_subsets[i] for i in current_indices]

    # Iterated Local Search (ILS)
    # Perturb the current best solution and repair, then polish.
    # This helps escape local minima that trap simple local search.
    
    if time.time() < deadline - 30.0:
        perturbation_strength = 0.05 # 5% of blocks
        max_iterations = 10 # Limit number of ILS iterations
        
        for _ in range(max_iterations):
            if time.time() > deadline - 10.0:
                break
                
            # 1. Perturbation: Randomly delete some blocks
            # We need to be careful not to delete too many, or repair will be expensive.
            # Let's delete a small percentage.
            n_perturb = max(1, int(len(current_indices) * perturbation_strength))
            if n_perturb >= len(current_indices):
                n_perturb = max(1, len(current_indices) // 2)
                
            # Randomly select blocks to remove
            remove_indices = set(rng.sample(range(len(current_indices)), n_perturb))
            
            # Create a new list of indices without the removed ones
            perturbed_indices = [idx for i, idx in enumerate(current_indices) if i not in remove_indices]
            
            # 2. Repair: Greedy set cover to fill the holes
            # Calculate coverage of perturbed set
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bi in perturbed_indices:
                coverage_count[block_t_indices[bi]] += 1
                
            uncovered = coverage_count == 0
            
            # Greedy repair
            while uncovered.any() and time.time() < deadline - 5.0:
                # Calculate gains for all blocks NOT in perturbed_indices
                # To speed up, we only consider blocks that cover at least one uncovered subset?
                # Or just calculate gains for all and mask out selected ones.
                # Since n_k is large, calculating gains for all is O(n_k * C(k,t)).
                # C(19,8) is ~75582. C(8,5) is 56. 75582 * 56 ~ 4M ops. Fast in numpy.
                
                # Mask out already selected blocks
                # We don't have a direct mask, so we calculate gains and then set selected ones to -inf
                # Or simpler: just calculate gains, find max, if max is selected, set to 0 and repeat?
                # No, we must pick a NEW block.
                
                # Efficient way:
                # 1. Calculate gains for all blocks.
                # 2. Identify candidates (gains > 0).
                # 3. Filter candidates to those not in perturbed_indices.
                # 4. Pick best.
                
                gains = (uncovered[block_t_indices] * t_weights[block_t_indices]).sum(axis=1)
                
                # Filter out already selected blocks
                # Create a set of selected indices for fast lookup
                selected_set = set(perturbed_indices)
                
                # Find valid candidates
                valid_mask = (gains > 0)
                # Iterate through valid candidates? No, vectorize.
                # We can create a boolean mask for "not selected"
                # But selected_indices is a list of ints.
                
                # Let's just iterate through the top candidates from gains.
                # Sort indices by gain descending?
                # Or just find the max gain among non-selected.
                
                # Trick: Set gains of selected blocks to -1
                # But we need to do it efficiently.
                
                # Alternative:
                # 1. Get all indices with gain > 0
                # 2. Check which ones are not in selected_set
                
                candidate_indices = np.where(valid_mask)[0]
                
                if len(candidate_indices) == 0:
                    break # No improvement possible? Should not happen if uncovered.any()
                    
                # Filter candidates
                # This might be slow if candidate_indices is large.
                # But usually, gains > 0 implies the block covers some uncovered sets.
                
                # Heuristic: Sample from top candidates?
                # Let's just check the top 100 candidates by gain.
                
                # Sort candidate_indices by gains descending
                # top_k = min(100, len(candidate_indices))
                # order = np.argsort(gains[candidate_indices])[::-1][:top_k]
                # top_candidates = candidate_indices[order]
                
                # Simpler: Just pick the global max, if it's selected, pick the next best?
                # This is inefficient.
                
                # Let's use the set approach.
                # Since we are in a loop, we want this to be fast.
                
                # Optimization:
                # We can maintain a list of "available" blocks?
                # Or just check the max gain block. If it's selected, set its gain to 0 and re-max?
                # This is O(N) in the worst case if many top blocks are selected.
                
                # Let's try:
                best_gain = -1
                best_idx = -1
                
                # Iterate through top candidates only to save time?
                # Let's take the top 20 candidates by gain.
                if len(candidate_indices) > 20:
                    # Get indices of top 20 gains
                    top_20_idx = np.argsort(gains[candidate_indices])[::-1][:20]
                    top_candidates = candidate_indices[top_20_idx]
                else:
                    top_candidates = candidate_indices
                    
                for idx in top_candidates:
                    if idx not in selected_set:
                        if gains[idx] > best_gain:
                            best_gain = gains[idx]
                            best_idx = idx
                            
                if best_idx == -1:
                    # If top 20 were all selected, we might need to look deeper.
                    # Fallback: iterate through all candidate_indices
                    for idx in candidate_indices:
                        if idx not in selected_set:
                            if gains[idx] > best_gain:
                                best_gain = gains[idx]
                                best_idx = idx
                    if best_idx == -1:
                        break # No valid block found? Error state.
                        
                # Add best_idx to perturbed_indices
                perturbed_indices.append(best_idx)
                selected_set.add(best_idx)
                coverage_count[block_t_indices[best_idx]] += 1
                uncovered[block_t_indices[best_idx]] = False
                
            # 3. Polish: Run local search on the repaired solution
            if time.time() < deadline - 5.0:
                polished_indices, polished_size = local_search_improve(perturbed_indices, deadline - 5.0)
                
                # Accept if better
                if polished_size < best_size:
                    best_size = polished_size
                    best_solution = [k_subsets[i] for i in polished_indices]
                    current_indices = polished_indices # Update current for next perturbation

    if best_solution is None:
        return []
        
    return [list(b) for b in best_solution]
