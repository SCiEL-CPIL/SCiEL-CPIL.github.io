import time
import random
import itertools
import sys

# Global cache for precomputed data
_PREPARE_CACHE = {}

def _prepare(v, k, t):
    key = (v, k, t)
    if key in _PREPARE_CACHE:
        return _PREPARE_CACHE[key]
    
    # Generate all t-subsets
    t_subs = [tuple(c) for c in itertools.combinations(range(v), t)]
    n_t = len(t_subs)
    index = {sub: i for i, sub in enumerate(t_subs)}
    
    # Generate all k-blocks
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Precompute masks for all k-blocks
    masks = []
    # Also build an index: subset_index -> list of block_indices
    sub_to_blocks = [[] for _ in range(n_t)]
    
    for b_idx, b in enumerate(blocks):
        m = 0
        for sub in itertools.combinations(b, t):
            s_idx = index[sub]
            m |= (1 << s_idx)
            sub_to_blocks[s_idx].append(b_idx)
        masks.append(m)
        
    _PREPARE_CACHE[key] = (blocks, masks, n_t, index, sub_to_blocks, n_blocks)
    return _PREPARE_CACHE[key]

def _greedy_select(masks, n_blocks, n_t, rng, deadline):
    uncovered = (1 << n_t) - 1
    selected = []
    selected_set = set()
    
    # We will use a list of active block indices to scan
    # Scanning all n_blocks is ~50k, which is fast in Python if optimized
    # But we can prune: if a block is already selected, skip it.
    
    while uncovered and time.time() < deadline:
        best_score = -1
        best_block_idx = -1
        
        # Optimization: Randomize the order of scanning to break ties differently in each restart
        # But shuffling 50k items every iteration is expensive.
        # Instead, we can just use a random tie-breaker in the score.
        
        for i in range(n_blocks):
            if i in selected_set:
                continue
            
            m = masks[i]
            # Quick check: if no overlap, skip
            # (m & uncovered) is expensive if we do it for all, but bit_count is fast
            new_cov = (m & uncovered).bit_count()
            if new_cov == 0:
                continue
            
            # Score: primarily new coverage. 
            # Add small random noise to break ties and diversify restarts
            score = new_cov + rng.random() * 1e-6
            
            if score > best_score:
                best_score = score
                best_block_idx = i
        
        if best_block_idx == -1:
            break
            
        selected.append(best_block_idx)
        selected_set.add(best_block_idx)
        
        # Update uncovered
        m = masks[best_block_idx]
        uncovered &= ~m
            
    return selected

def _local_search(selected, masks, n_blocks, n_t, sub_to_blocks, deadline, rng):
    # selected is list of indices
    # We maintain counts for each t-subset
    counts = [0] * n_t
    selected_set = set(selected)
    
    # Initialize counts
    for i in selected:
        m = masks[i]
        mm = m
        while mm:
            lb = mm & (-mm)
            idx = lb.bit_length() - 1
            counts[idx] += 1
            mm &= mm - 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try deletions
        to_remove = -1
        for pos, i in enumerate(selected):
            m = masks[i]
            can_remove = True
            mm = m
            while mm:
                lb = mm & (-mm)
                idx = lb.bit_length() - 1
                if counts[idx] == 1:
                    can_remove = False
                    break
                mm &= mm - 1
            if can_remove:
                to_remove = pos
                break
        
        if to_remove != -1:
            i = selected[to_remove]
            m = masks[i]
            mm = m
            while mm:
                lb = mm & (-mm)
                idx = lb.bit_length() - 1
                counts[idx] -= 1
                mm &= mm - 1
            selected.pop(to_remove)
            selected_set.discard(i)
            improved = True
            continue
        
        # 2. Try swaps
        # For each block i in selected, find its unique subsets
        swap_found = False
        for pos, i in enumerate(selected):
            m_i = masks[i]
            unique_mask = 0
            unique_indices = []
            mm = m_i
            while mm:
                lb = mm & (-mm)
                idx = lb.bit_length() - 1
                if counts[idx] == 1:
                    unique_mask |= lb
                    unique_indices.append(idx)
                mm &= mm - 1
                
            if unique_mask == 0:
                # Should have been removed in step 1, but if not, it's redundant?
                # If unique is 0, all subsets in i are covered by others. So i is redundant.
                # Why wasn't it removed? Step 1 checks `counts[idx] == 1`. 
                # If no idx has count 1, then it is removable.
                # So this case shouldn't happen if step 1 ran correctly.
                # However, if step 1 found a *different* removable block and removed it,
                # the counts changed. But we `continue` after removal.
                # So if we are here, no block was removed in this iteration.
                # Thus, if unique is 0, i is removable.
                # Let's force removal if unique is 0.
                # Actually, step 1 scans ALL blocks. If i had unique 0, step 1 would have found it (or another).
                # If step 1 found another, we would have continued.
                # So if we are in the swap loop, it means NO block was removable in step 1.
                # Therefore, every block has at least one unique subset.
                # So unique_mask should not be 0.
                continue

            # Find a block j not in selected that covers all unique_indices
            # Use sub_to_blocks to find candidates
            # Intersection of lists for each unique index
            # Start with the list for the first unique index
            if not unique_indices:
                continue
            
            # Pick the rarest unique subset to minimize candidate list size
            # (Optional optimization, but lists are small anyway)
            # Let's just intersect the lists
            
            # Get the list for the first unique index
            candidate_list = sub_to_blocks[unique_indices[0]]
            
            # If there are more unique indices, intersect
            if len(unique_indices) > 1:
                # Convert to sets for intersection? Or just filter?
                # Lists can be long. 
                # Let's filter candidate_list by checking other unique indices
                # For each candidate in candidate_list, check if it covers all other unique indices
                # This is O(len(candidate_list) * len(unique_indices))
                
                # Optimization: If any unique index has a very small list, start with that
                # Let's find the min length list
                min_len = len(candidate_list)
                min_idx = 0
                for u_idx in unique_indices[1:]:
                    l_len = len(sub_to_blocks[u_idx])
                    if l_len < min_len:
                        min_len = l_len
                        min_idx = unique_indices.index(u_idx) # This is O(n), but n is small
                        # Actually, unique_indices is small, so this is fine
                        # But let's just pick the one with smallest list directly
                        # We already have candidate_list from index 0.
                        # Let's re-initialize candidate_list to the smallest one.
                        if l_len < len(candidate_list):
                            candidate_list = sub_to_blocks[u_idx]
                            # We need to check against ALL unique indices, including the original first one
                            # So we can't just skip the first one.
                            # Let's just stick to the first one for simplicity, it's usually fine.
                
                # Let's just filter the current candidate_list
                # For each candidate j in candidate_list:
                #   Check if j covers all unique_indices
                
                filtered_candidates = []
                for j in candidate_list:
                    if j in selected_set:
                        continue
                    # Check if j covers all unique indices
                    m_j = masks[j]
                    ok = True
                    for u_idx in unique_indices:
                        if not (m_j & (1 << u_idx)):
                            ok = False
                            break
                    if ok:
                        filtered_candidates.append(j)
                
                if not filtered_candidates:
                    continue
                
                # Pick a candidate
                # Heuristic: Pick the one that covers the most total subsets? 
                # Or just random?
                # Random is good for diversity.
                j = rng.choice(filtered_candidates)
                
            else:
                # Only one unique index
                # Filter candidate_list
                filtered_candidates = [j for j in candidate_list if j not in selected_set]
                if not filtered_candidates:
                    continue
                j = rng.choice(filtered_candidates)
                
            # Perform swap: remove i, add j
            # Remove i
            mm = m_i
            while mm:
                lb = mm & (-mm)
                idx = lb.bit_length() - 1
                counts[idx] -= 1
                mm &= mm - 1
            selected_set.discard(i)
            
            # Add j
            m_j = masks[j]
            mm = m_j
            while mm:
                lb = mm & (-mm)
                idx = lb.bit_length() - 1
                counts[idx] += 1
                mm &= mm - 1
            selected_set.add(j)
            
            selected[pos] = j
            swap_found = True
            improved = True
            break # Restart loop after swap
            
    return selected

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0 # Safety margin
    
    blocks, masks, n_t, index, sub_to_blocks, n_blocks = _prepare(v, k, t)
    
    best_size = float('inf')
    best_selected = []
    
    # Iterated Local Search
    num_restarts = 30
    time_per_restart = (deadline - t0) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        rng = random.Random(42 + r)
        
        # 1. Greedy Construction
        sel = _greedy_select(masks, n_blocks, n_t, rng, deadline - 5.0)
        
        # 2. Local Search
        sel = _local_search(sel, masks, n_blocks, n_t, sub_to_blocks, deadline - 5.0, rng)
        
        if len(sel) < best_size:
            best_size = len(sel)
            best_selected = list(sel)
            
        # 3. Perturbation for ILS (if we have time)
        # We can perturb the best solution found so far, or the current one.
        # Let's perturb the current one to explore neighborhood.
        # But we already saved the best.
        # For the next restart, we can start from the best solution and perturb it?
        # Or just do a fresh greedy?
        # Fresh greedy is simpler and often effective.
        # Let's try: if r > 0, we can start from best_selected and perturb.
        # But _greedy_select starts from scratch.
        # To implement ILS properly, we should modify the initial state for the next iteration.
        # However, the current structure is "Restart Greedy + LS".
        # This is effectively ILS if the random seed changes the greedy path.
        
        # Additional idea: If we have time, do a "repair" step on the best solution
        # to see if we can improve it further with a different random seed in LS?
        # LS is already randomized via rng.
        
    # Final validity check and fallback
    # Ensure all t-subsets are covered
    if best_selected:
        covered = 0
        for i in best_selected:
            covered |= masks[i]
        
        if covered != (1 << n_t) - 1:
            # Fallback: add any missing blocks greedily
            uncovered = ((1 << n_t) - 1) & ~covered
            selected_set = set(best_selected)
            for i in range(n_blocks):
                if i in selected_set:
                    continue
                if masks[i] & uncovered:
                    best_selected.append(i)
                    selected_set.add(i)
                    uncovered &= ~masks[i]
                    if not uncovered:
                        break

    # Return list of blocks (as lists of ints)
    result = [list(blocks[i]) for i in best_selected]
    return result
