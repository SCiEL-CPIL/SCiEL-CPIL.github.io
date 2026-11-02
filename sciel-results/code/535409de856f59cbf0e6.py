"""Improved program: Multi-start greedy with local search for kissing number in dimension 11."""
import numpy as np
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    D = 11

    # Generate weight-4 vectors (norm^2=4): C(11,4)*16 = 5280
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    
    # We focus on weight-4 vectors to ensure max norm is 2.
    # Adding weight-2 or 1 vectors is possible but they are "easier" to fit?
    # Actually, weight-4 vectors are the "bottleneck" because they are far apart.
    # If we find a large set of weight-4 vectors, we can try to add weight-2/1 vectors later.
    # But for now, let's maximize the weight-4 set.
    
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # => v_i·v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    def greedy_randomized(remaining_init, time_limit, seed):
        """Randomized min-degree greedy."""
        rng = np.random.default_rng(seed)
        remaining = remaining_init.copy()
        selected = []
        while time.time() < time_limit and remaining.any():
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            min_deg = deg.min()
            if min_deg >= n:
                break
            mask = (deg == min_deg) & (remaining > 0)
            picks = np.where(mask)[0]
            if len(picks) == 0:
                break
            new_idx = rng.choice(picks)
            selected.append(new_idx)
            remaining = remaining & (1 - conflict[new_idx])
            remaining[new_idx] = 0
        return selected

    def local_search(selected_list, time_limit):
        """Local search: 1-swap and 2-swap."""
        selected = list(selected_list)
        sel_set = set(selected)
        
        # Precompute full degree for tie-breaking or heuristic
        full_deg = conflict.sum(axis=1)
        
        # compat_with_all[j] = 1 if j is compatible with all currently selected
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try to add a vertex (if any)
            addable = np.where(compat_with_all > 0)[0]
            if len(addable) > 0:
                # Pick the one with min full degree (least constraining)
                best_add = addable[np.argmin(full_deg[addable])]
                selected.append(best_add)
                sel_set.add(best_add)
                compat_with_all = compat_with_all & compat[best_add]
                compat_with_all[best_add] = 0
                improved = True
                continue

            # 2. Try 1-swap: Remove one, add one
            # For each selected s, check if there is a candidate j compatible with (selected - {s})
            # and NOT compatible with s (or just not selected).
            # Actually, if j is compatible with all selected, it would have been added in step 1.
            # So we look for j compatible with all selected EXCEPT s.
            
            # To speed up, we can precompute for each s, the candidates compatible with all others.
            # This is expensive to recompute every time.
            # Alternative: Iterate through selected s.
            
            # Optimization: Only try swaps for a subset of s if selected is large
            trial_indices = range(len(selected))
            if len(selected) > 100:
                # Random sample
                trial_indices = np.random.choice(len(selected), size=min(20, len(selected)), replace=False)
            
            for si in trial_indices:
                if time.time() > time_limit:
                    break
                s = selected[si]
                
                # Compute mask of candidates compatible with all selected except s
                # mask_except_s[j] = 1 if j compatible with all s' in selected, s' != s
                mask_except_s = np.ones(n, dtype=np.uint8)
                for s2 in selected:
                    if s2 != s:
                        mask_except_s = mask_except_s & compat[:, s2]
                
                # We want j such that:
                # 1. j is in mask_except_s
                # 2. j is NOT in selected
                # 3. j is NOT compatible with s? No, if j is compatible with s, it's in compat_with_all, which is empty.
                # So j must be incompatible with s.
                # Actually, if j is compatible with s, then j is compatible with ALL selected.
                # Since we found no addable in step 1, there are NO j compatible with ALL selected.
                # So any j in mask_except_s is automatically incompatible with s (unless j==s, but we exclude selected).
                
                mask_except_s[list(sel_set)] = 0 # Exclude selected
                
                candidates_for_swap = np.where(mask_except_s > 0)[0]
                
                if len(candidates_for_swap) > 0:
                    # Pick the best candidate (min full degree)
                    best_swap = candidates_for_swap[np.argmin(full_deg[candidates_for_swap])]
                    
                    # Perform swap
                    selected.remove(s)
                    sel_set.discard(s)
                    selected.append(best_swap)
                    sel_set.add(best_swap)
                    
                    # Update compat_with_all
                    # We removed s, added best_swap
                    # We need to recompute? Or update incrementally?
                    # Removing s: compat_with_all becomes "looser" (more 1s)
                    # Adding best_swap: compat_with_all becomes "tighter" (fewer 1s)
                    # It's safer to recompute if the set is small, or update carefully.
                    # Recomputing is O(K*N). K~1000, N~5000. 5e6 ops. Fast enough.
                    compat_with_all = np.ones(n, dtype=np.uint8)
                    for s3 in selected:
                        compat_with_all = compat_with_all & compat[s3]
                    for s3 in selected:
                        compat_with_all[s3] = 0
                        
                    improved = True
                    break # Restart loop with new state

            # 3. Try 2-swap: Remove one, add two
            if not improved and time.time() < time_limit:
                # Similar to 1-swap, but we need two compatible candidates from mask_except_s
                trial_indices = range(len(selected))
                if len(selected) > 100:
                    trial_indices = np.random.choice(len(selected), size=min(10, len(selected)), replace=False)
                
                for si in trial_indices:
                    if time.time() > time_limit:
                        break
                    s = selected[si]
                    
                    mask_except_s = np.ones(n, dtype=np.uint8)
                    for s2 in selected:
                        if s2 != s:
                            mask_except_s = mask_except_s & compat[:, s2]
                    
                    mask_except_s[list(sel_set)] = 0
                    candidates_for_swap = np.where(mask_except_s > 0)[0]
                    
                    if len(candidates_for_swap) >= 2:
                        # Find a pair (c1, c2) in candidates_for_swap that are compatible
                        # Sort by full degree
                        sorted_cands = candidates_for_swap[np.argsort(full_deg[candidates_for_swap])]
                        
                        found_pair = False
                        # Check first few candidates
                        for i in range(min(10, len(sorted_cands))):
                            c1 = sorted_cands[i]
                            # Find c2 compatible with c1
                            # c2 must be in sorted_cands, index > i
                            compat_with_c1 = compat[c1]
                            # We want c2 in sorted_cands such that compat[c1, c2] == 1
                            # We can just check the next few
                            for j in range(i + 1, min(i + 20, len(sorted_cands))):
                                c2 = sorted_cands[j]
                                if compat_with_c1[c2]:
                                    # Do the swap
                                    selected.remove(s)
                                    sel_set.discard(s)
                                    selected.append(c1)
                                    sel_set.add(c1)
                                    selected.append(c2)
                                    sel_set.add(c2)
                                    
                                    compat_with_all = np.ones(n, dtype=np.uint8)
                                    for s3 in selected:
                                        compat_with_all = compat_with_all & compat[s3]
                                    for s3 in selected:
                                        compat_with_all[s3] = 0
                                        
                                    improved = True
                                    found_pair = True
                                    break
                            if found_pair:
                                break
                    if improved:
                        break

        return selected

    best_selected = []
    best_size = 0
    
    # Multi-start
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        remaining_init = np.ones(n, dtype=np.uint8)
        
        # Greedy
        sel = greedy_randomized(remaining_init, deadline - 5.0, seed=r*1000)
        
        # Local Search
        sel = local_search(sel, deadline - 2.0)
        
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel

    if not best_selected:
        # Fallback: return empty or a single vector if something went wrong
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    result = candidates[best_selected]
    return result.astype(np.int64)
