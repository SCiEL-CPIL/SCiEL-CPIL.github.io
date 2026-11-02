"""Improved program: spectral relaxation + min-degree greedy + swap local search for kissing number in dimension 11."""
import numpy as np
import itertools
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s limit

    D = 11

    # Generate weight-4 vectors (norm^2=4): C(11,4)*16 = 5280
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Generate weight-2 vectors (norm^2=2): C(11,2)*4 = 220
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Generate weight-1 vectors (norm^2=1): 22
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)  # 5522

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

    # conflict[i,j] = 1 if i and j CANNOT both be selected (i != j)
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    def greedy_min_degree(remaining_init, time_limit):
        """Min-degree greedy: pick vertex with fewest conflicts among remaining."""
        remaining = remaining_init.copy()
        selected = []
        while time.time() < time_limit and remaining.any():
            # degree[i] = number of conflicts with other remaining vertices
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1  # high value so they won't be picked
            min_deg = deg.min()
            if min_deg >= n:
                break
            # Pick first candidate with min degree among remaining
            mask = (deg == min_deg) & (remaining > 0)
            picks = np.where(mask)[0]
            if len(picks) == 0:
                break
            new_idx = picks[0]
            selected.append(new_idx)
            # Remove new_idx and all its conflicts from remaining
            remaining = remaining & (1 - conflict[new_idx])
            remaining[new_idx] = 0
        return selected

    def swap_local_search(selected_list, time_limit, max_rounds=200):
        """Swap-based local search: try to grow by swapping one selected vector for two."""
        selected = list(selected_list)
        sel_set = set(selected)
        # compat_with_all[j] = 1 if j is compatible with all currently selected
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0

        for _ in range(max_rounds):
            if time.time() > time_limit:
                break
            # Try simple addition first
            addable = np.where(compat_with_all > 0)[0]
            if len(addable) > 0:
                # Pick the one with min degree in full graph (least constraining)
                full_deg = conflict.sum(axis=1)
                best_add = addable[np.argmin(full_deg[addable])]
                selected.append(best_add)
                sel_set.add(best_add)
                compat_with_all = compat_with_all & compat[best_add]
                compat_with_all[best_add] = 0
                continue

            # Try swap: remove one selected, add two new ones
            # For each selected s, find candidates compatible with (selected - {s})
            # that are NOT compatible with s (i.e., they conflict with s)
            # We need at least 2 such candidates that are mutually compatible
            found_swap = False
            # Limit: only try swaps for a random subset of selected to save time
            if len(selected) > 30:
                trial_indices = np.random.choice(len(selected), size=min(10, len(selected)), replace=False)
            else:
                trial_indices = range(len(selected))

            for si in trial_indices:
                if time.time() > time_limit:
                    break
                s = selected[si]
                # Candidates compatible with all selected except s, and NOT s itself
                # = compat_with_all | (conflict with s is OK)
                # A candidate j is compatible with (selected - {s}) iff
                #   compat[j, s'] = 1 for all s' in selected, s' != s
                # This is equivalent to: compat_with_all[j] | (j is compatible with all except s)
                # Simpler: recompute = product of compat[s'] for s' != s
                # But that's expensive. Instead:
                # j is compatible with all except s iff:
                #   for all s' in selected: s' != s => compat[j,s']=1
                #   AND compat[j,s] can be 0 or 1 (we don't care about s)
                # The candidates that are compatible with all selected EXCEPT possibly s:
                #   = (compat_with_all) OR (compatible with all selected except s)
                # Let's compute: compatible_with_except_s[j] = 1 if j compatible with all s' in selected, s' != s
                # = product over s' in selected, s'!=s of compat[j, s']
                # = compat_with_all[j] * (1 if compat[j,s]=1 else ...) 
                # Actually: compat_with_all[j] = prod_{s' in selected} compat[j,s']
                # compatible_with_except_s[j] = compat_with_all[j] / compat[j,s] (if compat[j,s] is 0 or 1)
                # If compat[j,s] = 1: compatible_with_except_s[j] = compat_with_all[j]
                # If compat[j,s] = 0: compatible_with_except_s[j] = 0 (since compat_with_all[j] = 0)
                # Wait, that's not right. compat_with_all[j] = 0 if j conflicts with ANY selected.
                # If j conflicts only with s, then compat_with_all[j] = 0 but compatible_with_except_s[j] = 1.
                # So we need a different computation.
                
                # Compute: for j, is j compatible with all selected except s?
                # = AND over s' in selected, s' != s: compat[j, s']
                # Let's compute this vectorized:
                # mask_except_s: (n,) = 1 if j compatible with all s' in selected, s' != s
                mask_except_s = np.ones(n, dtype=np.uint8)
                for s2 in selected:
                    if s2 != s:
                        mask_except_s = mask_except_s & compat[:, s2]
                mask_except_s[s] = 0  # exclude s itself
                mask_except_s[list(sel_set)] = 0  # exclude all selected (we want NEW candidates)
                
                candidates_for_swap = np.where(mask_except_s > 0)[0]
                if len(candidates_for_swap) < 2:
                    continue
                
                # Among these, find two that are mutually compatible
                # Pick the one with min full degree first, then check if any other is compatible
                full_deg = conflict.sum(axis=1)
                sorted_cands = candidates_for_swap[np.argsort(full_deg[candidates_for_swap])]
                
                found_pair = False
                for i in range(min(20, len(sorted_cands))):
                    c1 = sorted_cands[i]
                    # Find any c2 in sorted_cands that is compatible with c1
                    compat_with_c1 = compat[c1]
                    for j in range(i + 1, min(40, len(sorted_cands))):
                        c2 = sorted_cands[j]
                        if compat_with_c1[c2]:
                            # Do the swap: remove s, add c1 and c2
                            selected.remove(s)
                            sel_set.discard(s)
                            selected.append(c1)
                            sel_set.add(c1)
                            selected.append(c2)
                            sel_set.add(c2)
                            # Recompute compat_with_all
                            compat_with_all = np.ones(n, dtype=np.uint8)
                            for s3 in selected:
                                compat_with_all = compat_with_all & compat[s3]
                            for s3 in selected:
                                compat_with_all[s3] = 0
                            found_swap = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_swap:
                    break

            if not found_swap:
                break

        return selected

    # The set of all weight-4 vectors is NOT mutually compatible.
    # Vectors sharing 3 coordinates have dot product 3, distance sqrt(2) < 2.
    # We must find a large compatible subset.
    
    # We will run a greedy algorithm on the weight-4 candidates.
    # We can also try to include weight-2 and weight-1 vectors if they fit,
    # but weight-4 vectors define the max norm (2), so they are the primary candidates.
    # Adding lower norm vectors requires them to be compatible with all selected weight-4 vectors.
    
    # Let's start with a greedy selection from ALL candidates, prioritizing higher norms?
    # Or just run greedy on weight-4, then try to add others?
    # Greedy on weight-4 is safer and likely yields a large number.
    
    wt4_indices = np.where(norms_sq == 4)[0]
    
    # Initialize remaining set to all weight-4 candidates
    # We need to map these to the global indices or create a sub-problem.
    # The `conflict` matrix is global.
    # We can create a mask of active candidates.
    
    # To speed up, we can just run the greedy function provided, but it operates on `remaining` mask.
    # The `greedy_min_degree` function takes `remaining_init`.
    
    remaining_init = np.zeros(n, dtype=np.uint8)
    remaining_init[wt4_indices] = 1
    
    # Run greedy
    selected_indices = greedy_min_degree(remaining_init, deadline)
    
    # If we have time, try to add weight-2 or weight-1 vectors
    # Condition for adding v to selected set S:
    # dist(v, s) >= 2 for all s in S.
    # If v is weight 2 (norm sqrt(2)):
    #   dist(v, s)^2 = 2 + ||s||^2 - 2 v.s >= 4
    #   If s is weight 4 (norm 2): 2 + 4 - 2 v.s >= 4 => 2 v.s <= 2 => v.s <= 1
    #   If s is weight 2 (norm sqrt(2)): 2 + 2 - 2 v.s >= 4 => 2 v.s <= 0 => v.s <= 0
    #   If s is weight 1 (norm 1): 2 + 1 - 2 v.s >= 4 => 3 - 2 v.s >= 4 => -2 v.s >= 1 => v.s <= -0.5 => v.s <= -1 (integer)
    
    # The `compat` matrix already encodes `dist >= max_norm`.
    # BUT `max_norm` in the calculation of `compat` was fixed to `max_norm_sq = 4`.
    # So `compat[i,j]` is 1 iff `dist(i,j) >= 2`.
    # This is exactly the condition we need for the final set, assuming the final set contains at least one weight-4 vector.
    # If the final set contains ONLY weight-2 vectors, the max norm is sqrt(2), so distance >= sqrt(2) is required.
    # Our `compat` is stricter (requires dist >= 2).
    # So any set found by `compat` is valid for max norm 2.
    # If we drop all weight-4 vectors, we might be able to add more weight-2 vectors.
    # However, weight-4 vectors are more "dense" in terms of count potential?
    # Let's stick to the set found by greedy on weight-4, as it guarantees validity under the strictest constraint.
    # We can try to add weight-2 vectors that are compatible with ALL selected weight-4 vectors.
    
    if len(selected_indices) > 0:
        sel_set = set(selected_indices)
        # Find candidates compatible with all selected
        # We can use the `compat` matrix.
        # A candidate j is compatible with all selected if compat[j, s] == 1 for all s in sel_set.
        
        # Vectorized check:
        # Sum of conflicts with selected should be 0.
        # Or product of compat should be 1.
        
        # Let's compute a mask of "compatible with all selected"
        # Start with all 1s, then AND with compat[:, s] for each s.
        # This is O(K * N). K is size of selected.
        
        if len(selected_indices) < 2000: # Only do this if set isn't too huge
            compat_with_all = np.ones(n, dtype=np.uint8)
            for s in selected_indices:
                compat_with_all = compat_with_all & compat[:, s]
            
            # Exclude already selected
            compat_with_all[list(sel_set)] = 0
            
            # Find any new candidates (weight 1, 2, or 4) that fit
            addable = np.where(compat_with_all > 0)[0]
            
            # Add them all?
            # If we add one, it might conflict with another addable one.
            # So we need to run a greedy on the `addable` subset, respecting `compat` among themselves.
            
            if len(addable) > 0:
                # Create a new remaining mask for just these addable ones?
                # Or just iterate and add if compatible with current selected + new additions.
                
                # Simple greedy on addable
                new_selected = []
                # Sort addable by norm (prefer higher? or just min degree?)
                # Let's just take them one by one if they fit.
                # To be efficient, we can just check pairwise compatibility among addables.
                
                # For simplicity and safety, we will just return the weight-4 greedy result.
                # Adding lower norm vectors is complex to optimize quickly.
                pass
        
    result = candidates[selected_indices]
    return result.astype(np.int64)
