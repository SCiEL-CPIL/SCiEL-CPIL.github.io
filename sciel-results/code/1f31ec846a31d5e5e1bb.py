import numpy as np
import itertools
import random
import time
import sys

def construct(q, n, radius):
    total_words = q ** n
    if total_words > 5000:
        # Fallback to greedy for larger problems if needed, though task is fixed
        # But for this task, n=7, q=3 -> 2187
        pass
        
    # 1. Generate words and compute masks
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    masks = [0] * total_words
    # Precompute balls
    for i, w in enumerate(words):
        ball_val = 0
        # Center
        ball_val |= (1 << i)
        # Neighbors
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                j = word_to_idx[new_w]
                ball_val |= (1 << j)
        masks[i] = ball_val
        
    # 2. Preprocessing: Dominance Reduction
    # If mask[i] is a subset of mask[j], i is dominated by j.
    # We want to keep a list of non-dominated sets.
    # O(N^2) check is 2187^2 ~ 4.7M operations. Fast in Python? 
    # Bitwise AND is fast.
    active_sets = list(range(total_words))
    
    # Optimization: Sort by popcount descending? 
    # If A is subset of B, popcount(A) <= popcount(B).
    # We can just check pairs.
    # To speed up, we can skip checks if popcount(A) > popcount(B).
    popcounts = [m.bit_count() for m in masks]
    
    # Create a list of (index, mask, popcount)
    candidates = [(i, masks[i], popcounts[i]) for i in range(total_words)]
    
    # Remove dominated
    # i is dominated by j if masks[i] & masks[j] == masks[i]
    # We want to remove i.
    # Note: If masks[i] == masks[j], we can keep one.
    
    # Efficient way: 
    # Sort by popcount descending. 
    # For each i, check if it is a subset of any j already kept?
    # No, if j is kept and i is subset of j, remove i.
    # If i is kept and j is subset of i, remove j (but j comes later if sorted desc? No, j has smaller popcount, so j comes later).
    # So, iterate through sorted candidates.
    # Maintain a list `kept`.
    # For current candidate c:
    #   Check if c is a subset of any k in `kept`.
    #   If yes, discard c.
    #   If no, add c to `kept`.
    
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    kept_indices = []
    kept_masks = []
    
    for i, m, pc in candidates:
        is_dominated = False
        for km in kept_masks:
            if (m & km) == m:
                is_dominated = True
                break
        if not is_dominated:
            kept_indices.append(i)
            kept_masks.append(m)
            
    # Now we have a reduced set of codewords.
    # Map from original index to new index? 
    # We just work with the kept_indices.
    
    # 3. Forced Sets
    # Build word_to_kept_sets
    # For each word, which kept sets cover it?
    word_to_sets = [[] for _ in range(total_words)]
    for idx, orig_idx in enumerate(kept_indices):
        m = kept_masks[idx]
        # Iterate bits in m
        temp = m
        while temp:
            lsb = temp & -temp
            w_idx = lsb.bit_length() - 1
            word_to_sets[w_idx].append(idx)
            temp ^= lsb
            
    forced_code = []
    uncovered = (1 << total_words) - 1
    
    # Iteratively find forced sets
    changed = True
    while changed:
        changed = False
        for w_idx in range(total_words):
            if (uncovered >> w_idx) & 1:
                if len(word_to_sets[w_idx]) == 0:
                    # No set covers this word? Should not happen if radius is valid.
                    # But if we removed sets, maybe?
                    # Dominance reduction preserves coverage union.
                    # So this shouldn't happen.
                    return [list(w) for w in words] # Fallback
                elif len(word_to_sets[w_idx]) == 1:
                    s_idx = word_to_sets[w_idx][0]
                    if s_idx not in forced_code:
                        forced_code.append(s_idx)
                        uncovered &= ~kept_masks[s_idx]
                        changed = True
                        # Update word_to_sets? 
                        # We don't need to update word_to_sets for forced sets if we just track `uncovered`.
                        # But for the next iteration, we need to know which sets are "available".
                        # Actually, if a set is forced, we commit to it.
                        # We can remove it from consideration for other words?
                        # No, we just remove its coverage from `uncovered`.
                        # The `word_to_sets` lists still contain it.
                        # We need to filter out forced sets from `word_to_sets` for the branching phase.
                        # Or we can just check if a set is in `forced_code` during branching.
                        
    # Remove forced sets from consideration
    forced_set_ids = set(forced_code)
    
    # Filter word_to_sets to only contain non-forced sets
    for w_idx in range(total_words):
        word_to_sets[w_idx] = [s for s in word_to_sets[w_idx] if s not in forced_set_ids]
        
    # If uncovered is 0, we are done.
    if uncovered == 0:
        code_indices = forced_code
        return [list(words[kept_indices[i]]) for i in code_indices]

    # 4. Greedy Initial Solution
    def greedy_solve(uncovered_mask, available_sets):
        # available_sets is a list of indices into kept_masks
        # Returns list of indices
        code = []
        unc = uncovered_mask
        # Make a copy of available sets to modify? 
        # We just pick best each time.
        remaining_sets = list(available_sets)
        
        while unc:
            best_idx = -1
            best_cover = 0
            for s_idx in remaining_sets:
                cover = (kept_masks[s_idx] & unc).bit_count()
                if cover > best_cover:
                    best_cover = cover
                    best_idx = s_idx
            if best_idx == -1:
                break
            code.append(best_idx)
            unc &= ~kept_masks[best_idx]
            # Remove used set? 
            # In Set Cover, sets can be reused? No, distinct codewords.
            # But if we pick a set, we don't pick it again.
            remaining_sets.remove(best_idx)
            
        return code

    # Available sets are those not in forced_code
    available_sets = [i for i in range(len(kept_indices)) if i not in forced_set_ids]
    
    init_code = greedy_solve(uncovered, available_sets)
    best_size = len(forced_code) + len(init_code)
    best_code_indices = forced_code + init_code
    
    # 5. Branch and Bound
    start_time = time.time()
    max_time = 110.0
    
    # Precompute max set size for LB
    max_set_size = max([ (m & uncovered).bit_count() for m in kept_masks if kept_masks.index(m) in available_sets ] or [1])
    # Actually, max size might change as uncovered shrinks.
    # We can use a static upper bound on coverage: max(popcounts)
    global_max_pop = max(popcounts)
    
    def lb_calc(unc_mask, curr_size):
        # Simple LB: ceil(popcount(unc) / global_max_pop)
        # Better LB: 
        # Find a word w in unc with min coverage.
        # If min coverage is c, we need at least 1 set.
        # This is weak.
        # Use: ceil(popcount(unc) / max_possible_coverage)
        return (unc_mask.bit_count() + global_max_pop - 1) // global_max_pop

    def bnb(unc_mask, curr_code, curr_size, depth):
        nonlocal best_size, best_code_indices
        
        if time.time() - start_time > max_time:
            return
            
        if unc_mask == 0:
            if curr_size < best_size:
                best_size = curr_size
                best_code_indices = curr_code[:]
            return
            
        # LB Pruning
        lb = lb_calc(unc_mask, curr_size)
        if curr_size + lb >= best_size:
            return
            
        # Select variable (uncovered word)
        # Find word w in unc_mask that has the fewest covering sets in `available_sets`
        # Note: `available_sets` is static, but we should consider only sets that cover w.
        # We need to find w such that len({s in available_sets | w in mask[s]}) is minimized.
        
        min_count = float('inf')
        target_word = -1
        target_sets = []
        
        # Iterate through uncovered words
        temp = unc_mask
        while temp:
            lsb = temp & -temp
            w_idx = lsb.bit_length() - 1
            # Count how many available sets cover w_idx
            # We can precompute for each w_idx, the list of ALL sets covering it.
            # Then filter by available?
            # Or just iterate available_sets and check?
            # available_sets size is ~1800. unc words ~2000. 3.6M checks. Too slow per node.
            
            # Better: Precompute `word_to_sets` for the *current* available sets?
            # That's expensive.
            
            # Alternative: Just pick the first uncovered word?
            # Or pick the one with smallest `word_to_sets[w_idx]` (from the original list, filtered by forced).
            # Since forced sets are removed, `word_to_sets[w_idx]` contains only non-forced sets.
            # But some non-forced sets might have been "excluded" by previous branches?
            # In this B&B formulation, we don't exclude sets globally, we just branch on inclusion.
            # Wait, standard B&B for Set Cover:
            # We pick a word w. We MUST pick one of the sets covering w.
            # We branch on each set S covering w.
            # In the branch where we pick S, we add S to code, and remove S's coverage from unc.
            # We do NOT remove S from the pool for other branches (siblings).
            # So `available_sets` is constant for the depth.
            
            # So we can just use `word_to_sets[w_idx]` directly.
            count = len(word_to_sets[w_idx])
            if count < min_count:
                min_count = count
                target_word = w_idx
                target_sets = word_to_sets[w_idx]
            temp ^= lsb
            
        if target_word == -1:
            return
            
        if min_count == 0:
            return # Dead end
            
        # Order target_sets by coverage of unc_mask (descending)
        # This helps find good solutions early
        def cover_score(s_idx):
            return (kept_masks[s_idx] & unc_mask).bit_count()
            
        target_sets.sort(key=cover_score, reverse=True)
        
        for s_idx in target_sets:
            # Branch: Include s_idx
            new_unc = unc_mask & ~kept_masks[s_idx]
            new_code = curr_code + [s_idx]
            bnb(new_unc, new_code, curr_size + 1, depth + 1)
            
            # Optional: Pruning if we found a solution with size curr_size + 1 + lb(new_unc) >= best_size?
            # The recursive call handles this.

    # Run B&B
    bnb(uncovered, forced_code[:], len(forced_code), 0)
    
    # Construct final list
    final_indices = best_code_indices
    # Map back to original word indices
    orig_indices = [kept_indices[i] for i in final_indices]
    # Ensure distinct? The problem says codewords must be distinct.
    # Our sets are distinct masks (after dominance reduction, masks are unique? 
    # If two sets had identical masks, one would dominate the other and be removed.
    # So masks are unique.
    # Are the words distinct? Yes, indices are unique.
    
    code_words = [words[i] for i in orig_indices]
    
    return [list(c) for c in code_words]
