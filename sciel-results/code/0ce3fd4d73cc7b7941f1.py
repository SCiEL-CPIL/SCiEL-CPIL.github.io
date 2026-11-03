import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    points = list(range(v))
    
    # Map each t-subset to an index
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute masks for all possible blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Precompute block_t_indices as numpy array: shape (n_blocks, C(k,t))
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)
    
    block_t_indices = np.empty((n_blocks, n_t_in_k), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_subset_index[tuple(block[p] for p in lc)]
    
    # Precompute t_to_blocks: list of lists of block indices
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_blocks):
        for j in range(n_t_in_k):
            t_idx = block_t_indices[i, j]
            t_to_blocks[t_idx].append(i)
    
    # Convert to sets for fast intersection
    t_to_blocks_sets = [set(lst) for lst in t_to_blocks]
    
    # Also keep a list-of-lists version for iteration
    t_to_blocks_lists = [list(s) for s in t_to_blocks_sets]
    
    # Precompute block_masks as Python ints for bit operations in local search
    block_masks = [0] * n_blocks
    for i in range(n_blocks):
        mask = 0
        for idx in block_t_indices[i]:
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # Convert block_t_indices to list of lists for local search
    block_t_indices_lists = [list(row) for row in block_t_indices]
    
    def greedy_cover(weights=None, rng=None):
        """Greedy set cover with optional weights, using NumPy for speed."""
        if rng is None:
            rng = random.Random()
        
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        
        while uncovered.any():
            if time.time() > deadline:
                break
            
            if weights is None:
                gains = uncovered[block_t_indices].sum(axis=1)
            else:
                # Weighted gain: sum of weights for uncovered t-subsets in each block
                weighted_uncovered = np.where(uncovered, weights, 0.0)
                gains = weighted_uncovered[block_t_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
            
            # Pick randomly among top candidates for diversity
            threshold = max_gain * 0.95
            candidates = np.where(gains >= threshold)[0]
            if len(candidates) == 0:
                candidates = np.where(gains == max_gain)[0]
            
            best_idx = rng.choice(candidates)
            selected.append(int(best_idx))
            uncovered[block_t_indices[best_idx]] = False
        
        return selected
    
    def local_search(selected):
        """Aggressive local search: removal + 1-swap + 2-swap using set intersections."""
        if not selected:
            return selected
        
        selected = list(selected)
        selected_set = set(selected)
        
        # Initialize counts
        counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected:
            counts[block_t_indices[b_idx]] += 1
        
        # Phase 1: Iterative removal of redundant blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_selected = []
            for b_idx in selected:
                ts = block_t_indices[b_idx]
                if np.all(counts[ts] > 1):
                    counts[ts] -= 1
                    selected_set.discard(b_idx)
                    changed = True
                else:
                    new_selected.append(b_idx)
            selected = new_selected
        
        # Phase 2: 1-swap local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_selected = list(selected)
            new_set = set(new_selected)
            
            # Recompute counts
            counts = np.zeros(n_t, dtype=np.int32)
            for b_idx in new_selected:
                counts[block_t_indices[b_idx]] += 1
            
            for pos, b_idx in enumerate(list(new_selected)):
                if time.time() > deadline:
                    break
                
                ts = block_t_indices[b_idx]
                # Find t-subsets uniquely covered by b_idx
                unique_mask = counts[ts] == 1
                if not unique_mask.any():
                    # Redundant
                    counts[ts] -= 1
                    new_selected.remove(b_idx)
                    new_set.discard(b_idx)
                    improved = True
                    break
                
                unique_indices = ts[unique_mask]
                
                # Find candidates that cover all unique t-subsets
                if len(unique_indices) == 0:
                    continue
                    
                candidates = t_to_blocks_sets[unique_indices[0]]
                for t_idx in unique_indices[1:]:
                    candidates = candidates.intersection(t_to_blocks_sets[t_idx])
                
                candidates.difference_update(new_set)
                
                if candidates:
                    # Pick candidate with best additional coverage
                    best_cand = None
                    best_score = 0
                    cand_list = list(candidates)
                    # Sample if too many
                    if len(cand_list) > 50:
                        cand_list = random.sample(cand_list, 50)
                    
                    for cand in cand_list:
                        # Score: additional t-subsets covered
                        diff = block_masks[cand] & ~block_masks[b_idx]
                        score = diff.bit_count()
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                    
                    if best_cand is not None and best_score > 0:
                        # Perform swap
                        counts[block_t_indices[b_idx]] -= 1
                        counts[block_t_indices[best_cand]] += 1
                        new_selected.remove(b_idx)
                        new_selected.append(best_cand)
                        new_set.discard(b_idx)
                        new_set.add(best_cand)
                        improved = True
                        break
            
            selected = new_selected
        
        # Phase 3: 2-swap local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_selected = list(selected)
            new_set = set(new_selected)
            
            counts = np.zeros(n_t, dtype=np.int32)
            for b_idx in new_selected:
                counts[block_t_indices[b_idx]] += 1
            
            n_sel = len(new_selected)
            if n_sel < 2:
                break
            
            for i in range(n_sel):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_sel):
                    b1 = new_selected[i]
                    b2 = new_selected[j]
                    
                    # Find t-subsets uniquely covered by b1 or b2
                    ts1 = block_t_indices[b1]
                    ts2 = block_t_indices[b2]
                    
                    # t-subsets in b1 not in b2 with count==1
                    set1 = set(ts1)
                    set2 = set(ts2)
                    only1 = set1 - set2
                    only2 = set2 - set1
                    
                    unique_to_pair = []
                    for idx in only1:
                        if counts[idx] == 1:
                            unique_to_pair.append(idx)
                    for idx in only2:
                        if counts[idx] == 1:
                            unique_to_pair.append(idx)
                    
                    if not unique_to_pair:
                        # Both redundant
                        counts[ts1] -= 1
                        counts[ts2] -= 1
                        new_selected.remove(b1)
                        new_selected.remove(b2)
                        new_set.discard(b1)
                        new_set.discard(b2)
                        improved = True
                        break
                    
                    # Find 2 blocks that cover all unique_to_pair
                    # Use intersection approach
                    candidates_from_first = t_to_blocks_sets[unique_to_pair[0]]
                    
                    best_pair = None
                    best_score = 0
                    
                    cand_list = list(candidates_from_first)
                    if len(cand_list) > 30:
                        cand_list = random.sample(cand_list, 30)
                    
                    for c1 in cand_list:
                        if c1 in new_set or c1 == b1 or c1 == b2:
                            continue
                        
                        # Check what c1 covers of unique_to_pair
                        c1_mask = block_masks[c1]
                        remaining = [ui for ui in unique_to_pair if not (c1_mask >> ui) & 1]
                        
                        if not remaining:
                            # c1 covers all, find good c2
                            best_c2 = None
                            best_c2_score = 0
                            c2_candidates = t_to_blocks_lists[unique_to_pair[0]]
                            for c2 in c2_candidates:
                                if c2 in new_set or c2 == b1 or c2 == b2 or c2 == c1:
                                    continue
                                diff = (block_masks[c2] & ~block_masks[b1] & ~block_masks[b2])
                                score = diff.bit_count()
                                if score > best_c2_score:
                                    best_c2_score = score
                                    best_c2 = c2
                            if best_c2 is not None and best_c2_score > 0:
                                if best_c2_score > best_score:
                                    best_score = best_c2_score
                                    best_pair = (c1, best_c2)
                        else:
                            # Need c2 to cover remaining
                            c2_candidates = t_to_blocks_lists[remaining[0]]
                            for c2 in c2_candidates:
                                if c2 in new_set or c2 == b1 or c2 == b2 or c2 == c1:
                                    continue
                                # Check if c2 covers all remaining
                                c2_mask = block_masks[c2]
                                covers = all((c2_mask >> ui) & 1 for ui in remaining)
                                if not covers:
                                    continue
                                combined_new = (block_masks[c1] | block_masks[c2]) & ~block_masks[b1] & ~block_masks[b2]
                                score = combined_new.bit_count()
                                if score > best_score:
                                    best_score = score
                                    best_pair = (c1, c2)
                    
                    if best_pair and best_score > 0:
                        c1, c2 = best_pair
                        counts[ts1] -= 1
                        counts[ts2] -= 1
                        counts[block_t_indices[c1]] += 1
                        counts[block_t_indices[c2]] += 1
                        new_selected.remove(b1)
                        new_selected.remove(b2)
                        new_selected.append(c1)
                        new_selected.append(c2)
                        new_set.discard(b1)
                        new_set.discard(b2)
                        new_set.add(c1)
                        new_set.add(c2)
                        improved = True
                        break
                
                if improved:
                    break
            
            selected = new_selected
        
        return selected
    
    # Multi-restart with Lagrangian relaxation
    best_solution = None
    best_len = float('inf')
    rng = random.Random(42)
    
    # First: simple greedy
    sol1 = greedy_cover(rng=rng)
    sol1 = local_search(sol1)
    if sol1:
        best_solution = sol1
        best_len = len(sol1)
    
    # Second: Lagrangian-guided greedy
    weights = np.ones(n_t, dtype=np.float64)
    for iteration in range(8):
        if time.time() > deadline - 25:
            break
        
        sol = greedy_cover(weights=weights, rng=rng)
        if not sol:
            break
        
        covered = np.zeros(n_t, dtype=bool)
        for b_idx in sol:
            covered[block_t_indices[b_idx]] = True
        
        # Update weights: increase for uncovered, decrease for covered
        weights = np.where(covered, weights * 0.5, weights * 2.0)
        # Normalize to prevent overflow
        max_w = np.max(weights)
        if max_w > 0:
            weights = weights / max_w
        
        sol = local_search(sol)
        if sol and len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
    
    # Third: Randomized greedy (multiple trials)
    for trial in range(5):
        if time.time() > deadline - 15:
            break
        
        sol = greedy_cover(rng=rng)
        if sol:
            sol = local_search(sol)
            if sol and len(sol) < best_len:
                best_solution = sol
                best_len = len(sol)
    
    # Final: Try to improve best_solution with more aggressive local search
    if best_solution and time.time() < deadline - 5:
        best_solution = local_search(best_solution)
    
    # Convert indices back to blocks
    if best_solution:
        return [list(all_blocks[i]) for i in best_solution]
    else:
        return []
