import time
import random
import itertools
import numpy as np

_CACHE = {}

def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    all_subsets = list(itertools.combinations(points, t))
    
    n_blocks = len(all_blocks)
    n_subsets = len(all_subsets)
    
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    block_subsets = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(subset_to_idx[s])
        block_subsets.append(subs)
        
    blocks_covering_subset = [[] for _ in range(n_subsets)]
    for i, subs in enumerate(block_subsets):
        for s_idx in subs:
            blocks_covering_subset[s_idx].append(i)
    
    # Precompute block_subsets as numpy arrays for vectorized ops
    block_subsets_np = []
    for subs in block_subsets:
        block_subsets_np.append(np.array(subs, dtype=np.int32))
    
    # Build incidence matrix: block_subsets_matrix[i] = set of subset indices block i covers
    # For memory efficiency with C(19,7)=50388 blocks and C(19,4)=3876 subsets,
    # use a sparse representation via sorted arrays
    
    _CACHE[key] = (all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets)
    return _CACHE[key]

def _greedy_construct(v, k, t, deadline, rng, seed_offset=0):
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    uncovered = np.ones(n_subsets, dtype=np.int8)
    n_uncovered = n_subsets
    gain = np.array([len(block_subsets[i]) for i in range(n_blocks)], dtype=np.int32)
    selected = []
    selected_set = set()
    
    while n_uncovered > 0 and time.time() < deadline:
        best_block = -1
        best_score = -1
        
        for i in range(n_blocks):
            if i in selected_set:
                continue
            if gain[i] > best_score:
                best_score = gain[i]
                best_block = i
                
        if best_block == -1 or best_score <= 0:
            break
            
        selected.append(best_block)
        selected_set.add(best_block)
        
        for s_idx in block_subsets[best_block]:
            if uncovered[s_idx] == 1:
                uncovered[s_idx] = 0
                n_uncovered -= 1
                for b_idx in blocks_covering_subset[s_idx]:
                    if b_idx not in selected_set:
                        gain[b_idx] -= 1
                        
    return selected, selected_set

def _minimize(selected, selected_set, block_subsets, n_subsets, rng, deadline):
    counts = np.zeros(n_subsets, dtype=np.int32)
    for b in selected:
        for s_idx in block_subsets[b]:
            counts[s_idx] += 1
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        items = list(selected)
        random.shuffle(items)
        for b in items:
            if b not in selected_set:
                continue
            is_redundant = True
            for s_idx in block_subsets[b]:
                if counts[s_idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                for s_idx in block_subsets[b]:
                    counts[s_idx] -= 1
                selected_set.remove(b)
                selected.remove(b)
                changed = True
    
    return selected, selected_set, counts

def _local_search(selected, selected_set, counts, block_subsets, blocks_covering_subset, n_subsets, deadline, rng):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Remove redundant blocks
        i = 0
        while i < len(selected):
            b = selected[i]
            is_redundant = True
            for s_idx in block_subsets[b]:
                if counts[s_idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                for s_idx in block_subsets[b]:
                    counts[s_idx] -= 1
                selected.pop(i)
                selected_set.remove(b)
                improved = True
            else:
                i += 1
                
        if time.time() > deadline:
            break
            
        # Try swaps: for each block, try to find a replacement that covers its critical subsets
        current_selected = list(selected)
        random.shuffle(current_selected)
        
        for b_out in current_selected:
            if b_out not in selected_set:
                continue
                
            crit_subs = [s_idx for s_idx in block_subsets[b_out] if counts[s_idx] == 1]
            if not crit_subs:
                continue
                
            crit_subs_sorted = sorted(crit_subs, key=lambda s: len(blocks_covering_subset[s]))
            
            candidates = set(blocks_covering_subset[crit_subs_sorted[0]])
            for s_idx in crit_subs_sorted[1:]:
                candidates &= set(blocks_covering_subset[s_idx])
                if not candidates:
                    break
                    
            candidates -= selected_set
            
            if not candidates:
                continue
            
            # Score candidates: prefer those that minimize new critical subsets and maximize coverage
            best_b_in = -1
            best_score = float('inf')
            for b_in in candidates:
                # Count how many subsets would become critical (count drops to 1)
                # and how many new subsets get added
                score = 0
                for s_idx in block_subsets[b_in]:
                    c = counts[s_idx]
                    if c == 0:
                        score += 3  # new coverage
                    elif c == 1:
                        score += 1  # stays critical, neutral
                    elif c == 2:
                        score -= 2  # becomes critical, bad
                    else:
                        score -= 1  # was well-covered, slightly worse
                # Also penalize blocks that don't cover any currently uncovered
                if score < best_score:
                    best_score = score
                    best_b_in = b_in
                    
            if best_b_in == -1:
                continue
            
            # Perform swap
            for s_idx in block_subsets[b_out]:
                counts[s_idx] -= 1
            selected_set.remove(b_out)
            selected.remove(b_out)
            
            for s_idx in block_subsets[best_b_in]:
                counts[s_idx] += 1
            selected_set.add(best_b_in)
            selected.append(best_b_in)
            
            improved = True
            
            # Minimize after swap
            j = 0
            while j < len(selected):
                b = selected[j]
                is_redundant = True
                for s_idx in block_subsets[b]:
                    if counts[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s_idx in block_subsets[b]:
                        counts[s_idx] -= 1
                    selected.pop(j)
                    selected_set.remove(b)
                    improved = True
                else:
                    j += 1
                    
            if time.time() > deadline:
                break
                
    return selected, selected_set, counts

def _fill_complete(selected, selected_set, block_subsets, blocks_covering_subset, n_subsets, deadline):
    """Ensure all subsets are covered by adding blocks as needed."""
    counts = np.zeros(n_subsets, dtype=np.int32)
    for b in selected:
        for s_idx in block_subsets[b]:
            counts[s_idx] += 1
    
    # Find uncovered subsets
    uncovered = np.where(counts == 0)[0]
    if len(uncovered) == 0:
        return selected, selected_set, counts
    
    # For each uncovered subset, find a block that covers it
    for s_idx in uncovered:
        if time.time() > deadline:
            break
        # Find a block covering this subset, preferring one that covers most other uncovered subsets
        best_block = -1
        best_gain = -1
        for b_idx in blocks_covering_subset[s_idx]:
            if b_idx in selected_set:
                continue
            gain = 0
            for s in block_subsets[b_idx]:
                if counts[s] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
        if best_block == -1:
            # Try any block
            for b_idx in blocks_covering_subset[s_idx]:
                if b_idx not in selected_set:
                    best_block = b_idx
                    break
        if best_block != -1:
            for s in block_subsets[best_block]:
                counts[s] += 1
            selected.append(best_block)
            selected_set.add(best_block)
    
    return selected, selected_set, counts

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0
    
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    rng = random.Random(42)
    
    # Multi-start: try several greedy initializations
    best_selected = None
    best_set = None
    best_count = float('inf')
    
    num_starts = 3
    start_budget = (deadline - t0) / (num_starts + 2)
    
    for start_idx in range(num_starts):
        start_deadline = t0 + (start_idx + 1) * start_budget
        if time.time() > start_deadline:
            break
            
        selected, selected_set = _greedy_construct(v, k, t, start_deadline, rng, start_idx)
        
        # Minimize
        selected, selected_set, counts = _minimize(selected, selected_set, block_subsets, n_subsets, rng, start_deadline)
        
        # Fill to ensure completeness
        selected, selected_set, counts = _fill_complete(selected, selected_set, block_subsets, blocks_covering_subset, n_subsets, start_deadline)
        
        # Local search
        ls_deadline = min(start_deadline + start_budget, deadline - 10)
        selected, selected_set, counts = _local_search(
            selected, selected_set, counts,
            block_subsets, blocks_covering_subset,
            n_subsets, ls_deadline, rng
        )
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_selected = selected[:]
            best_set = selected_set.copy()
    
    # If no multi-start completed, use the last one
    if best_selected is None:
        selected, selected_set = _greedy_construct(v, k, t, deadline - 5, rng, 0)
        selected, selected_set, counts = _minimize(selected, selected_set, block_subsets, n_subsets, rng, deadline - 5)
        selected, selected_set, counts = _fill_complete(selected, selected_set, block_subsets, blocks_covering_subset, n_subsets, deadline - 5)
        best_selected = selected
        best_set = selected_set
    
    # Final polish with remaining time
    if time.time() < deadline - 5:
        counts = np.zeros(n_subsets, dtype=np.int32)
        for b in best_selected:
            for s_idx in block_subsets[b]:
                counts[s_idx] += 1
        best_selected, best_set, counts = _local_search(
            best_selected, best_set, counts,
            block_subsets, blocks_covering_subset,
            n_subsets, deadline - 3, rng
        )
        # Final fill to ensure completeness
        best_selected, best_set, counts = _fill_complete(best_selected, best_set, block_subsets, blocks_covering_subset, n_subsets, deadline - 1)
    
    result = [list(all_blocks[i]) for i in best_selected]
    return result
