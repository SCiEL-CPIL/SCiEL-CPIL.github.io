import time
import random
import itertools
import numpy as np

# Global cache for precomputed data
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
    
    # Map subset tuple to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # For each block, list of subset indices it covers
    block_subsets = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(subset_to_idx[s])
        block_subsets.append(subs)
        
    # For each subset, list of block indices that cover it
    # This is crucial for fast swap search
    blocks_covering_subset = [[] for _ in range(n_subsets)]
    for i, subs in enumerate(block_subsets):
        for s_idx in subs:
            blocks_covering_subset[s_idx].append(i)
            
    # Convert to numpy arrays for greedy phase if needed, 
    # but for local search, lists of lists are fine.
    # For greedy, we can use a sparse matrix representation or just iterate.
    # Let's keep it simple with lists for local search.
    
    _CACHE[key] = (all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets)
    return _CACHE[key]

def _greedy_construct(v, k, t, deadline, rng):
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    uncovered = np.ones(n_subsets, dtype=np.int8)
    counts = np.zeros(n_subsets, dtype=np.int32)
    selected = []
    selected_set = set()
    
    # Build a sparse incidence matrix: block_subsets as a list of arrays
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_subsets]
    
    while uncovered.any() and time.time() < deadline:
        # Vectorized scoring: for each unselected block, count uncovered subsets it covers
        best_block = -1
        best_score = -1
        
        # Process in chunks to avoid huge memory usage
        chunk_size = 5000
        for start in range(0, n_blocks, chunk_size):
            end = min(start + chunk_size, n_blocks)
            for i in range(start, end):
                if i in selected_set:
                    continue
                subs = block_subset_arrays[i]
                score = uncovered[subs].sum()
                if score > best_score:
                    best_score = score
                    best_block = i
        
        if best_block == -1 or best_score <= 0:
            break
            
        selected.append(best_block)
        selected_set.add(best_block)
        
        # Update uncovered and counts
        subs = block_subset_arrays[best_block]
        counts[subs] += 1
        uncovered[subs] = 0
        
    return selected, counts.tolist(), selected_set

def _local_search(selected, counts, selected_set, block_subsets, blocks_covering_subset, n_subsets, deadline, rng):
    # selected: list of block indices
    # counts: list of int
    # selected_set: set of block indices
    
    # Precompute numpy arrays for faster counting
    counts_np = np.array(counts, dtype=np.int32)
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_subsets]
    
    improved = True
    no_improve_rounds = 0
    max_no_improve = 3
    
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Remove redundant blocks (multiple passes)
        for _pass in range(3):
            i = 0
            while i < len(selected):
                b = selected[i]
                subs = block_subset_arrays[b]
                # Check if any subset has count == 1
                if not (counts_np[subs] == 1).any():
                    # Remove block b
                    counts_np[subs] -= 1
                    selected.pop(i)
                    selected_set.remove(b)
                    improved = True
                else:
                    i += 1
            
            if not improved:
                break
        
        if time.time() > deadline:
            break
            
        if no_improve_rounds >= max_no_improve:
            break
            
        # Phase 2: Swap
        # Find the block with the most critical subsets (count == 1)
        best_swap_out = -1
        best_crit_count = 0
        best_crit_subs = []
        
        for b in selected:
            subs = block_subset_arrays[b]
            crit_mask = counts_np[subs] == 1
            crit_count = crit_mask.sum()
            if crit_count > best_crit_count:
                best_crit_count = crit_count
                best_swap_out = b
                best_crit_subs = subs[crit_mask].tolist()
        
        if best_swap_out == -1 or best_crit_count == 0:
            no_improve_rounds += 1
            continue
        
        # Find candidates for B that cover all critical subsets
        best_crit_subs_sorted = sorted(best_crit_subs, key=lambda s: len(blocks_covering_subset[s]))
        
        candidates = set(blocks_covering_subset[best_crit_subs_sorted[0]])
        for s_idx in best_crit_subs_sorted[1:]:
            candidates &= set(blocks_covering_subset[s_idx])
        
        candidates -= selected_set
        
        if not candidates:
            # Try a few random blocks as swap-out candidates
            found_candidate = False
            for _ in range(10):
                b = rng.choice(selected)
                subs = block_subset_arrays[b]
                crit_mask = counts_np[subs] == 1
                if not crit_mask.any():
                    continue
                crit_subs = subs[crit_mask].tolist()
                crit_subs_sorted = sorted(crit_subs, key=lambda s: len(blocks_covering_subset[s]))
                cands = set(blocks_covering_subset[crit_subs_sorted[0]])
                for s_idx in crit_subs_sorted[1:]:
                    cands &= set(blocks_covering_subset[s_idx])
                cands -= selected_set
                if cands:
                    best_swap_out = b
                    candidates = cands
                    found_candidate = True
                    break
            
            if not found_candidate:
                no_improve_rounds += 1
                continue
        
        # Pick the best B from candidates
        # Score: maximize coverage of low-count subsets (they're "scarce")
        best_b = -1
        best_b_score = -1
        
        for b in candidates:
            subs = block_subset_arrays[b]
            c = counts_np[subs]
            # Weighted score: count 1 gets high weight, count 2 medium, count 0 low
            score = (c == 1).sum() * 10 + (c == 2).sum() * 5 + (c == 0).sum() * 2
            if score > best_b_score:
                best_b_score = score
                best_b = b
        
        if best_b == -1:
            no_improve_rounds += 1
            continue
        
        # Perform swap
        subs_out = block_subset_arrays[best_swap_out]
        subs_in = block_subset_arrays[best_b]
        
        counts_np[subs_out] -= 1
        counts_np[subs_in] += 1
        
        selected_set.remove(best_swap_out)
        selected.remove(best_swap_out)
        
        selected_set.add(best_b)
        selected.append(best_b)
        
        improved = True
        no_improve_rounds = 0
        
    counts = counts_np.tolist()
    return selected, counts, selected_set

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin
    
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    rng = random.Random(12345)
    
    # Phase 1: Greedy Construction
    # We use 50% of the time for construction
    cons_deadline = t0 + (deadline - t0) * 0.5
    
    selected, counts, selected_set = _greedy_construct(v, k, t, cons_deadline, rng)
    
    # Phase 2: Local Search
    ls_deadline = deadline - 5.0
    selected, counts, selected_set = _local_search(
        selected, counts, selected_set, 
        block_subsets, blocks_covering_subset, 
        n_subsets, ls_deadline, rng
    )
    
    # Phase 3: Second Local Search Pass with different random seed (if time remains)
    if time.time() < deadline - 15.0:
        rng2 = random.Random(54321)
        selected, counts, selected_set = _local_search(
            selected, counts, selected_set, 
            block_subsets, blocks_covering_subset, 
            n_subsets, deadline - 5.0, rng2
        )
    
    # Convert to list of lists of points
    result = [list(all_blocks[i]) for i in selected]
    
    return result
