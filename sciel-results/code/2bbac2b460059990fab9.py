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
    blocks_covering_subset = [[] for _ in range(n_subsets)]
    for i, subs in enumerate(block_subsets):
        for s_idx in subs:
            blocks_covering_subset[s_idx].append(i)
            
    _CACHE[key] = (all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets)
    return _CACHE[key]

def _greedy_construct(v, k, t, deadline, rng):
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    counts = [0] * n_subsets
    selected = []
    selected_set = set()
    
    # To speed up "is it uncovered" checks, we can just use counts == 0.
    # But finding a random uncovered subset requires knowing which are uncovered.
    # We can maintain a list of uncovered indices, but updating it is O(1) per subset?
    # No, removing from a list is O(N).
    # Alternative: Just pick a random index and check if count==0.
    # If many are uncovered, this is fast. If few are uncovered, this is slow.
    # Better: Maintain a list of uncovered indices.
    uncovered_indices = list(range(n_subsets))
    
    while uncovered_indices and time.time() < deadline:
        # Pick a random uncovered subset
        # Using a list and random.choice is O(1) if we just pick, but removal is O(N).
        # We can swap the last element to the position and pop.
        idx_in_list = rng.randrange(len(uncovered_indices))
        target_idx = uncovered_indices[idx_in_list]
        
        # Remove target_idx from uncovered_indices
        uncovered_indices[idx_in_list] = uncovered_indices[-1]
        uncovered_indices.pop()
        
        # Find the best block covering target_idx
        candidates = blocks_covering_subset[target_idx]
        
        best_block = -1
        best_gain = -1
        
        # Evaluate candidates
        # Gain = number of subsets in block with count == 0
        # Note: We just removed target_idx from uncovered_indices, but its count is still 0.
        # Other subsets in the block might have count > 0.
        
        for b_idx in candidates:
            gain = 0
            for s_idx in block_subsets[b_idx]:
                if counts[s_idx] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                
        if best_block == -1:
            # Should not happen if target_idx was uncovered
            break
            
        # Add best_block
        selected.append(best_block)
        selected_set.add(best_block)
        
        # Update counts and uncovered_indices
        for s_idx in block_subsets[best_block]:
            if counts[s_idx] == 0:
                # It was uncovered, so it should be in uncovered_indices
                # We need to remove it.
                # Finding it in the list is O(N).
                # To avoid O(N) search, we can use a dict: subset_idx -> index_in_uncovered_list
                # But that's complex.
                # Alternative: Just mark it as covered in counts, and let the "random pick" 
                # occasionally pick an already covered one? 
                # If we pick a covered one, gain will be 0 (or low), and we might pick a bad block.
                # Better to keep the list accurate.
                
                # Since n_subsets is ~38k, O(N) search is 38k ops.
                # Greedy steps are ~150. 150 * 38k * (avg candidates) ...
                # Actually, we only remove one subset per "pick", but the block covers ~35 subsets.
                # So we remove up to 35 subsets per step.
                # 35 * 38k = 1.3M ops per step. 150 steps = 200M ops. Might be slow in Python.
                
                # Optimization: Instead of removing immediately, we can just check 
                # `if counts[s_idx] == 0` in the gain calculation.
                # And for picking the target, we can pick from the full range and check?
                # No, that's inefficient if few are uncovered.
                
                # Let's try a different approach for the list:
                # We don't strictly need to remove from the list immediately.
                # We can just rely on `counts[s_idx] == 0` in the gain calc.
                # For picking the target, we can pick a random index from 0 to n_subsets-1 
                # and check if counts[idx] == 0. 
                # If the fraction of uncovered is high, this is fast.
                # If low, it's slow.
                # Given we start with all uncovered and end with 0, the average is 50%.
                # So on average, 2 tries per pick.
                # This is much faster than O(N) removal.
                
                pass # We will handle this in the main loop logic change below
        
        # Re-implementing the loop with the "random index check" strategy
        
    # The above loop was flawed in implementation details. 
    # Let's restart the greedy function with the "random index check" strategy.
    
    selected.clear()
    selected_set.clear()
    counts = [0] * n_subsets
    
    while time.time() < deadline:
        # Check if any uncovered
        # This is O(N). Doing it every step is 150 * 38k = 5.7M ops. Acceptable.
        uncovered_exists = False
        for i in range(n_subsets):
            if counts[i] == 0:
                uncovered_exists = True
                break
        
        if not uncovered_exists:
            break
            
        # Pick a random uncovered subset
        # Try a few times to find an uncovered one
        target_idx = -1
        for _ in range(10):
            cand = rng.randrange(n_subsets)
            if counts[cand] == 0:
                target_idx = cand
                break
        
        if target_idx == -1:
            # Fallback: linear scan
            for i in range(n_subsets):
                if counts[i] == 0:
                    target_idx = i
                    break
            if target_idx == -1:
                break
                
        # Find best block covering target_idx
        candidates = blocks_covering_subset[target_idx]
        best_block = -1
        best_gain = -1
        
        for b_idx in candidates:
            gain = 0
            for s_idx in block_subsets[b_idx]:
                if counts[s_idx] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                
        if best_block == -1:
            break
            
        selected.append(best_block)
        selected_set.add(best_block)
        
        for s_idx in block_subsets[best_block]:
            counts[s_idx] += 1
            
    return selected, counts, selected_set

def _local_search(selected, counts, selected_set, block_subsets, blocks_covering_subset, n_subsets, deadline, rng):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Remove redundant blocks
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
            
        # Phase 2: Swap
        # Find block with most critical subsets
        best_swap_out = -1
        best_crit_count = 0
        best_crit_subs = []
        
        for b in selected:
            crit_subs = [s_idx for s_idx in block_subsets[b] if counts[s_idx] == 1]
            if len(crit_subs) > best_crit_count:
                best_crit_count = len(crit_subs)
                best_swap_out = b
                best_crit_subs = crit_subs
                
        if best_swap_out == -1:
            break
            
        if not best_crit_subs:
            continue
            
        best_crit_subs_sorted = sorted(best_crit_subs, key=lambda s: len(blocks_covering_subset[s]))
        
        candidates = set(blocks_covering_subset[best_crit_subs_sorted[0]])
        for s_idx in best_crit_subs_sorted[1:]:
            candidates &= set(blocks_covering_subset[s_idx])
            
        candidates -= selected_set
        
        if not candidates:
            # Try a few random blocks
            found_candidate = False
            for _ in range(5):
                b = rng.choice(selected)
                crit_subs = [s_idx for s_idx in block_subsets[b] if counts[s_idx] == 1]
                if not crit_subs:
                    continue
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
                break
                
        best_b = -1
        best_b_score = -1
        
        for b in candidates:
            score = 0
            for s_idx in block_subsets[b]:
                c = counts[s_idx]
                if c == 1:
                    score += 10
                elif c == 2:
                    score += 5
                elif c == 0:
                    score += 2
            if score > best_b_score:
                best_b_score = score
                best_b = b
                
        if best_b == -1:
            break
            
        # Perform swap
        for s_idx in block_subsets[best_swap_out]:
            counts[s_idx] -= 1
        selected_set.remove(best_swap_out)
        selected.remove(best_swap_out)
        
        for s_idx in block_subsets[best_b]:
            counts[s_idx] += 1
        selected_set.add(best_b)
        selected.append(best_b)
        
        improved = True
        
    return selected, counts, selected_set

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0
    
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    rng = random.Random(12345)
    
    # Phase 1: Greedy
    cons_deadline = t0 + (deadline - t0) * 0.5
    selected, counts, selected_set = _greedy_construct(v, k, t, cons_deadline, rng)
    
    # Phase 2: Local Search
    ls_deadline = deadline - 5.0
    selected, counts, selected_set = _local_search(
        selected, counts, selected_set, 
        block_subsets, blocks_covering_subset, 
        n_subsets, ls_deadline, rng
    )
    
    # Phase 3: Second Local Search Pass
    if time.time() < deadline - 10.0:
        selected, counts, selected_set = _local_search(
            selected, counts, selected_set, 
            block_subsets, blocks_covering_subset, 
            n_subsets, deadline - 5.0, rng
        )
        
    result = [list(all_blocks[i]) for i in selected]
    
    return result
