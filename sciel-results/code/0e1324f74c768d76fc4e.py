import itertools
import random
import time
import sys
import numpy as np

# Global cache
_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    start_time = time.time()
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # Precompute block_triples as a numpy array for speed
    block_triples = []
    for block in all_blocks:
        block_triples.append([triple_to_idx[tri] for tri in itertools.combinations(block, t)])
    
    # Build blocks_containing
    blocks_containing = [[] for _ in range(n_triples)]
    for i, tris in enumerate(block_triples):
        for tri_idx in tris:
            blocks_containing[tri_idx].append(i)
    
    # Build block_masks using numpy for speed
    # Each block covers C(k,t) triples
    # Use a numpy-based approach: for each triple, mark which blocks contain it
    triple_block_matrix = np.zeros((n_triples, n_blocks), dtype=bool)
    for i, tris in enumerate(block_triples):
        for tri_idx in tris:
            triple_block_matrix[tri_idx, i] = True
    
    # block_masks as list of ints (bitmasks) - but for large n_triples this is problematic
    # Instead, store the triple_block_matrix and use it directly
    # For greedy, we need: for each block, how many uncovered triples it covers
    
    # Actually, let's use a different approach for speed:
    # block_triples as numpy array of lists
    # For gain computation in greedy, use numpy
    
    full_mask = (1 << n_triples) - 1
    
    _CACHE[key] = (all_blocks, triple_block_matrix, full_mask, blocks_containing, n_blocks, n_triples, block_triples)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, triple_block_matrix, full_mask, blocks_containing, n_blocks, seed, block_triples):
    rng = random.Random(seed)
    
    # uncovered: boolean array of which triples are uncovered
    uncovered = np.ones(len(block_triples), dtype=bool)  # all triples uncovered initially
    n_triples = len(block_triples)
    
    blocks_used = []
    active_blocks = set(range(n_blocks))
    
    while uncovered.any():
        best_block_idx = -1
        max_gain = 0
        uncovered_count = uncovered.sum()
        
        # Vectorized gain computation
        # For each active block, count how many uncovered triples it covers
        active_list = list(active_blocks)
        if len(active_list) > 1000:
            # Use numpy for large active set
            active_arr = np.array(active_list)
            # triple_block_matrix is (n_triples, n_blocks)
            # We want for each block in active_arr: sum of uncovered[i] where triple_block_matrix[i, block] is True
            # This is: triple_block_matrix[:, active_arr] @ uncovered
            gains = triple_block_matrix[:, active_arr].T @ uncovered.astype(np.int32)
            best_pos = np.argmax(gains)
            best_block_idx = active_list[best_pos]
            max_gain = int(gains[best_pos])
        else:
            for idx in active_list:
                gain = int(triple_block_matrix[:, idx] @ uncovered.astype(np.int32))
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = idx
                    if gain == uncovered_count:
                        break
        
        if best_block_idx == -1 or max_gain == 0:
            break
            
        blocks_used.append(best_block_idx)
        # Update uncovered
        uncovered = uncovered & ~triple_block_matrix[:, best_block_idx].astype(bool)
        active_blocks.remove(best_block_idx)
        
    return blocks_used

def _local_search(v, k, t, initial_blocks, triple_block_matrix, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, deadline):
    rng = random.Random(seed)
    
    selected = set(initial_blocks)
    coverage_count = np.zeros(n_triples, dtype=np.int32)
    
    for b in selected:
        for tri_idx in block_triples[b]:
            coverage_count[tri_idx] += 1
    
    best_size = len(selected)
    best_selected = set(selected)
    
    no_improve_time = time.time()
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to remove a redundant block
        if selected:
            items = list(selected)
            rng.shuffle(items)
            for b in items:
                redundant = True
                for tri_idx in block_triples[b]:
                    if coverage_count[tri_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for tri_idx in block_triples[b]:
                        coverage_count[tri_idx] -= 1
                    selected.remove(b)
                    improved = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                    no_improve_time = time.time()
                    break
        
        if not improved:
            # 2. Try swaps: find block with unique triples, replace with better one
            swap_done = False
            sel_list = list(selected)
            rng.shuffle(sel_list)
            
            for b in sel_list:
                # Find unique triples for b
                unique_tris = []
                for tri_idx in block_triples[b]:
                    if coverage_count[tri_idx] == 1:
                        unique_tris.append(tri_idx)
                
                if not unique_tris:
                    continue
                
                # Pick a random unique triple to find candidates
                tri_idx = rng.choice(unique_tris)
                candidates = blocks_containing[tri_idx]
                
                if len(candidates) <= 1:
                    continue
                
                # Shuffle candidates and try a few
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                for b_prime in cand_list[:80]:
                    if b_prime in selected:
                        continue
                    
                    # Check if b_prime covers all unique triples of b
                    covers_all = True
                    for ut in unique_tris:
                        if not triple_block_matrix[ut, b_prime]:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Perform swap
                    for tri_idx2 in block_triples[b]:
                        coverage_count[tri_idx2] -= 1
                    selected.remove(b)
                    
                    for tri_idx2 in block_triples[b_prime]:
                        coverage_count[tri_idx2] += 1
                    selected.add(b_prime)
                    
                    swap_done = True
                    improved = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                    no_improve_time = time.time()
                    break
                
                if swap_done:
                    break
        
        if not improved:
            # 3. Simulated Annealing: accept worse moves with some probability to escape local minima
            sa_done = False
            sel_list = list(selected)
            rng.shuffle(sel_list)
            
            # Temperature based on how long we've been stuck
            stuck_time = time.time() - no_improve_time
            temperature = max(0.1, 2.0 - stuck_time * 0.1)
            
            for b in sel_list:
                # Find unique triples for b
                unique_tris = []
                for tri_idx in block_triples[b]:
                    if coverage_count[tri_idx] == 1:
                        unique_tris.append(tri_idx)
                
                if not unique_tris:
                    continue
                
                # Try to replace b with a random block that covers at least some unique triples
                tri_idx = rng.choice(unique_tris)
                candidates = blocks_containing[tri_idx]
                
                if len(candidates) <= 1:
                    continue
                
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                # Try a few candidates
                for b_prime in cand_list[:30]:
                    if b_prime in selected:
                        continue
                    
                    # Compute how many unique triples b_prime covers
                    covered_unique = sum(1 for ut in unique_tris if triple_block_matrix[ut, b_prime])
                    
                    # If it covers all unique triples, it's a valid swap (same size)
                    # If it covers fewer, it's a "worse" move but might help escape
                    if covered_unique < len(unique_tris):
                        # Accept with probability based on temperature
                        # The "cost" is the number of unique triples not covered
                        cost = len(unique_tris) - covered_unique
                        if rng.random() > np.exp(-cost / temperature):
                            continue
                    
                    # Perform swap
                    for tri_idx2 in block_triples[b]:
                        coverage_count[tri_idx2] -= 1
                    selected.remove(b)
                    
                    for tri_idx2 in block_triples[b_prime]:
                        coverage_count[tri_idx2] += 1
                    selected.add(b_prime)
                    
                    sa_done = True
                    improved = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                    no_improve_time = time.time()
                    break
                
                if sa_done:
                    break
        
        if not improved:
            # 4. Try 2-opt exchange: remove one block, add two new blocks
            if len(selected) >= 2:
                two_opt_done = False
                sel_list = list(selected)
                rng.shuffle(sel_list)
                
                for b in sel_list:
                    unique_tris = []
                    for tri_idx in block_triples[b]:
                        if coverage_count[tri_idx] == 1:
                            unique_tris.append(tri_idx)
                    
                    if not unique_tris:
                        continue
                    
                    tri_to_cands = {}
                    for ut in unique_tris:
                        cands = blocks_containing[ut]
                        valid_cands = [c for c in cands if c not in selected]
                        if not valid_cands:
                            break
                        tri_to_cands[ut] = valid_cands
                    
                    if len(tri_to_cands) < len(unique_tris):
                        continue
                    
                    all_cands = set(tri_to_cands[unique_tris[0]])
                    for ut in unique_tris[1:]:
                        all_cands &= set(tri_to_cands[ut])
                    
                    if len(all_cands) < 2:
                        continue
                    
                    cand_list = list(all_cands)
                    rng.shuffle(cand_list)
                    
                    for i in range(min(len(cand_list), 30)):
                        for j in range(i + 1, min(len(cand_list), 30)):
                            b1, b2 = cand_list[i], cand_list[j]
                            
                            covers_all = True
                            for ut in unique_tris:
                                if not (triple_block_matrix[ut, b1] or triple_block_matrix[ut, b2]):
                                    covers_all = False
                                    break
                            
                            if not covers_all:
                                continue
                            
                            for tri_idx2 in block_triples[b]:
                                coverage_count[tri_idx2] -= 1
                            selected.remove(b)
                            
                            for tri_idx2 in block_triples[b1]:
                                coverage_count[tri_idx2] += 1
                            selected.add(b1)
                            
                            for tri_idx2 in block_triples[b2]:
                                coverage_count[tri_idx2] += 1
                            selected.add(b2)
                            
                            two_opt_done = True
                            improved = True
                            if len(selected) < best_size:
                                best_size = len(selected)
                                best_selected = set(selected)
                            no_improve_time = time.time()
                            break
                        
                        if two_opt_done:
                            break
                    
                    if two_opt_done:
                        break
        
        if not improved:
            if time.time() - no_improve_time > 3.0:
                break
    
    return list(best_selected)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    all_blocks, triple_block_matrix, full_mask, blocks_containing, n_blocks, n_triples, block_triples = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    
    while time.time() < deadline - 5:
        initial_blocks = _greedy_construct(v, k, t, all_blocks, triple_block_matrix, full_mask, blocks_containing, n_blocks, seed, block_triples)
        
        if not initial_blocks:
            seed += 1
            continue
        
        remaining = deadline - time.time()
        ls_deadline = min(time.time() + min(12.0, remaining - 2.0), deadline - 3.0)
        
        if ls_deadline <= time.time():
            break
            
        blocks = _local_search(v, k, t, initial_blocks, triple_block_matrix, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, ls_deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            print(f"New best: {best_len} blocks at t={time.time()-start_time:.1f}s")
        
        seed += 1
    
    if best_blocks is None:
        blocks = _greedy_construct(v, k, t, all_blocks, triple_block_matrix, full_mask, blocks_containing, n_blocks, 999, block_triples)
        best_blocks = blocks
    
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result
