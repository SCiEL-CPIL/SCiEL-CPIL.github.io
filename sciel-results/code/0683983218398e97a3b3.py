import itertools
import time
import random
import numpy as np
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    num_triplets = len(all_triplets)
    triplet_to_id = {trip: i for i, trip in enumerate(all_triplets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    triples_per_block = int(math.comb(k, t))
    
    # Map block tuple to index for O(1) lookup in local search
    block_to_idx = {block: i for i, block in enumerate(all_blocks)}
    
    # Precompute block_indices: shape (num_blocks, triples_per_block)
    block_indices = np.empty((num_blocks, triples_per_block), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        idxs = [triplet_to_id[sub] for sub in itertools.combinations(block, t)]
        block_indices[i] = idxs
        
    # Precompute frequency for weighted greedy
    freq = np.bincount(block_indices.flatten(), minlength=num_triplets)
    inv_freq = 1.0 / (freq + 1e-9)
    
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multiple Greedy Strategies
    num_restarts = 20
    seed = 0
    while time.time() < deadline - 30:
        strategy = seed % 3
        if strategy == 0:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_triplets, deadline, mode='greedy')
        elif strategy == 1:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_triplets, deadline, mode='weighted', inv_freq=inv_freq)
        else:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_triplets, deadline, mode='randomized', inv_freq=inv_freq, seed=seed)
        
        if blocks is None:
            seed += 1
            continue
            
        # Quick redundancy removal
        blocks = _remove_redundant_fast(blocks, block_indices, num_triplets)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        seed += 1
        
        if best_len <= 69:
            break

    # Phase 2: Local Search
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = _local_search_merge(best_blocks, all_blocks, block_indices, num_triplets, block_to_idx, deadline)
        
    if best_blocks is None:
        best_blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_triplets, deadline, mode='greedy')
        if best_blocks is None:
            best_blocks = []
            
    return [list(all_blocks[i]) for i in best_blocks]

def _greedy_cover_vectorized(all_blocks, block_indices, num_triplets, deadline, mode='greedy', inv_freq=None, seed=0):
    uncovered = np.ones(num_triplets, dtype=bool)
    selected_indices = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            return None
            
        if mode == 'greedy':
            gains = uncovered[block_indices].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False
            
        elif mode == 'weighted':
            w = np.where(uncovered, inv_freq, 0.0)
            gains = w[block_indices].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False
            
        elif mode == 'randomized':
            gains = uncovered[block_indices].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 10:
                rng = random.Random(seed + len(selected_indices))
                candidates = rng.sample(list(candidates), 10)
            best_idx = candidates[np.random.choice(len(candidates))]
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False
            
    if np.any(uncovered):
        return None

    return selected_indices

def _remove_redundant_fast(blocks, block_indices, num_triplets):
    if not blocks:
        return blocks
    
    counts = np.zeros(num_triplets, dtype=np.int32)
    for b_idx in blocks:
        counts[block_indices[b_idx]] += 1
        
    improved = list(blocks)
    n = len(improved)
    
    for i in range(n - 1, -1, -1):
        b_idx = improved[i]
        block_trips = block_indices[b_idx]
        if np.any(counts[block_trips] == 1):
            continue
        counts[block_trips] -= 1
        improved.pop(i)
        
    return improved

def _local_search_merge(blocks, all_blocks, block_indices, num_triplets, block_to_idx, deadline):
    if not blocks:
        return blocks
        
    current_blocks = list(blocks)
    
    while time.time() < deadline:
        changed = False
        
        # 1. Remove redundant
        new_blocks = _remove_redundant_fast(current_blocks, block_indices, num_triplets)
        if len(new_blocks) < len(current_blocks):
            current_blocks = new_blocks
            changed = True
            continue
            
        # 2. Merge 2-for-1
        n = len(current_blocks)
        if n < 2:
            break
            
        counts = np.zeros(num_triplets, dtype=np.int32)
        for b_idx in current_blocks:
            counts[block_indices[b_idx]] += 1
            
        merged = False
        for i in range(n):
            if time.time() > deadline:
                break
            if merged:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                if merged:
                    break
                    
                b1 = current_blocks[i]
                b2 = current_blocks[j]
                
                trips1 = block_indices[b1]
                trips2 = block_indices[b2]
                
                exc1 = trips1[counts[trips1] == 1]
                exc2 = trips2[counts[trips2] == 1]
                
                if len(exc1) == 0 or len(exc2) == 0:
                    continue
                
                points_needed = set()
                for tid in exc1:
                    points_needed.update(all_triplets[tid])
                for tid in exc2:
                    points_needed.update(all_triplets[tid])
                    
                if len(points_needed) > k:
                    continue
                    
                cand_set = set(points_needed)
                for p in range(v):
                    if len(cand_set) == k:
                        break
                    if p not in cand_set:
                        cand_set.add(p)
                        
                if len(cand_set) != k:
                    continue
                    
                cand_tuple = tuple(sorted(cand_set))
                
                if cand_tuple not in block_to_idx:
                    continue
                c_idx = block_to_idx[cand_tuple]
                
                trips_c = block_indices[c_idx]
                
                affected = set(trips1) | set(trips2) | set(trips_c)
                
                valid = True
                for tid in affected:
                    new_count = counts[tid]
                    if tid in trips1:
                        new_count -= 1
                    if tid in trips2:
                        new_count -= 1
                    if tid in trips_c:
                        new_count += 1
                    if new_count == 0:
                        valid = False
                        break
                        
                if valid:
                    new_blocks = []
                    for b in current_blocks:
                        if b != b1 and b != b2:
                            new_blocks.append(b)
                    new_blocks.append(c_idx)
                    
                    current_blocks = new_blocks
                    merged = True
                    changed = True
                    break
                    
        if not merged and not changed:
            break
            
    return current_blocks

# Need to define v, k, all_triplets in scope for _local_search_merge
# They are passed or available in construct?
# _local_search_merge uses v, k, all_triplets.
# I should pass them or make them global? 
# Better to pass them.

# Update signature of _local_search_merge to accept v, k, all_triplets.
