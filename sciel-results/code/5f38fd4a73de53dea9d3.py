"""Optimized covering design C(24,7,3) using weighted greedy + local search."""
import itertools
import random
import time
import math
import numpy as np


def _precompute_masks(v, k, t):
    """Precompute all t-subsets and map each k-subset to its covered t-subset indices."""
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_idx = {}
    for i, s in enumerate(all_t_subsets):
        t_idx[s] = i
    
    # For each k-subset, compute which t-subsets it covers
    # We'll build a lookup: for a given block (tuple of k points), return list of t-subset indices
    # But storing all C(24,7) = 346104 blocks is feasible
    # Instead, we compute on-the-fly but cache
    
    return all_t_subsets, t_idx, n_t


def _block_to_indices(block, t_idx, t):
    """Convert a block to list of t-subset indices it covers."""
    indices = []
    for c in itertools.combinations(block, t):
        idx = t_idx[c]
        indices.append(idx)
    return indices


def _greedy_construct(v, k, t, rng, all_t_subsets, t_idx, n_t):
    """Weighted greedy construction."""
    # counts[i] = number of blocks covering t-subset i
    counts = np.zeros(n_t, dtype=np.int32)
    blocks = []
    block_indices_list = []  # list of lists of t-subset indices per block
    
    # Weights: initially all 1, updated as coverage increases
    # Use a priority: prefer blocks that cover more uncovered subsets
    
    uncovered_count = n_t
    
    while uncovered_count > 0:
        # Find the most "needed" t-subset (lowest count)
        # To speed up, pick a random uncovered one
        uncovered_idx = [i for i in range(n_t) if counts[i] == 0]
        if not uncovered_idx:
            break
        seed_idx = rng.choice(uncovered_idx)
        seed = all_t_subsets[seed_idx]
        
        # Build a block containing the seed, greedily adding points
        block_set = set(seed)
        remaining = [x for x in range(v) if x not in block_set]
        rng.shuffle(remaining)
        
        # Greedily add points to maximize new coverage
        while len(block_set) < k and remaining:
            best_x = None
            best_gain = -1
            # Sample a subset of candidates for speed
            sample_size = min(20, len(remaining))
            candidates = rng.sample(remaining, sample_size)
            
            current_block = list(block_set)
            for x in candidates:
                gain = 0
                # Compute how many new t-subsets would be covered
                for rest in itertools.combinations(current_block, t - 1):
                    c = tuple(sorted(rest + [x]))
                    if c in t_idx:
                        idx = t_idx[c]
                        if counts[idx] == 0:
                            gain += 10
                        elif counts[idx] == 1:
                            gain += 1
                gain += rng.random() * 0.1  # small perturbation
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block_set.add(best_x)
            remaining.remove(best_x)
        
        if len(block_set) == k:
            block = tuple(sorted(block_set))
            block_indices = _block_to_indices(block, t_idx, t)
            
            # Check if this block adds new coverage
            new_cover = sum(1 for idx in block_indices if counts[idx] == 0)
            if new_cover > 0:
                blocks.append(block)
                block_indices_list.append(block_indices)
                for idx in block_indices:
                    counts[idx] += 1
                uncovered_count -= new_cover
    
    return blocks, block_indices_list, counts


def _local_search(v, k, t, blocks, block_indices_list, counts, n_t, all_t_subsets, t_idx, rng, time_limit):
    """Local search: remove redundant blocks, swap blocks to improve coverage."""
    start_time = time.time()
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    
    T = 30.0
    T_min = 0.5
    cooling = 0.9995
    
    no_improve = 0
    max_no_improve = 500
    
    n_blocks = len(blocks)
    
    while time.time() - start_time < time_limit:
        # 1. Try to remove a redundant block
        removed = False
        order = list(range(n_blocks))
        rng.shuffle(order)
        for i in order:
            if i >= n_blocks:
                continue
            redundant = True
            for idx in block_indices_list[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_indices_list[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_indices_list.pop(i)
                n_blocks -= 1
                
                if n_blocks < best_count:
                    best_count = n_blocks
                    best_blocks = [list(b) for b in blocks]
                    no_improve = 0
                removed = True
                break
        
        if removed:
            continue
        
        # 2. Try to swap a block
        if n_blocks > 0:
            i = rng.randrange(n_blocks)
            old_block = blocks[i]
            old_indices = set(block_indices_list[i])
            
            # Find critical subsets (count == 1)
            critical = [idx for idx in block_indices_list[i] if counts[idx] == 1]
            
            # Points needed to cover critical subsets
            points_needed = set()
            for idx in critical:
                points_needed.update(all_t_subsets[idx])
            
            if len(points_needed) > k:
                continue
            
            new_block_set = set(points_needed)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            # Greedily fill to maximize weighted gain
            for _ in range(k - len(new_block_set)):
                if not cands:
                    break
                best_x = None
                best_gain = -1.0
                current = list(new_block_set)
                for x in cands:
                    gain = 0.0
                    for rest in itertools.combinations(current, t - 1):
                        c = tuple(sorted(rest + [x]))
                        if c in t_idx:
                            idx = t_idx[c]
                            if counts[idx] == 0:
                                gain += 10.0
                            elif counts[idx] == 1:
                                gain += 1.0
                    gain += rng.random() * 0.05
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                new_block_set.add(best_x)
                cands.remove(best_x)
            
            if len(new_block_set) == k:
                new_block = tuple(sorted(new_block_set))
                if new_block == old_block:
                    continue
                
                new_indices = _block_to_indices(new_block, t_idx, t)
                new_indices_set = set(new_indices)
                
                # Check validity: no subset loses its last cover
                valid = True
                for idx in old_indices:
                    if idx not in new_indices_set and counts[idx] <= 1:
                        valid = False
                        break
                
                if valid:
                    # Compute net gain
                    net_gain = 0.0
                    for idx in new_indices_set - old_indices:
                        if counts[idx] == 0:
                            net_gain += 10.0
                        elif counts[idx] == 1:
                            net_gain += 1.0
                    for idx in old_indices - new_indices_set:
                        if counts[idx] == 1:
                            net_gain -= 5.0
                    
                    # Accept with SA probability
                    if net_gain >= 0 or (T > T_min and rng.random() < math.exp(net_gain / T)):
                        # Apply swap
                        for idx in old_indices:
                            counts[idx] -= 1
                        for idx in new_indices:
                            counts[idx] += 1
                        
                        blocks[i] = new_block
                        block_indices_list[i] = new_indices
        
        # 3. Occasionally try to merge two blocks
        if n_blocks >= 2 and rng.random() < 0.15:
            i, j = rng.sample(range(n_blocks), 2)
            crit1 = [idx for idx in block_indices_list[i] if counts[idx] == 1]
            crit2 = [idx for idx in block_indices_list[j] if counts[idx] == 1]
            
            points_needed = set()
            for idx in crit1 + crit2:
                points_needed.update(all_t_subsets[idx])
            
            if len(points_needed) <= k:
                new_block_set = set(points_needed)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                for _ in range(k - len(new_block_set)):
                    if not cands:
                        break
                    best_x = None
                    best_gain = -1.0
                    current = list(new_block_set)
                    for x in cands:
                        gain = 0.0
                        for rest in itertools.combinations(current, t - 1):
                            c = tuple(sorted(rest + [x]))
                            if c in t_idx:
                                idx = t_idx[c]
                                if counts[idx] == 0:
                                    gain += 10.0
                                elif counts[idx] == 1:
                                    gain += 1.0
                        gain += rng.random() * 0.05
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is None:
                        break
                    new_block_set.add(best_x)
                    cands.remove(best_x)
                
                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    new_indices = _block_to_indices(new_block, t_idx, t)
                    new_indices_set = set(new_indices)
                    
                    valid = True
                    for idx in crit1 + crit2:
                        if idx not in new_indices_set:
                            valid = False
                            break
                    
                    if valid:
                        # Remove both, add new
                        for idx in block_indices_list[i]:
                            counts[idx] -= 1
                        for idx in block_indices_list[j]:
                            counts[idx] -= 1
                        for idx in new_indices:
                            counts[idx] += 1
                        
                        if i < j:
                            blocks.pop(j)
                            block_indices_list.pop(j)
                            blocks.pop(i)
                            block_indices_list.pop(i)
                        else:
                            blocks.pop(i)
                            block_indices_list.pop(i)
                            blocks.pop(j)
                            block_indices_list.pop(j)
                        n_blocks -= 2
                        
                        blocks.append(new_block)
                        block_indices_list.append(new_indices)
                        n_blocks += 1
                        
                        if n_blocks < best_count:
                            best_count = n_blocks
                            best_blocks = [list(b) for b in blocks]
                            no_improve = 0
        
        # 4. Restart if stuck
        if no_improve > max_no_improve:
            new_rng = random.Random(rng.randint(0, 10**9))
            new_blocks, new_bil, new_counts = _greedy_construct(v, k, t, new_rng, all_t_subsets, t_idx, n_t)
            if len(new_blocks) < n_blocks:
                blocks = new_blocks
                block_indices_list = new_bil
                counts = new_counts
                n_blocks = len(blocks)
                if n_blocks < best_count:
                    best_count = n_blocks
                    best_blocks = [list(b) for b in blocks]
            no_improve = 0
            T = max(T_min, T * 0.5)
        
        if n_blocks < best_count:
            no_improve = 0
        else:
            no_improve += 1
        
        T = max(T_min, T * cooling)
    
    # Final polish: exhaustive removal
    for _ in range(20):
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            redundant = True
            for idx in block_indices_list[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_indices_list[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_indices_list.pop(i)
                changed = True
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
        if not changed:
            break
    
    return best_blocks


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 140.0  # leave 10s buffer for safety
    
    all_t_subsets, t_idx, n_t = _precompute_masks(v, k, t)
    
    # Build initial covering
    blocks, block_indices_list, counts = _greedy_construct(v, k, t, rng, all_t_subsets, t_idx, n_t)
    
    # Local search
    best_blocks = _local_search(v, k, t, blocks, block_indices_list, counts, n_t, all_t_subsets, t_idx, rng, time_limit - (time.time() - start_time))
    
    return best_blocks
