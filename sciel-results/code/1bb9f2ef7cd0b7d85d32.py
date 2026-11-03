"""Randomized local search for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

_CACHE = {}

def _get_precomputed(v, k, t):
    if (v, k, t) in _CACHE:
        return _CACHE[(v, k, t)]
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_indices = [None] * n_blocks
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    for b_idx, block in enumerate(all_blocks):
        idxs = []
        for c in itertools.combinations(block, t):
            idxs.append(subset_to_idx[c])
        block_indices[b_idx] = np.array(idxs, dtype=np.int32)
        
    _CACHE[(v, k, t)] = {
        'all_t_subsets': all_t_subsets,
        'n_subsets': n_subsets,
        'subset_to_idx': subset_to_idx,
        'all_blocks': all_blocks,
        'n_blocks': n_blocks,
        'block_indices': block_indices,
        'block_to_idx': block_to_idx
    }
    return _CACHE[(v, k, t)]

def _greedy_cover(v, k, t, rng, pre):
    """Greedy cover using precomputed data."""
    all_t_subsets = pre['all_t_subsets']
    n_subsets = pre['n_subsets']
    all_blocks = pre['all_blocks']
    block_indices = pre['block_indices']
    block_to_idx = pre['block_to_idx']
    
    uncovered = np.ones(n_subsets, dtype=bool)
    blocks = []
    block_set = set()
    
    while uncovered.any():
        uncovered_indices = np.where(uncovered)[0]
        seed_idx = rng.choice(uncovered_indices)
        seed = all_t_subsets[seed_idx]
        
        block_set_local = set(seed)
        current_block = list(seed)
        
        while len(current_block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block_set_local]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(current_block, t-1):
                    c = tuple(sorted(list(rest) + [x]))
                    if c in pre['subset_to_idx']:
                        idx = pre['subset_to_idx'][c]
                        if uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            current_block.append(best_x)
            block_set_local.add(best_x)
            
        if len(current_block) == k:
            current_block.sort()
            b_tuple = tuple(current_block)
            if b_tuple not in block_set:
                block_set.add(b_tuple)
                blocks.append(b_tuple)
                b_idx = block_to_idx[b_tuple]
                uncovered[block_indices[b_idx]] = False
                    
    return blocks

def _try_remove_redundant(block_idxs, counts, block_indices, rng, n_remove=1):
    """Try to remove up to n_remove redundant blocks. Returns number removed."""
    removed = 0
    attempts = 0
    max_attempts = max(50, len(block_idxs) * 2)
    
    while removed < n_remove and attempts < max_attempts:
        attempts += 1
        if len(block_idxs) == 0:
            break
        i = rng.randrange(len(block_idxs))
        b_idx = block_idxs[i]
        covered = block_indices[b_idx]
        
        if np.all(counts[covered] > 1):
            counts[covered] -= 1
            block_idxs.pop(i)
            removed += 1
            
    return removed

def _build_replacement_block(points_needed, k, v, t, counts, block_indices, block_to_idx, all_t_subsets, subset_to_idx, rng):
    """Build a new block containing points_needed, greedily filling to maximize weighted coverage."""
    if len(points_needed) > k:
        return None
    
    new_block_set = set(points_needed)
    cands = [x for x in range(v) if x not in new_block_set]
    rng.shuffle(cands)
    
    while len(new_block_set) < k:
        best_x = None
        best_gain = -1
        # Take up to 12 candidates to speed up
        trial_cands = cands[:min(12, len(cands))]
        
        for x in trial_cands:
            gain = 0
            for rest in itertools.combinations(new_block_set, t-1):
                c = tuple(sorted(list(rest) + [x]))
                if c in subset_to_idx:
                    idx = subset_to_idx[c]
                    cnt = counts[idx]
                    if cnt == 0:
                        gain += 100
                    elif cnt == 1:
                        gain += 10
                    elif cnt == 2:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None:
            # No good candidate found in trial, pick random
            if cands:
                best_x = cands[0]
            else:
                break
        
        new_block_set.add(best_x)
        cands.remove(best_x)
    
    if len(new_block_set) < k:
        # Fill remaining randomly
        remaining = [x for x in range(v) if x not in new_block_set]
        for x in remaining[:k - len(new_block_set)]:
            new_block_set.add(x)
    
    if len(new_block_set) == k:
        new_block = tuple(sorted(new_block_set))
        return new_block
    return None

def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0
    
    pre = _get_precomputed(v, k, t)
    all_t_subsets = pre['all_t_subsets']
    n_subsets = pre['n_subsets']
    all_blocks = pre['all_blocks']
    n_blocks = pre['n_blocks']
    block_indices = pre['block_indices']
    block_to_idx = pre['block_to_idx']
    subset_to_idx = pre['subset_to_idx']
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, pre)
    block_idxs = [block_to_idx[b] for b in blocks]
    
    counts = np.zeros(n_subsets, dtype=np.int32)
    for b_idx in block_idxs:
        counts[block_indices[b_idx]] += 1
        
    # Initial removal pass
    _try_remove_redundant(block_idxs, counts, block_indices, rng, n_remove=100)
    
    best_block_idxs = list(block_idxs)
    best_count = len(block_idxs)
    
    no_improve = 0
    max_no_improve = 400
    
    while time.time() - start_time < time_limit:
        # Phase 1: Try to remove redundant blocks (multiple)
        removed = _try_remove_redundant(block_idxs, counts, block_indices, rng, n_remove=5)
        
        if removed > 0:
            if len(block_idxs) < best_count:
                best_count = len(block_idxs)
                best_block_idxs = list(block_idxs)
                no_improve = 0
            continue
        
        # Phase 2: Try to swap a block
        if len(block_idxs) > 0:
            i = rng.randrange(len(block_idxs))
            old_b_idx = block_idxs[i]
            old_covered = block_indices[old_b_idx]
            
            critical_mask = (counts[old_covered] == 1)
            
            if np.any(critical_mask):
                critical_indices = old_covered[critical_mask]
                
                points_needed = set()
                for idx in critical_indices:
                    points_needed.update(all_t_subsets[idx])
                
                if len(points_needed) > k:
                    no_improve += 1
                    continue
                
                # Build replacement block
                new_block = _build_replacement_block(
                    points_needed, k, v, t, counts, block_indices, 
                    block_to_idx, all_t_subsets, subset_to_idx, rng
                )
                
                if new_block is not None:
                    new_b_idx = block_to_idx[new_block]
                    new_covered = block_indices[new_b_idx]
                    
                    # Verify validity: new block covers all critical subsets
                    valid = True
                    for idx in critical_indices:
                        if idx not in set(new_covered):
                            valid = False
                            break
                    
                    if valid and new_b_idx != old_b_idx:
                        counts[old_covered] -= 1
                        counts[new_covered] += 1
                        block_idxs[i] = new_b_idx
                        # Try to remove after swap
                        _try_remove_redundant(block_idxs, counts, block_indices, rng, n_remove=3)
            else:
                # No critical subsets: swap to a block that covers low-count subsets
                new_block = _build_replacement_block(
                    set(), k, v, t, counts, block_indices,
                    block_to_idx, all_t_subsets, subset_to_idx, rng
                )
                if new_block is not None:
                    new_b_idx = block_to_idx[new_block]
                    if new_b_idx != old_b_idx:
                        counts[old_covered] -= 1
                        counts[new_covered] += 1
                        block_idxs[i] = new_b_idx
                        _try_remove_redundant(block_idxs, counts, block_indices, rng, n_remove=3)
        
        if len(block_idxs) < best_count:
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > max_no_improve:
            # Restart with fresh greedy
            new_rng = random.Random(rng.randint(0, 10**9))
            new_blocks = _greedy_cover(v, k, t, new_rng, pre)
            new_block_idxs = [block_to_idx[b] for b in new_blocks]
            new_counts = np.zeros(n_subsets, dtype=np.int32)
            for b_idx in new_block_idxs:
                new_counts[block_indices[b_idx]] += 1
            _try_remove_redundant(new_block_idxs, new_counts, block_indices, new_rng, n_remove=100)
            
            if len(new_block_idxs) < len(block_idxs):
                block_idxs = new_block_idxs
                counts = new_counts
                if len(block_idxs) < best_count:
                    best_count = len(block_idxs)
                    best_block_idxs = list(block_idxs)
            no_improve = 0
    
    # Final polish: aggressive removal
    _try_remove_redundant(best_block_idxs, np.zeros(n_subsets, dtype=np.int32), block_indices, rng, n_remove=0)
    # Recompute counts for best and do final removal
    final_counts = np.zeros(n_subsets, dtype=np.int32)
    for b_idx in best_block_idxs:
        final_counts[block_indices[b_idx]] += 1
    _try_remove_redundant(best_block_idxs, final_counts, block_indices, rng, n_remove=200)
    
    return [list(all_blocks[i]) for i in best_block_idxs]
# EVOLVE-BLOCK-END
