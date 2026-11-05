"""Covering design C(19,8,5) - numpy-accelerated greedy with random restarts."""
import time
import random
import itertools
import numpy as np

_DEADLINE = 145  # seconds, safety margin before 150s kill


def construct(v, k, t):
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_combos = list(itertools.combinations(range(v), t))
    n_t = len(all_t_combos)
    
    # Precompute all k-subsets
    all_k_combos = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_combos)
    
    # Build mask -> index for t-subsets
    mask_to_idx = {}
    for i, combo in enumerate(all_t_combos):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        mask_to_idx[mask] = i
    
    # For each block, get covered t-subset indices (each block covers C(k,t) t-subsets)
    coverage_matrix = np.zeros((n_blocks, 56), dtype=np.int32)  # C(8,5)=56
    for bi, combo in enumerate(all_k_combos):
        covered = []
        for sub in itertools.combinations(combo, t):
            mask = 0
            for x in sub:
                mask |= (1 << x)
            covered.append(mask_to_idx[mask])
        coverage_matrix[bi, :] = covered
    
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=np.uint8)
        blocks_used = []
        
        while True:
            if time.time() - start_time > _DEADLINE:
                return None
            
            remaining = uncovered.sum()
            if remaining == 0:
                break
            
            # counts[bi] = number of uncovered t-subsets covered by block bi
            counts = uncovered[coverage_matrix].sum(axis=1)
            
            best_count = counts.max()
            if best_count == 0:
                break
            
            # Get all blocks with max count, pick one randomly for diversification
            candidates = np.where(counts == best_count)[0]
            best_block = candidates[rng.randint(0, len(candidates) - 1)]
            
            blocks_used.append(all_k_combos[best_block])
            uncovered[coverage_matrix[best_block, :]] = 0
        
        return [list(b) for b in blocks_used]
    
    # Try multiple random seeds to find the best covering
    best_blocks = None
    best_len = float('inf')
    
    for seed in range(200):
        if time.time() - start_time > _DEADLINE:
            break
        
        result = greedy_cover(seed)
        if result is not None:
            if len(result) < best_len:
                best_len = len(result)
                best_blocks = result
    
    if best_blocks is None:
        return []
    
    # Deduplicate and ensure valid format
    seen = set()
    final = []
    for b in best_blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            final.append(sorted(b))
    
    return final
