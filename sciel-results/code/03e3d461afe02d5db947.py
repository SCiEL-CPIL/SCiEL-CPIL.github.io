import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks)
    blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks)
    
    # Precompute block_indices: shape (num_blocks, t_subsets_per_block)
    t_subsets_per_block = 35  # C(7,3) = 35
    block_indices = np.empty((num_blocks, t_subsets_per_block), dtype=np.int32)
    for i, block in enumerate(blocks):
        for j, t_sub in enumerate(itertools.combinations(block, t)):
            block_indices[i, j] = t_subset_to_idx[t_sub]
    
    # Precompute frequency of each t-subset
    freq = np.bincount(block_indices.flatten(), minlength=num_t_subsets)
    
    # Inverse frequency for weighting (smooth surrogate for rarity)
    inv_freq = 1.0 / (freq + 1e-9)
    
    def greedy_cover(seed, time_budget):
        """Weighted greedy with ratio-based scoring (newly covered / cost)."""
        rng = random.Random(seed)
        uncovered = np.ones(num_t_subsets, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            if time.time() > time_budget:
                break
                
            # Ratio-based gain: newly covered triples / block size (k)
            # Since all blocks have same size k, this is equivalent to absolute gain
            # but we add a small penalty for overlap with already selected blocks
            w = np.where(uncovered, inv_freq, 0.0)
            gains = np.sum(w[block_indices], axis=1)
            
            # Add small random perturbation to break ties
            gains += rng.uniform(0, 0.01, size=num_blocks)
            
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]
            
            if best_gain < 1e-6:  # No meaningful gain
                break
                
            selected.append(best_idx)
            uncovered[block_indices[best_idx]] = False
        
        return selected
    
    def complete_cover(selected, deadline):
        """Ensure full coverage by adding blocks as needed."""
        # Compute coverage
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_indices[b_idx]] += 1
        uncovered_mask = cover_count == 0
        
        while np.any(uncovered_mask):
            if time.time() > deadline:
                break
            # Find block covering most uncovered triples
            w = np.where(uncovered_mask, 1.0, 0.0)
            gains = np.sum(w[block_indices], axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected.append(best_idx)
            cover_count[block_indices[best_idx]] += 1
            uncovered_mask = cover_count == 0
        
        return selected
    
    def reduce_cover(selected):
        """Remove redundant blocks (those not uniquely covering any t-subset)."""
        if not selected:
            return selected
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_indices[b_idx]] += 1
        
        active = [True] * len(selected)
        changed = True
        while changed:
            changed = False
            for i in range(len(selected)):
                if not active[i]:
                    continue
                b_idx = selected[i]
                tids = block_indices[b_idx]
                if np.min(cover_count[tids]) > 1:
                    active[i] = False
                    cover_count[tids] -= 1
                    changed = True
        
        return [selected[i] for i in range(len(selected)) if active[i]]
    
    def local_search(selected, deadline):
        """Try to reduce block count via swap-based local search."""
        if len(selected) < 2:
            return selected
        
        selected = list(selected)
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_indices[b_idx]] += 1
        
        rng = random.Random(42)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            order = list(range(len(selected)))
            rng.shuffle(order)
            for i in order:
                if time.time() > deadline - 5:
                    break
                b_idx = selected[i]
                # Temporarily remove
                cover_count[block_indices[b_idx]] -= 1
                uncovered = cover_count == 0
                if not np.any(uncovered):
                    # Block was redundant
                    selected.pop(i)
                    improved = True
                    break
                # Try to find replacement that covers all uncovered
                unc_idx = np.where(uncovered)[0]
                if len(unc_idx) <= 35:
                    # Find blocks covering all uncovered triples
                    candidate_sets = [set(np.where(block_indices[:, j] == unc_idx[j])[0].tolist()) 
                                      for j in range(len(unc_idx))]
                    candidates = set.intersection(*candidate_sets) if candidate_sets else set()
                    already = set(selected)
                    candidates -= already
                    if candidates:
                        new_b = next(iter(candidates))
                        selected.pop(i)
                        selected.append(new_b)
                        cover_count[block_indices[new_b]] += 1
                        improved = True
                        break
                    else:
                        cover_count[block_indices[b_idx]] += 1
                else:
                    cover_count[block_indices[b_idx]] += 1
        
        return reduce_cover(selected)
    
    best_cover = None
    
    # Phase 1: Multiple greedy restarts with different seeds
    seed = 0
    while time.time() < deadline - 30:
        budget = deadline - 25
        cover = greedy_cover(seed, budget)
        cover = complete_cover(cover, deadline - 15)
        cover = reduce_cover(cover)
        
        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover
        
        seed += 1
    
    if best_cover is None:
        best_cover = greedy_cover(0, deadline - 10)
        best_cover = complete_cover(best_cover, deadline - 5)
        best_cover = reduce_cover(best_cover)
    
    # Phase 2: Local search on the best cover
    best_cover = local_search(best_cover, deadline - 3)
    
    # Phase 3: If time remains, try more restarts
    while time.time() < deadline - 3:
        seed = random.randint(1000, 100000)
        budget = deadline - 5
        cover = greedy_cover(seed, budget)
        cover = complete_cover(cover, deadline - 3)
        cover = reduce_cover(cover)
        if len(cover) < len(best_cover):
            best_cover = cover
    
    # Final safety: ensure validity
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in best_cover:
        cover_count[block_indices[b_idx]] += 1
    if (cover_count == 0).any():
        # Add missing blocks
        uncovered_mask = cover_count == 0
        while np.any(uncovered_mask):
            w = np.where(uncovered_mask, 1.0, 0.0)
            gains = np.sum(w[block_indices], axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            best_cover.append(best_idx)
            cover_count[block_indices[best_idx]] += 1
            uncovered_mask = cover_count == 0
    
    # Convert indices to actual blocks
    result = [list(blocks[b_idx]) for b_idx in best_cover]
    
    # Ensure blocks are distinct
    seen = set()
    unique_result = []
    for block in result:
        key = tuple(block)
        if key not in seen:
            seen.add(key)
            unique_result.append(block)
    
    return unique_result
