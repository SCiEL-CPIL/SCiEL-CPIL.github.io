import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
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
    
    # Inverse frequency for weighting
    inv_freq = 1.0 / (freq + 1e-9)
    
    uncovered = np.ones(num_t_subsets, dtype=bool)
    selected_indices = []
    
    # Phase 1: Weighted greedy with better tie-breaking
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        w = np.where(uncovered, inv_freq, 0.0)
        gains = np.sum(w[block_indices], axis=1)
        
        # Among blocks with max gain, prefer those with lower overlap with already-selected blocks
        best_gain = np.max(gains)
        candidates = np.where(np.isclose(gains, best_gain, rtol=1e-6))[0]
        
        if len(candidates) == 1:
            best_idx = candidates[0]
        else:
            # Prefer blocks that cover more "rare" uncovered subsets
            # and have less overlap with already selected
            if selected_indices:
                sel_arr = np.array(selected_indices)
                # Count overlap: how many t-subsets of candidate are already covered
                cand_tsubs = block_indices[candidates]  # shape (n_cand, 35)
                # For each candidate, count how many of its t-subsets are NOT uncovered
                overlap = np.sum(~uncovered[cand_tsubs], axis=1)
                # Also weight by inverse frequency of uncovered subsets it covers
                unc_weight = np.sum(w[cand_tsubs], axis=1)
                score = unc_weight - 0.1 * overlap
                best_local = np.argmax(score)
            else:
                best_local = 0
            best_idx = candidates[best_local]
        
        if best_gain == 0:
            break
            
        selected_indices.append(best_idx)
        uncovered[block_indices[best_idx]] = False
    
    # Phase 2: Aggressive redundancy removal
    if time.time() < deadline - 20:
        selected_indices = _remove_redundant(selected_indices, block_indices, num_t_subsets, deadline)
    
    # Phase 3: Local search - try to swap blocks to reduce count
    if time.time() < deadline - 15:
        selected_indices = _local_search_improved(
            selected_indices, blocks, block_indices, num_t_subsets, deadline)
    
    # Rebuild selected_blocks from indices
    selected_blocks = [list(blocks[idx]) for idx in selected_indices]
    
    return selected_blocks

def _remove_redundant(selected_indices, block_indices, num_t_subsets, deadline):
    """Repeatedly remove any block that doesn't uniquely cover any t-subset."""
    coverage = np.zeros(num_t_subsets, dtype=np.int32)
    for idx in selected_indices:
        coverage[block_indices[idx]] += 1
    
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        new_indices = []
        for idx in selected_indices:
            tsubs = block_indices[idx]
            if np.any(coverage[tsubs] == 1):
                new_indices.append(idx)
            else:
                coverage[tsubs] -= 1
                changed = True
        selected_indices = new_indices
    
    return selected_indices

def _local_search_improved(selected_indices, blocks, block_indices, num_t_subsets, deadline):
    """Improved local search: try to remove blocks by finding replacements."""
    coverage = np.zeros(num_t_subsets, dtype=np.int32)
    for idx in selected_indices:
        coverage[block_indices[idx]] += 1
    
    n_blocks_total = len(blocks)
    rng = np.random.default_rng(42)
    
    # Try to remove blocks one at a time
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        n = len(selected_indices)
        
        for i in range(n):
            if time.time() > deadline - 5:
                break
            
            current_idx = selected_indices[i]
            current_tsubs = block_indices[current_idx]
            
            # Find t-subsets that would become uncovered if we remove this block
            critical_mask = coverage[current_tsubs] == 1
            n_critical = int(np.sum(critical_mask))
            
            if n_critical == 0:
                # This block is fully redundant - just remove it
                coverage[current_tsubs] -= 1
                selected_indices.pop(i)
                improved = True
                continue
            
            if n_critical > 10:
                # Too many critical subsets to cover with one block
                continue
            
            # Try to find a replacement block that covers all critical t-subsets
            n_samples = min(2000, n_blocks_total)
            candidates = rng.choice(n_blocks_total, size=n_samples, replace=False)
            
            best_replacement = -1
            best_score = -1
            
            for cand_idx in candidates:
                cand_tsubs = block_indices[cand_idx]
                
                # Check if candidate covers all critical t-subsets
                critical_tsub_idxs = current_tsubs[critical_mask]
                
                if len(critical_tsub_idxs) == 0:
                    covers_all = True
                else:
                    cand_set = set(cand_tsubs.tolist())
                    covers_all = all(t in cand_set for t in critical_tsub_idxs)
                
                if not covers_all:
                    continue
                
                # Score: how many additional redundant coverages this creates
                extra_redundant = 0
                for tsub_idx in cand_tsubs:
                    if tsub_idx not in critical_tsub_idxs and coverage[tsub_idx] == 1:
                        extra_redundant += 1
                
                if extra_redundant > best_score:
                    best_score = extra_redundant
                    best_replacement = cand_idx
            
            if best_replacement >= 0:
                # Perform the swap
                coverage[block_indices[current_idx]] -= 1
                coverage[block_indices[best_replacement]] += 1
                selected_indices[i] = best_replacement
                improved = True
    
    # Final redundancy removal
    coverage = np.zeros(num_t_subsets, dtype=np.int32)
    for idx in selected_indices:
        coverage[block_indices[idx]] += 1
    
    changed = True
    while changed and time.time() < deadline - 3:
        changed = False
        new_indices = []
        for idx in selected_indices:
            tsubs = block_indices[idx]
            if np.any(coverage[tsubs] == 1):
                new_indices.append(idx)
            else:
                coverage[tsubs] -= 1
                changed = True
        selected_indices = new_indices
    
    return selected_indices
