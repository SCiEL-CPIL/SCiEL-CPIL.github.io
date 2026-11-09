"""
Strategy: Multi-Restart Randomized Greedy Set Cover with Vectorized Coverage
and Aggressive Local Search (removal + swap).

Key improvements over the deterministic greedy:
1. Randomized greedy: instead of always picking the max-gain block, we sample
   from a distribution weighted by gain^alpha to explore diverse solutions.
2. Multiple restarts: we run several randomized greedy passes and keep the best.
3. Aggressive local search: after each greedy pass, we do removal passes and
   swap-based local search to minimize block count.
4. We track the best solution across all restarts and return it.

The coverage matrix is precomputed once and reused across all restarts.
"""

import time
import numpy as np
import itertools
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    
    # Bitmasks for t-subsets
    t_masks = np.zeros(num_t, dtype=np.int64)
    for i, sub in enumerate(t_subsets):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_masks[i] = mask
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # Bitmasks for k-blocks
    k_masks = np.zeros(num_k, dtype=np.int64)
    for i, block in enumerate(k_blocks):
        mask = 0
        for p in block:
            mask |= (1 << p)
        k_masks[i] = mask
    
    # Precompute coverage matrix in chunks
    chunk_size = 1000
    cover_list = []
    for start in range(0, num_k, chunk_size):
        end = min(start + chunk_size, num_k)
        k_chunk = k_masks[start:end]
        check = (t_masks[None, :] & ~k_chunk[:, None]) == 0
        cover_list.append(check.astype(np.uint8))
    
    # Build a fast lookup: for each block index, get its cover row
    def get_cover_row(idx):
        chunk_idx = idx // chunk_size
        local_idx = idx % chunk_size
        return cover_list[chunk_idx][local_idx]
    
    # Helper: compute gains for all blocks given an uncovered mask
    def compute_gains(uncovered):
        gains = np.zeros(num_k, dtype=np.int32)
        for chunk_idx, cover_chunk in enumerate(cover_list):
            gains[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size] = cover_chunk.dot(uncovered)
        return gains
    
    # Helper: removal pass
    def perform_removal_pass(indices, counts):
        changed = False
        new_indices = []
        for idx in indices:
            cover_row = get_cover_row(idx)
            critical_mask = (cover_row == 1) & (counts == 1)
            if not np.any(critical_mask):
                counts -= cover_row
                changed = True
            else:
                new_indices.append(idx)
        return new_indices, counts, changed
    
    # Helper: swap-based local search
    def local_search_swap(indices, counts, time_budget):
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            current_count = len(indices)
            
            # Try to find a beneficial swap
            # For each selected block, find its critical subsets
            # Then look for an unselected block that covers all critical subsets
            # and leads to net reduction after removal pass
            
            indices_list = list(indices)
            for i_idx in range(len(indices_list)):
                if time.time() > time_budget:
                    break
                    
                idx = indices_list[i_idx]
                cover_row = get_cover_row(idx)
                
                critical_mask = (cover_row == 1) & (counts == 1)
                if not np.any(critical_mask):
                    continue
                
                crit_indices = np.where(critical_mask)[0]
                if len(crit_indices) == 0:
                    continue
                
                # Find candidate blocks that cover all critical subsets
                found_swap = False
                for c_idx, cover_chunk in enumerate(cover_list):
                    if time.time() > time_budget:
                        break
                    
                    sums = cover_chunk[:, crit_indices].sum(axis=1)
                    candidates = np.where(sums == len(crit_indices))[0]
                    
                    for local_cand in candidates:
                        cand_global_idx = c_idx * chunk_size + local_cand
                        
                        # Skip if already selected
                        if cand_global_idx in indices:
                            continue
                        
                        # Simulate swap
                        sim_counts = counts.copy()
                        sim_counts -= cover_row
                        sim_counts += get_cover_row(cand_global_idx)
                        
                        # Check no uncovered subsets
                        if np.any(sim_counts == 0):
                            continue
                        
                        sim_indices = [i for i in indices if i != idx]
                        sim_indices.append(cand_global_idx)
                        
                        sim_indices, sim_counts, _ = perform_removal_pass(sim_indices, sim_counts)
                        
                        if len(sim_indices) < current_count:
                            indices = sim_indices
                            counts = sim_counts
                            current_count = len(sim_indices)
                            improved = True
                            found_swap = True
                            break
                    
                    if found_swap:
                        break
                
                if found_swap:
                    break
        
        return indices, counts
    
    # Randomized greedy
    def randomized_greedy(rng, alpha=1.0):
        uncovered = np.ones(num_t, dtype=np.uint8)
        selected = []
        
        while np.sum(uncovered) > 0:
            if time.time() > deadline - 15:
                break
            
            gains = compute_gains(uncovered)
            
            if np.max(gains) == 0:
                break
            
            # Sample from distribution proportional to gains^alpha
            # Use a temperature-like approach: weight = gains^alpha
            weights = gains.astype(np.float64)
            # Avoid zero weights
            weights[weights < 1] = 0
            weights = np.power(weights, alpha)
            
            total_weight = np.sum(weights)
            if total_weight == 0:
                # Fallback to max gain
                best_idx = int(np.argmax(gains))
            else:
                # Sample
                probs = weights / total_weight
                # Use numpy random choice
                best_idx = rng.choice(num_k, p=probs)
            
            selected.append(best_idx)
            cover_row = get_cover_row(best_idx)
            uncovered[cover_row == 1] = 0
        
        return selected
    
    # Main loop: multiple restarts with different random seeds and alpha values
    best_indices = None
    best_count = float('inf')
    
    # Deterministic greedy first (alpha -> infinity, i.e., always max)
    def deterministic_greedy():
        uncovered = np.ones(num_t, dtype=np.uint8)
        selected = []
        
        while np.sum(uncovered) > 0:
            if time.time() > deadline - 15:
                break
            
            gains = compute_gains(uncovered)
            if np.max(gains) == 0:
                break
            
            best_idx = int(np.argmax(gains))
            selected.append(best_idx)
            cover_row = get_cover_row(best_idx)
            uncovered[cover_row == 1] = 0
        
        return selected
    
    # Run deterministic greedy
    det_selected = deterministic_greedy()
    if len(det_selected) > 0:
        # Local search
        counts = np.zeros(num_t, dtype=np.int32)
        for idx in det_selected:
            counts += get_cover_row(idx)
        
        changed = True
        while changed and time.time() < deadline - 10:
            det_selected, counts, changed = perform_removal_pass(det_selected, counts)
        
        if time.time() < deadline - 10:
            det_selected, counts = local_search_swap(det_selected, counts, deadline - 5)
        
        if len(det_selected) < best_count:
            best_count = len(det_selected)
            best_indices = list(det_selected)
    
    # Now run randomized greedy with different alpha values and seeds
    # We have limited time, so we do a few restarts
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        # Estimate time per greedy pass
        # Do a quick timing
        t0 = time.time()
        _ = deterministic_greedy()
        greedy_time = time.time() - t0
        
        # How many randomized passes can we afford?
        # Leave 10s for final local search
        available = remaining_time - 10
        if greedy_time > 0:
            num_passes = max(1, int(available / (greedy_time * 1.5)))
        else:
            num_passes = 1
        
        # Cap to avoid too many
        num_passes = min(num_passes, 20)
        
        alphas = [0.5, 1.0, 2.0, 3.0]
        
        for restart in range(num_passes):
            if time.time() > deadline - 15:
                break
            
            alpha = alphas[restart % len(alphas)]
            rng = random.Random(restart * 1000 + 42)
            
            sel = randomized_greedy(rng, alpha=alpha)
            if len(sel) == 0:
                continue
            
            # Local search
            counts = np.zeros(num_t, dtype=np.int32)
            for idx in sel:
                counts += get_cover_row(idx)
            
            changed = True
            while changed and time.time() < deadline - 10:
                sel, counts, changed = perform_removal_pass(sel, counts)
            
            if time.time() < deadline - 10:
                sel, counts = local_search_swap(sel, counts, deadline - 5)
            
            if len(sel) < best_count:
                best_count = len(sel)
                best_indices = list(sel)
    
    # Final: ensure we have a valid solution
    if best_indices is None or len(best_indices) == 0:
        # Fallback: deterministic greedy without local search
        uncovered = np.ones(num_t, dtype=np.uint8)
        selected = []
        while np.sum(uncovered) > 0:
            gains = compute_gains(uncovered)
            if np.max(gains) == 0:
                break
            best_idx = int(np.argmax(gains))
            selected.append(best_idx)
            cover_row = get_cover_row(best_idx)
            uncovered[cover_row == 1] = 0
        best_indices = selected
    
    # Final validation
    counts = np.zeros(num_t, dtype=np.int32)
    for idx in best_indices:
        counts += get_cover_row(idx)
    
    if np.any(counts == 0):
        # Should not happen, but fallback
        uncovered = np.ones(num_t, dtype=np.uint8)
        selected = []
        while np.sum(uncovered) > 0:
            gains = compute_gains(uncovered)
            if np.max(gains) == 0:
                break
            best_idx = int(np.argmax(gains))
            selected.append(best_idx)
            cover_row = get_cover_row(best_idx)
            uncovered[cover_row == 1] = 0
        best_indices = selected
    
    # Ensure no duplicate blocks
    seen = set()
    unique_indices = []
    for idx in best_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    best_indices = unique_indices
    
    return [list(k_blocks[i]) for i in best_indices]
