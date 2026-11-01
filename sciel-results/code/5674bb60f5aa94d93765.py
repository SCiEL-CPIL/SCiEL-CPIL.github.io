"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: NumPy-accelerated greedy + simulated annealing with vectorized delta-energy.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 115.0
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    
    all_k = list(itertools.combinations(range(v), k))
    n_k = len(all_k)
    
    # Precompute coverage indices for each block
    cov_size = len(list(itertools.combinations(range(k), t)))
    block_cov_flat = np.zeros((n_k, cov_size), dtype=np.int32)
    
    for b, ks in enumerate(all_k):
        for j, ts in enumerate(itertools.combinations(ks, t)):
            block_cov_flat[b, j] = t_index[ts]
            
    # t_to_blocks[s] = list of block indices that cover t-subset s
    t_to_blocks = [[] for _ in range(n_t)]
    for b in range(n_k):
        for s in block_cov_flat[b]:
            t_to_blocks[s].append(b)
            
    t_to_blocks_arr = [np.array(blks, dtype=np.int32) for blks in t_to_blocks]
    
    return all_t, t_index, all_k, block_cov_flat, t_to_blocks_arr, n_t, n_k


def _greedy_construct(v, k, t, rng):
    all_t, t_index, all_k, block_cov_flat, t_to_blocks_arr, n_t, n_k = _precompute(v, k, t)
    
    selected = set()
    cover_count = np.zeros(n_t, dtype=np.int32)
    uncovered_count = n_t
    
    while uncovered_count > 0 and _elapsed() < _DEADLINE - 25:
        # Compute gains for all blocks in one vectorized operation
        gains = np.sum(cover_count[block_cov_flat] == 0, axis=1)
        
        best_b = np.argmax(gains)
        best_gain = gains[best_b]
        
        if best_gain == 0:
            break
            
        selected.add(best_b)
        cov_indices = block_cov_flat[best_b]
        for s in cov_indices:
            if cover_count[s] == 0:
                uncovered_count -= 1
            cover_count[s] += 1
            
    return selected, all_k, block_cov_flat, t_to_blocks_arr, n_t, n_k


def _remove_redundant(selected, block_cov_flat, n_t):
    """Remove redundant blocks (those with no unique coverage)."""
    changed = True
    while changed:
        changed = False
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            for s in block_cov_flat[b]:
                cover_count[s] += 1
        
        for b in list(selected):
            cov_indices = block_cov_flat[b]
            if not np.any(cover_count[cov_indices] == 1):
                selected.discard(b)
                changed = True
                break
    return selected


def _simulated_annealing(selected, block_cov_flat, t_to_blocks_arr, n_t, n_k, rng, time_budget):
    """Simulated annealing on block selection with energy = uncovered + penalty * blocks."""
    selected = set(selected)
    
    # Initial energy computation
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        for s in block_cov_flat[b]:
            cover_count[s] += 1
    
    uncovered = int(np.sum(cover_count == 0))
    n_blocks = len(selected)
    penalty = 0.1  # Weight for block count in energy
    current_energy = uncovered + penalty * n_blocks
    best_energy = current_energy
    best_selected = set(selected)
    
    # Temperature schedule
    T_initial = 5.0
    T_final = 0.01
    T_current = T_initial
    
    # Determine number of iterations based on time budget
    # Estimate ~0.1ms per iteration for simple moves
    # We'll adaptively run until time budget
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        # Update temperature (exponential cooling)
        T_current = max(T_final, T_current * 0.9999)
        
        # Choose move type: add, remove, or swap
        move_type = rng.random()
        
        if move_type < 0.3 and len(selected) < n_k:
            # Add a random block
            b = rng.randrange(n_k)
            if b in selected:
                continue
            
            # Compute delta energy for adding b
            cov_b = block_cov_flat[b]
            new_uncovered_change = -int(np.sum(cover_count[cov_b] == 0))
            delta_energy = new_uncovered_change + penalty
            
            # Acceptance criterion
            if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / max(T_current, 1e-10)):
                selected.add(b)
                for s in cov_b:
                    cover_count[s] += 1
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = set(selected)
                    
        elif move_type < 0.6 and len(selected) > 1:
            # Remove a random block
            b = rng.choice(list(selected))
            cov_b = block_cov_flat[b]
            
            # Check if removal would leave uncovered subsets
            unique_coverage = int(np.sum(cover_count[cov_b] == 1))
            if unique_coverage > 0:
                # Would create uncovered subsets
                new_uncovered_change = unique_coverage
                delta_energy = new_uncovered_change - penalty
            else:
                delta_energy = -penalty
            
            # Acceptance criterion
            if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / max(T_current, 1e-10)):
                selected.discard(b)
                for s in cov_b:
                    cover_count[s] -= 1
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = set(selected)
                    
        else:
            # Swap: remove one, add another
            if len(selected) < 2:
                continue
                
            b_remove = rng.choice(list(selected))
            b_add = rng.randrange(n_k)
            if b_add in selected:
                continue
            
            cov_remove = block_cov_flat[b_remove]
            cov_add = block_cov_flat[b_add]
            
            # Compute net change in uncovered
            # Subsets only covered by b_remove and not by b_add become uncovered
            # Subsets covered by neither b_remove nor b_add but covered by b_add become covered
            
            # Temporarily remove b_remove
            for s in cov_remove:
                cover_count[s] -= 1
            
            # Count newly uncovered
            newly_uncovered = int(np.sum(cover_count[cov_remove] == 0))
            
            # Add b_add
            for s in cov_add:
                cover_count[s] += 1
            
            # Count newly covered
            # We need to track which were uncovered before adding b_add
            # Simpler: just compute delta as (newly_uncovered - newly_covered)
            # But we need to know how many were uncovered before this swap
            
            # Let's redo this more carefully
            # First, compute what would be uncovered if we remove b_remove
            temp_cover = cover_count.copy()
            for s in cov_remove:
                temp_cover[s] -= 1
            uncovered_after_remove = int(np.sum(temp_cover == 0))
            
            # Now add b_add
            for s in cov_add:
                temp_cover[s] += 1
            uncovered_after_swap = int(np.sum(temp_cover == 0))
            
            delta_uncovered = uncovered_after_swap - (uncovered - int(np.sum(cover_count == 0)))
            # Actually, current uncovered is what we have now
            current_uncovered = int(np.sum(cover_count == 0))
            delta_uncovered = uncovered_after_swap - current_uncovered
            
            # Block count stays same, so delta_energy = delta_uncovered
            delta_energy = delta_uncovered
            
            # Acceptance criterion
            if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / max(T_current, 1e-10)):
                # Apply the swap
                selected.discard(b_remove)
                for s in cov_remove:
                    cover_count[s] -= 1
                selected.add(b_add)
                for s in cov_add:
                    cover_count[s] += 1
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = set(selected)
            else:
                # Revert the temporary changes
                for s in cov_add:
                    cover_count[s] -= 1
                for s in cov_remove:
                    cover_count[s] += 1
    
    # Return the best solution found
    return best_selected


def _local_search(selected, block_cov_flat, t_to_blocks_arr, n_t, n_k, rng, time_budget):
    """Combined local search: removal + simulated annealing."""
    # First, remove redundant blocks
    selected = _remove_redundant(selected, block_cov_flat, n_t)
    
    # Then run simulated annealing with remaining time
    sa_budget = time_budget * 0.7
    selected = _simulated_annealing(selected, block_cov_flat, t_to_blocks_arr, n_t, n_k, rng, sa_budget)
    
    # Final cleanup: remove any redundant blocks
    selected = _remove_redundant(selected, block_cov_flat, n_t)
    
    return selected


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(42)
    
    # 1. Greedy construction
    selected, all_k, block_cov_flat, t_to_blocks_arr, n_t, n_k = _greedy_construct(v, k, t, rng)
    
    # 2. Local search with simulated annealing
    time_budget = _DEADLINE - 5
    selected = _local_search(selected, block_cov_flat, t_to_blocks_arr, n_t, n_k, rng, time_budget)
    
    # 3. Safety check & repair
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        for s in block_cov_flat[b]:
            cover_count[s] += 1
    
    missing = np.where(cover_count == 0)[0]
    if len(missing) > 0:
        missing_set = set(missing.tolist())
        while missing_set and _elapsed() < _DEADLINE - 2:
            # Find block covering most missing
            gain_counts = {}
            for s in missing_set:
                for b in t_to_blocks_arr[s]:
                    gain_counts[b] = gain_counts.get(b, 0) + 1
            if not gain_counts:
                break
            best_b = max(gain_counts, key=gain_counts.get)
            selected.add(best_b)
            for s in block_cov_flat[best_b]:
                if s in missing_set:
                    missing_set.discard(s)
    
    # Return as list of lists
    result = [list(all_k[b]) for b in selected]
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
