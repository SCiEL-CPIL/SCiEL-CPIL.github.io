"""Simulated annealing for ternary covering code K_3(7,1)."""
import numpy as np
import random
import time
import itertools


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius.
    Uses simulated annealing to minimize the number of codewords while ensuring
    full coverage.
    """
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        all_words = list(itertools.product(range(q), repeat=n))
        return all_words

    # Specific optimization for q=3, n=7, radius=1
    total_words = 3 ** n  # 2187
    
    # Precompute all words
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage: for each word i, list of words it covers (itself + distance 1)
    # There are 1 + 7*2 = 15 neighbors
    coverage = np.zeros((total_words, 15), dtype=np.int32)
    for i, w in enumerate(all_words):
        w_list = list(w)
        idx = 0
        coverage[i, idx] = i
        idx += 1
        for pos in range(7):
            original = w_list[pos]
            for val in range(3):
                if val == original:
                    continue
                w_list[pos] = val
                neighbor_idx = word_to_idx[tuple(w_list)]
                coverage[i, idx] = neighbor_idx
                idx += 1
            w_list[pos] = original
        
    # Greedy Initialization
    selected = np.zeros(total_words, dtype=bool)
    covered_count = np.zeros(total_words, dtype=int)
    
    while True:
        uncovered_mask = (covered_count == 0)
        if not uncovered_mask.any():
            break
        
        # Calculate score for each candidate: number of uncovered words it covers
        scores = np.sum(uncovered_mask[coverage], axis=1)
        
        # Mask out already selected
        scores[selected] = -1
        
        best_idx = int(np.argmax(scores))
        if scores[best_idx] <= 0:
            break
            
        selected[best_idx] = True
        covered_count[coverage[best_idx]] += 1
        
    initial_size = int(np.sum(selected))
    
    # Post-processing: try to remove redundant codewords
    changed = True
    while changed:
        changed = False
        selected_indices = np.where(selected)[0]
        for idx in selected_indices:
            if np.all(covered_count[coverage[idx]] > 1):
                selected[idx] = False
                covered_count[coverage[idx]] -= 1
                changed = True
                
    initial_size = int(np.sum(selected))
    
    # Simulated Annealing
    start_temp = 100.0
    cooling_rate = 0.99995
    time_limit = 110  # seconds, leave buffer
    
    start_time = time.time()
    temp = start_temp
    best_energy = initial_size
    best_selected = selected.copy()
    best_covered_count = covered_count.copy()
    
    current_energy = initial_size
    penalty = 1000
    
    rng = np.random.default_rng(seed=42)
    
    # Precompute indices for faster access
    selected_indices = np.where(selected)[0]
    
    while time.time() - start_time < time_limit:
        # Choose move type
        r = rng.random()
        if r < 0.7:
            move_type = 'swap'
        elif r < 0.85:
            move_type = 'remove'
        else:
            move_type = 'add'
            
        if move_type == 'swap':
            if len(selected_indices) == 0:
                continue
            
            # Pick a random selected word to remove
            pos = int(rng.integers(0, len(selected_indices)))
            idx_remove = selected_indices[pos]
            
            # Pick a candidate to add. 
            # Strategy: Pick a random unselected word.
            if len(selected_indices) >= total_words:
                continue
                
            # Efficiently pick a random unselected index
            while True:
                idx_add = int(rng.integers(0, total_words))
                if not selected[idx_add]:
                    break
            
            # Calculate energy change
            cov_remove = coverage[idx_remove]
            cov_add = coverage[idx_add]
            
            # Temporarily remove
            # Count how many words would become uncovered (count==1)
            would_uncover = int(np.sum((covered_count[cov_remove] == 1)))
            
            # Temporarily add
            # Count how many words would become covered (count==0)
            would_cover = int(np.sum((covered_count[cov_add] == 0)))
            
            # Delta size is 0 (swap)
            # Delta penalty: -would_cover * penalty + would_uncover * penalty
            delta_energy = would_uncover * penalty - would_cover * penalty
            
            # Acceptance
            if delta_energy <= 0 or (temp > 1e-6 and rng.random() < np.exp(-delta_energy / temp)):
                # Apply move
                selected[idx_remove] = False
                selected[idx_add] = True
                selected_indices[pos] = idx_add
                
                covered_count[cov_remove] -= 1
                covered_count[cov_add] += 1
                
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = selected.copy()
                    best_covered_count = covered_count.copy()
            # If rejected, state remains unchanged
            
        elif move_type == 'remove':
            if len(selected_indices) == 0:
                continue
                
            pos = int(rng.integers(0, len(selected_indices)))
            idx_remove = selected_indices[pos]
            
            cov_remove = coverage[idx_remove]
            would_uncover = int(np.sum((covered_count[cov_remove] == 1)))
            
            # Delta size is -1
            # Delta penalty: +would_uncover * penalty
            delta_energy = -1 + would_uncover * penalty
            
            if delta_energy <= 0 or (temp > 1e-6 and rng.random() < np.exp(-delta_energy / temp)):
                selected[idx_remove] = False
                selected_indices = np.delete(selected_indices, pos)
                
                covered_count[cov_remove] -= 1
                
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = selected.copy()
                    best_covered_count = covered_count.copy()
                    
        else: # add
            if len(selected_indices) >= total_words:
                continue
                
            # Pick random unselected
            while True:
                idx_add = int(rng.integers(0, total_words))
                if not selected[idx_add]:
                    break
            
            cov_add = coverage[idx_add]
            would_cover = int(np.sum((covered_count[cov_add] == 0)))
            
            # Delta size is +1
            # Delta penalty: -would_cover * penalty
            delta_energy = 1 - would_cover * penalty
            
            if delta_energy <= 0 or (temp > 1e-6 and rng.random() < np.exp(-delta_energy / temp)):
                selected[idx_add] = True
                selected_indices = np.append(selected_indices, idx_add)
                
                covered_count[cov_add] += 1
                
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = selected.copy()
                    best_covered_count = covered_count.copy()
        
        temp *= cooling_rate
        
    # Use best solution found
    final_selected = best_selected
    final_indices = np.where(final_selected)[0]
    
    # Verify and fix coverage if needed (safety net)
    final_covered_count = np.zeros(total_words, dtype=int)
    for idx in final_indices:
        final_covered_count[coverage[idx]] += 1
        
    if np.any(final_covered_count == 0):
        uncovered_words = np.where(final_covered_count == 0)[0]
        for w in uncovered_words:
            if final_covered_count[w] > 0:
                continue
            # Find any codeword that covers w
            for c in range(total_words):
                if final_selected[c]:
                    continue
                if w in coverage[c]:
                    final_selected[c] = True
                    final_covered_count[coverage[c]] += 1
                    break
    
    # Post-processing: try to remove redundant codewords
    changed = True
    while changed:
        changed = False
        final_indices = np.where(final_selected)[0]
        for idx in final_indices:
            if np.all(final_covered_count[coverage[idx]] > 1):
                final_selected[idx] = False
                final_covered_count[coverage[idx]] -= 1
                changed = True
                
    final_indices = np.where(final_selected)[0]
    return [list(all_words[idx]) for idx in final_indices]
