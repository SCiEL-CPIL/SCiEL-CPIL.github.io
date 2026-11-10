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
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words
    
    # Specific optimization for q=3, n=7, radius=1
    total_words = 3 ** n  # 2187
    
    # Precompute all words and their indices
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage: for each word i, list of words it covers (itself + distance 1)
    coverage = [None] * total_words
    for i, w in enumerate(all_words):
        covered = [i]
        w_list = list(w)
        for pos in range(7):
            original = w_list[pos]
            for val in range(3):
                if val == original:
                    continue
                w_list[pos] = val
                neighbor_idx = word_to_idx[tuple(w_list)]
                covered.append(neighbor_idx)
            w_list[pos] = original
        coverage[i] = covered
    
    # Convert coverage to numpy arrays for faster indexing
    coverage_arr = np.array(coverage, dtype=np.int32)
    
    # Also precompute reverse coverage: for each word, which codewords cover it
    # This is the same as coverage but we can use it differently
    # Actually coverage_arr[i] gives words covered by codeword i
    # We need: for each word j, which codewords i have j in coverage[i]
    # That's the transpose. Let's build it.
    reverse_coverage = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in coverage[i]:
            reverse_coverage[j].append(i)
    reverse_coverage_arr = np.array(reverse_coverage, dtype=np.int32)
    
    # Greedy initialization: build a feasible solution
    selected = np.zeros(total_words, dtype=bool)
    covered_count = np.zeros(total_words, dtype=int)
    
    # Greedy: repeatedly pick codeword covering most uncovered words
    for _ in range(200):
        uncovered_mask = (covered_count == 0)
        if not uncovered_mask.any():
            break
        # For each candidate, count how many uncovered words it covers
        # Use vectorized approach: for each candidate c, sum of uncovered_mask[coverage[c]]
        # This is O(N^2) but N=2187, so ~4.8M ops per step, acceptable for 200 steps
        best_idx = -1
        best_count = 0
        for c in range(total_words):
            if selected[c]:
                continue
            cnt = np.sum(uncovered_mask[coverage_arr[c]])
            if cnt > best_count:
                best_count = cnt
                best_idx = c
        if best_idx == -1:
            break
        selected[best_idx] = True
        covered_count[coverage_arr[best_idx]] += 1
    
    # Now we have a feasible solution. Let's record its size.
    initial_size = int(np.sum(selected))
    
    # Post-processing: try to remove redundant codewords
    # A codeword is redundant if removing it doesn't uncover any word
    selected_indices = np.where(selected)[0]
    for idx in selected_indices:
        cov = coverage_arr[idx]
        if np.all(covered_count[cov] > 1):
            selected[idx] = False
            covered_count[cov] -= 1
    
    initial_size = int(np.sum(selected))
    
    # Simulated Annealing
    start_temp = 30.0
    cooling_rate = 0.99985
    time_limit = 105  # seconds, leave buffer
    
    start_time = time.time()
    temp = start_temp
    best_energy = initial_size
    best_selected = selected.copy()
    best_covered_count = covered_count.copy()
    
    current_energy = initial_size
    
    # Precompute for faster random access
    rng = np.random.default_rng(seed=42)
    
    # Track which indices are selected for fast removal
    selected_list = np.where(selected)[0]
    
    while time.time() - start_time < time_limit:
        # Choose move type with weighted probability
        # Bias towards swaps and removals to reduce size
        r = rng.random()
        if r < 0.45:
            move_type = 'swap'
        elif r < 0.75:
            move_type = 'remove'
        else:
            move_type = 'add'
        
        if move_type == 'add':
            # Pick a random unselected word, or target an uncovered word
            if len(selected_list) >= total_words:
                continue
            
            # Try to add a word that covers many uncovered words
            uncovered_mask = (covered_count == 0)
            if uncovered_mask.any():
                # Find best candidate among unselected words
                best_idx = -1
                best_count = 0
                # Sample some candidates for speed
                sample_size = min(100, total_words - len(selected_list))
                candidates = rng.choice(total_words, size=sample_size, replace=False)
                for c in candidates:
                    if selected[c]:
                        continue
                    cnt = np.sum(uncovered_mask[coverage_arr[c]])
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = c
                if best_idx == -1:
                    # Fall back to random
                    while True:
                        idx = int(rng.integers(0, total_words))
                        if not selected[idx]:
                            break
                else:
                    idx = best_idx
            else:
                # No uncovered words, pick random unselected
                while True:
                    idx = int(rng.integers(0, total_words))
                    if not selected[idx]:
                        break
            
            # Apply addition
            cov = coverage_arr[idx]
            new_uncovered = int(np.sum((covered_count[cov] == 0)))
            delta_energy = 1 - new_uncovered * 1000
            covered_count[cov] += 1
            
            new_energy = current_energy + delta_energy
            if new_energy < current_energy or rng.random() < np.exp(-max(0, new_energy - current_energy) / temp):
                selected[idx] = True
                selected_list = np.append(selected_list, idx)
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = selected.copy()
                    best_covered_count = covered_count.copy()
            else:
                covered_count[cov] -= 1
        
        elif move_type == 'remove':
            if len(selected_list) == 0:
                continue
            pos = int(rng.integers(0, len(selected_list)))
            idx = selected_list[pos]
            
            # Check if removal causes uncovered words
            cov = coverage_arr[idx]
            would_uncover = int(np.sum((covered_count[cov] == 1)))
            delta_energy = -1 + would_uncover * 1000
            covered_count[cov] -= 1
            
            new_energy = current_energy + delta_energy
            if new_energy < current_energy or rng.random() < np.exp(-max(0, new_energy - current_energy) / temp):
                selected[idx] = False
                selected_list = np.delete(selected_list, pos)
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = selected.copy()
                    best_covered_count = covered_count.copy()
            else:
                covered_count[cov] += 1
        
        else:  # swap
            if len(selected_list) == 0:
                continue
            pos = int(rng.integers(0, len(selected_list)))
            idx_remove = selected_list[pos]
            
            # Pick random unselected word to add
            while True:
                idx_add = int(rng.integers(0, total_words))
                if not selected[idx_add]:
                    break
            
            cov_remove = coverage_arr[idx_remove]
            cov_add = coverage_arr[idx_add]
            
            # Apply removal
            would_uncover = int(np.sum((covered_count[cov_remove] == 1)))
            covered_count[cov_remove] -= 1
            
            # Apply addition
            new_uncovered = int(np.sum((covered_count[cov_add] == 0)))
            covered_count[cov_add] += 1
            
            delta_energy = -1 + would_uncover * 1000 + 1 - new_uncovered * 1000
            new_energy = current_energy + delta_energy
            
            if new_energy < current_energy or rng.random() < np.exp(-max(0, new_energy - current_energy) / temp):
                selected[idx_remove] = False
                selected[idx_add] = True
                selected_list[pos] = idx_add
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = selected.copy()
                    best_covered_count = covered_count.copy()
            else:
                covered_count[cov_remove] += 1
                covered_count[cov_add] -= 1
        
        temp *= cooling_rate
    
    # Use best solution found
    final_selected = best_selected
    final_indices = np.where(final_selected)[0]
    
    # Verify and fix coverage if needed
    final_covered_count = np.zeros(total_words, dtype=int)
    for idx in final_indices:
        final_covered_count[coverage_arr[idx]] += 1
    
    if np.any(final_covered_count == 0):
        uncovered_words = np.where(final_covered_count == 0)[0]
        for w in uncovered_words:
            if final_covered_count[w] > 0:
                continue
            # Find any codeword that covers w
            for c in reverse_coverage[w]:
                if final_selected[c]:
                    continue
                final_selected[c] = True
                final_covered_count[coverage_arr[c]] += 1
                if final_covered_count[w] > 0:
                    break
    
    # Post-processing: try to remove redundant codewords
    final_indices = np.where(final_selected)[0]
    final_covered_count = np.zeros(total_words, dtype=int)
    for idx in final_indices:
        final_covered_count[coverage_arr[idx]] += 1
    
    for idx in final_indices:
        cov = coverage_arr[idx]
        if np.all(final_covered_count[cov] > 1):
            final_selected[idx] = False
            final_covered_count[cov] -= 1
    
    final_indices = np.where(final_selected)[0]
    return [list(all_words[idx]) for idx in final_indices]
