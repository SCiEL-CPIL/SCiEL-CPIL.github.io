import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Precompute balls as indices
    weights = [3 ** (n - 1 - i) for i in range(n)]

    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Build inverse lists: for each word v, which codewords cover it
    # This is just balls[v] due to symmetry, but let's be explicit for clarity if needed
    # Actually, balls[v] IS the list of codewords covering v.
    
    # 1. Greedy Initialization
    # Use a slightly smarter greedy: prioritize words that cover more uncovered neighbors.
    # To speed up, we can use a heap or just simple iteration if fast enough.
    # 6561 iterations * O(total) check is ~ 43M ops, which is fine in numpy if vectorized.
    
    cov = np.zeros(total, dtype=np.int32)
    selected = []
    uncovered = np.ones(total, dtype=bool)
    
    # Precompute the list of balls for faster access? No, balls is already an array.
    
    # Simple greedy loop
    # We can optimize by maintaining a list of active candidates?
    # For now, standard greedy is robust and fast enough.
    
    # Optimization: Instead of checking all 6561 every time, we can just run it.
    # It typically takes < 0.5s.
    
    while np.any(uncovered) and time.time() < deadline - 30.0:
        # Count uncovered neighbors for each word
        # uncovered[balls] gives a (total, ball_size) boolean array
        # Sum along axis 1 gives count of uncovered neighbors
        counts = np.sum(uncovered[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        # Pick smallest index with max count for determinism
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        
        selected.append(best_idx)
        cov[balls[best_idx]] += 1
        uncovered = (cov == 0)
        
    # If greedy timed out, we have a partial cover.
    # Ensure we finish it if close to deadline, but usually it's fast.
    if np.any(uncovered) and time.time() < deadline - 10.0:
        # Force completion
        for w in np.where(uncovered)[0]:
             if cov[w] == 0:
                 # Find any codeword covering w
                 # w is in balls[w], so w covers itself.
                 # But we might have removed it? No, we only add.
                 # If w is uncovered, no selected code covers it.
                 # So we must add w or a neighbor.
                 # Adding w is always valid.
                 if w not in selected:
                     selected.append(w)
                     cov[balls[w]] += 1
                     uncovered = (cov == 0)

    # Convert selected to a set for O(1) lookup
    selected_set = set(selected)
    
    # 2. Local Search: Remove and Swap
    # We maintain 'cov' and 'selected_set'.
    
    def add_code(c):
        cov[balls[c]] += 1
        selected_set.add(c)
        
    def remove_code(c):
        cov[balls[c]] -= 1
        selected_set.discard(c)
        
    improved = True
    stall_count = 0
    max_stalls = 50
    
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # --- Phase 1: Remove redundant codewords ---
        # Iterate over a copy of the set to allow modification
        for c in list(selected_set):
            if np.min(cov[balls[c]]) > 1:
                remove_code(c)
                improved = True
                
        # --- Phase 2: Swap ---
        # Try to swap each codeword with a better one
        # To avoid O(N^2) in Python loops which is slow, we limit the number of swaps per pass
        # or use a smarter candidate generation.
        
        # Heuristic: Pick a random subset of codewords to try swapping, or just all of them.
        # Since N is ~500, iterating all is 500 iterations.
        # For each, finding valid swaps involves set intersection.
        
        # Optimization: Only try to swap if the codeword has unique coverage.
        # If a codeword has no unique coverage, it's redundant (handled in Phase 1).
        
        swaps_made = 0
        max_swaps_per_pass = 100  # Limit to prevent excessive time per pass
        
        # Shuffle to break symmetry? Deterministic is fine for reproducibility, but random might escape local minima.
        # Let's use a fixed order for determinism, but maybe randomize the starting point?
        # For now, fixed order.
        
        for c in list(selected_set):
            if swaps_made >= max_swaps_per_pass:
                break
            
            # Find unique words covered by c
            # unique_neighbors: words v in ball[c] such that cov[v] == 1
            # Note: cov[c] >= 1 always.
            
            unique_neighbors = balls[c][cov[balls[c]] == 1]
            
            if len(unique_neighbors) == 0:
                continue
            
            # Find intersection of balls for these unique neighbors
            # valid_w_set = intersection_{v in unique_neighbors} ball[v]
            
            # Start with first
            valid_w_set = set(balls[unique_neighbors[0]].tolist())
            for v in unique_neighbors[1:]:
                valid_w_set.intersection_update(balls[v].tolist())
                
            # Filter out current code and existing codes
            valid_w_set.discard(c)
            valid_w_set -= selected_set
            
            if not valid_w_set:
                continue
            
            # Score candidates: maximize redundancy (cover more words with cov > 1)
            # Score(w) = sum(1 for v in ball[w] if cov[v] > 1)
            
            best_w = -1
            best_score = -1
            
            # Convert to list
            w_list = list(valid_w_set)
            
            # If list is large, we might want to sample, but usually it's small (< 50)
            for w in w_list:
                score = np.sum(cov[balls[w]] > 1)
                if score > best_score:
                    best_score = score
                    best_w = w
                    
            if best_w != -1:
                # Perform swap
                remove_code(c)
                add_code(best_w)
                improved = True
                swaps_made += 1
                # Break to re-evaluate? Or continue?
                # Continuing is fine, but cov changed.
                # Let's continue to try more swaps in this pass if budget allows.
                # But if we swap, the 'unique_neighbors' for other codes might change.
                # It's safer to break and restart the pass?
                # Given the small number of swaps, breaking is safer for correctness of 'unique' calculation for subsequent items in the same pass.
                # However, we can just continue and rely on the next pass to fix any issues.
                # Let's break to ensure we don't make invalid moves based on stale cov.
                break
                
    # 3. Final Polish: High-Precision Swap
    # Use a tighter tolerance or more aggressive search for the last few seconds.
    # Try all possible swaps again, but with a better scoring function?
    # Or just run the same loop again.
    
    if time.time() < deadline - 1.0:
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for c in list(selected_set):
                if np.min(cov[balls[c]]) > 1:
                    remove_code(c)
                    improved = True
                    continue
                
                unique_neighbors = balls[c][cov[balls[c]] == 1]
                if len(unique_neighbors) == 0:
                    continue
                
                valid_w_set = set(balls[unique_neighbors[0]].tolist())
                for v in unique_neighbors[1:]:
                    valid_w_set.intersection_update(balls[v].tolist())
                
                valid_w_set.discard(c)
                valid_w_set -= selected_set
                
                if not valid_w_set:
                    continue
                
                best_w = -1
                best_score = -1
                
                # Try to find a swap that minimizes the increase in max coverage?
                # Or just maximize redundancy as before.
                for w in valid_w_set:
                    # Calculate score
                    score = np.sum(cov[balls[w]] > 1)
                    # Tie-breaker: minimize max cov after swap?
                    # This is expensive. Let's stick to redundancy.
                    if score > best_score:
                        best_score = score
                        best_w = w
                        
                if best_w != -1:
                    remove_code(c)
                    add_code(best_w)
                    improved = True
                    break
                    
    # Final Verification
    final_cov = np.zeros(total, dtype=np.int32)
    for c in selected_set:
        final_cov[balls[c]] += 1
        
    if np.any(final_cov == 0):
        # Add missing words
        missing = np.where(final_cov == 0)[0]
        for m in missing:
            add_code(m)
            final_cov[balls[m]] += 1

    # Convert to lists of digits
    result = []
    for w in selected_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
