import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters (not expected in this task, but good practice)
    if q == 3 and n >= 4:
        # Simple tetracode extension, not optimal but valid
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute weights for base-3 digits
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls (neighborhoods of radius 1)
    # balls[w] is a list of 17 indices: w itself and its 16 neighbors
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
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
                
    all_indices = np.arange(total, dtype=np.int32)
    
    # Precompute ball membership as a boolean matrix for fast subset checks
    # ball_membership[w, v] = True if v is in balls[w]
    ball_membership = np.zeros((total, total), dtype=bool)
    for w in range(total):
        ball_membership[w, balls[w]] = True
    
    best_size = total
    best_set = None
    
    # Simulated Annealing approach
    # Start with a greedy solution
    selected = []
    cover_count = np.zeros(total, dtype=np.int32)
    uncovered_mask = np.ones(total, dtype=bool)
    
    # Initial greedy construction
    while np.any(uncovered_mask):
        if time.time() > deadline - 10.0:
            break
        counts = uncovered_mask[balls].sum(axis=1)
        if np.max(counts) == 0:
            break
        max_count = np.max(counts)
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[np.random.randint(len(candidates))]
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        uncovered_mask = (cover_count == 0)
    
    if np.any(uncovered_mask):
        # Fallback to simple greedy if timeout
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
    
    best_size = len(selected)
    best_set = selected.copy()
    
    # Simulated annealing parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.995
    T = T_start
    
    # SA loop
    while time.time() < deadline - 5.0 and T > T_end:
        T *= cooling_rate
        
        # Random move: pick a random codeword and either remove it, swap it, or add a new one
        move_type = np.random.random()
        
        if move_type < 0.3 and len(selected) > 100:
            # Try to remove a random codeword
            idx = np.random.randint(len(selected))
            c = selected[idx]
            
            # Check if removing c leaves any word uncovered
            cover_count[balls[c]] -= 1
            uncovered = np.any(cover_count == 0)
            
            if not uncovered:
                # Accept removal (improvement)
                selected.pop(idx)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected.copy()
                    if best_size <= 403:
                        break
            else:
                # Revert and accept with probability exp(-delta/T)
                cover_count[balls[c]] += 1
                # delta = number of newly uncovered words (penalty)
                # For simplicity, use a fixed penalty for invalid states
                delta = 100  # High penalty for invalid
                if np.random.random() < np.exp(-delta / T):
                    # Accept invalid state temporarily
                    selected.pop(idx)
        elif move_type < 0.7:
            # Try to swap a random codeword with another
            if len(selected) == 0:
                continue
            idx = np.random.randint(len(selected))
            c = selected[idx]
            
            # Find candidates that cover all critical words of c
            critical_mask = (cover_count[balls[c]] == 1)
            critical_words = balls[c][critical_mask]
            
            if len(critical_words) == 0:
                # Can remove c
                cover_count[balls[c]] -= 1
                selected.pop(idx)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected.copy()
                    if best_size <= 403:
                        break
                continue
            
            candidate_mask = np.ones(total, dtype=bool)
            for cw in critical_words:
                candidate_mask &= ball_membership[:, cw]
            
            valid_candidates = np.where(candidate_mask)[0]
            valid_candidates = valid_candidates[valid_candidates != c]
            
            if len(valid_candidates) > 0:
                c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                
                # Perform swap
                cover_count[balls[c]] -= 1
                cover_count[balls[c_prime]] += 1
                selected[idx] = c_prime
        else:
            # Try to add a new codeword (to escape local minima)
            # Pick a random uncovered word or a random word
            if np.any(cover_count == 0):
                uncovered_words = np.where(cover_count == 0)[0]
                c_new = uncovered_words[np.random.randint(len(uncovered_words))]
            else:
                c_new = np.random.randint(total)
            
            # Check if c_new is already in selected
            if c_new in selected:
                continue
            
            # Add c_new
            cover_count[balls[c_new]] += 1
            selected.append(c_new)
            
            # With some probability, accept this (even though it increases size)
            # This helps escape local minima
            if np.random.random() < 0.1:  # 10% chance to keep the added codeword
                pass  # Keep it
            else:
                # Revert
                cover_count[balls[c_new]] -= 1
                selected.pop()
        
        # Periodically clean up: remove redundant codewords
        if np.random.random() < 0.1:
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # Track best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            if best_size <= 403:
                break
            if best_size <= 486:
                break

    # Final local search polish on best_set
    selected = best_set.copy()
    cover_count = np.zeros(total, dtype=np.int32)
    for c in selected:
        cover_count[balls[c]] += 1
    
    # Phase 1: Remove redundant
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
    
    # Phase 2: Swap
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                improved = True
                continue
            
            critical_mask = (cover_count[balls[c]] == 1)
            critical_words = balls[c][critical_mask]
            
            if len(critical_words) == 0:
                cover_count[balls[c]] -= 1
                improved = True
                continue

            candidate_mask = np.ones(total, dtype=bool)
            for cw in critical_words:
                candidate_mask &= ball_membership[:, cw]
            
            valid_candidates = np.where(candidate_mask)[0]
            valid_candidates = valid_candidates[valid_candidates != c]
            
            if len(valid_candidates) > 0:
                c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                cover_count[balls[c]] -= 1
                cover_count[balls[c_prime]] += 1
                new_selected.append(c_prime)
                improved = True
            else:
                new_selected.append(c)
        
        selected = new_selected
    
    if len(selected) < best_size:
        best_size = len(selected)
        best_set = selected.copy()

    # Convert best_set to output format
    if best_set is None:
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_set = selected

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
