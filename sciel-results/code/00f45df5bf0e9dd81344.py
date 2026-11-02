import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        # Tetracode product for n=8, general product for others
        if n == 8:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # This gives a radius 2 covering with 81 words.
            # We need radius 1. This fallback is just to return *something* valid if main fails, 
            # but 81 words radius 2 is not valid for radius 1 task.
            # The task requires radius 1.
            # So we must return a valid radius 1 code.
            # Greedy is safer.
            pass
        
        # Simple greedy fallback
        total = q ** n
        weights = [q ** (n - 1 - i) for i in range(n)]
        
        balls = np.zeros((total, 1 + (q-1)*n), dtype=np.int32)
        for w in range(total):
            balls[w, 0] = w
            idx = 1
            for pos in range(n):
                weight = weights[pos]
                digit = (w // weight) % q
                for new_digit in range(q):
                    if new_digit == digit:
                        continue
                    new_w = w + (new_digit - digit) * weight
                    balls[w, idx] = new_w
                    idx += 1

        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            if np.max(counts) == 0:
                break
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            
        # Convert to list
        result = []
        for w in selected:
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            digits.reverse()
            result.append(digits)
        return result

    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute neighbors (balls)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(3):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
        
    # Initialize with a greedy set to get a valid starting point
    # Greedy is fast and gives ~430-440 words
    selected = []
    cover_count = np.zeros(total, dtype=np.int32)
    uncovered_mask = np.ones(total, dtype=bool)
    
    while np.any(uncovered_mask) and time.time() < deadline - 60.0:
        counts = uncovered_mask[balls].sum(axis=1)
        if np.max(counts) == 0:
            break
        max_count = np.max(counts)
        candidates = np.where(counts == max_count)[0]
        # Randomize to get different starts
        best_idx = candidates[np.random.randint(len(candidates))]
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        uncovered_mask = (cover_count == 0)
        
    if np.any(uncovered_mask):
        # Fallback if greedy failed (shouldn't happen)
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]
        
    # Local Search Phase 1: Greedy Removal
    # Remove any word that is not uniquely covering any point
    changed = True
    while changed and time.time() < deadline - 30.0:
        changed = False
        new_selected = []
        for c in selected:
            # Check if min count in ball(c) is > 1
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
        
    best_size = len(selected)
    best_set = selected.copy()
    
    # Local Search Phase 2: Simulated Annealing / Randomized Swaps
    # We try to swap words to allow further removals or just reduce size if we add/remove pairs
    # Actually, let's do "1-in 1-out" swaps that maintain validity, 
    # followed by another removal pass.
    
    # To make this efficient, we track which words are "critical" (min count == 1 in their ball)
    # A swap c -> c' is valid if ball(c) \ ball(c') is not covered by other words.
    # i.e., for all x in ball(c), if x not in ball(c'), cover_count[x] > 1.
    
    # We will iterate through several restarts of this local search with random perturbations.
    
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
            
        # Perturb: Randomly swap a few words
        current_selected = best_set.copy()
        current_cover = np.zeros(total, dtype=np.int32)
        for c in current_selected:
            current_cover[balls[c]] += 1
            
        # Random Perturbation: Swap 5 random words with random others
        for _ in range(5):
            if len(current_selected) < 2:
                break
            i = np.random.randint(len(current_selected))
            c_out = current_selected[i]
            c_in = np.random.randint(total)
            if c_in in current_selected:
                continue
            # Check if swap is valid
            # We need to remove c_out and add c_in.
            # Temporarily remove c_out
            current_cover[balls[c_out]] -= 1
            # Check if any point in ball(c_out) becomes 0
            if np.min(current_cover[balls[c_out]]) < 0:
                current_cover[balls[c_out]] += 1
                continue
            # Add c_in
            current_cover[balls[c_in]] += 1
            current_selected[i] = c_in
            
        # Refinement: Greedy Removal
        changed = True
        while changed and time.time() < deadline - 15.0:
            changed = False
            new_selected = []
            for c in current_selected:
                if np.min(current_cover[balls[c]]) > 1:
                    current_cover[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            current_selected = new_selected
            
        # Refinement: Try to swap to enable removal
        # This is expensive. Let's just do a few random valid swaps.
        for _ in range(50):
            if time.time() > deadline - 15.0:
                break
            if len(current_selected) < 2:
                break
            i = np.random.randint(len(current_selected))
            c_out = current_selected[i]
            c_in = np.random.randint(total)
            if c_in in current_selected:
                continue
            
            # Check validity of swap
            # Remove c_out
            current_cover[balls[c_out]] -= 1
            valid = True
            if np.min(current_cover[balls[c_out]]) < 0:
                valid = False
            
            if valid:
                # Add c_in
                current_cover[balls[c_in]] += 1
                current_selected[i] = c_in
                
                # Try to remove something else now
                # Just do one removal pass
                new_selected = []
                for c in current_selected:
                    if np.min(current_cover[balls[c]]) > 1:
                        current_cover[balls[c]] -= 1
                    else:
                        new_selected.append(c)
                current_selected = new_selected
            
            else:
                # Revert
                current_cover[balls[c_out]] += 1
                
        if len(current_selected) < best_size:
            best_size = len(current_selected)
            best_set = current_selected.copy()
            
            if best_size <= 403:
                break

    # Final cleanup
    # Convert best_set to output format
    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]
        
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
