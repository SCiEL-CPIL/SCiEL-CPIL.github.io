import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 150s limit (evaluator kills at 150s)
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (valid format, worst case)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Weights for positions (big-endian)
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    # Precompute balls for all words
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
    
    # Precompute a boolean mask for quick set membership
    # selected_mask[i] == True if word i is in selected set
    # This is faster than a Python set for numpy operations
    
    # 1. Greedy Construction
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    selected_set = set()
    current_uncovered = np.ones(total, dtype=bool)
    
    while np.any(current_uncovered):
        if time.time() > deadline - 30.0:
            break
            
        # Count uncovered neighbors for each word
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        
        if max_count == 0:
            break
        
        # Deterministic: pick smallest index among max-count candidates
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        
        selected.append(best_idx)
        selected_set.add(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
        
    # Ensure completion if greedy timed out
    if np.any(current_uncovered) and time.time() < deadline - 10.0:
        while np.any(current_uncovered):
            uncovered_indices = np.where(current_uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            best_idx = uncovered_indices[0]
            selected.append(best_idx)
            selected_set.add(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
            if time.time() > deadline - 10.0:
                break
    
    # 2. Aggressive Pruning (Greedy Removal)
    # We want to remove as many codewords as possible.
    # A codeword can be removed if all words in its ball are covered by at least one other codeword.
    
    changed = True
    while changed and time.time() < deadline - 20.0:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                selected_set.remove(c)
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
        
    # Try reverse order pass
    if time.time() < deadline - 20.0:
        changed = True
        while changed and time.time() < deadline - 20.0:
            changed = False
            new_selected = []
            for c in reversed(selected):
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    selected_set.remove(c)
                    changed = True
                else:
                    new_selected.append(c)
            selected = list(reversed(new_selected))
    
    # 3. Local Search: Swap-based improvement
    # Try to replace a codeword with another to allow further removals.
    
    if time.time() < deadline - 5.0:
        improved = True
        max_iterations = 20  # Increased iterations for better convergence
        iter_count = 0
        while improved and time.time() < deadline - 5.0 and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            for i in range(len(selected)):
                if time.time() > deadline - 5.0:
                    break
                c = selected[i]
                
                # Try to replace c with a neighbor w
                for w in balls[c]:
                    if w in selected_set:
                        continue
                    
                    # Candidate swap: remove c, add w
                    # 1. Remove c
                    cover_count[balls[c]] -= 1
                    # Check if any word is now uncovered
                    if np.any(cover_count == 0):
                        cover_count[balls[c]] += 1  # Restore
                        continue
                    
                    # 2. Add w
                    cover_count[balls[w]] += 1
                    
                    # 3. Check if we can remove another codeword d
                    removable_d = None
                    for j in range(len(selected)):
                        if j == i:  # c
                            continue
                        d = selected[j]
                        if np.min(cover_count[balls[d]]) > 1:
                            removable_d = d
                            break
                    
                    if removable_d is not None:
                        # Perform the swap: Remove c, Add w, Remove d
                        cover_count[balls[removable_d]] -= 1
                        
                        # Update selected list
                        selected[i] = w
                        selected_set.remove(c)
                        selected_set.add(w)
                        
                        selected.remove(removable_d)
                        selected_set.remove(removable_d)
                        
                        improved = True
                        break
                    else:
                        # Swap didn't help remove another node.
                        # Restore state
                        cover_count[balls[w]] -= 1
                        cover_count[balls[c]] += 1
                    
                    if improved:
                        break
                
                if improved:
                    break
    
    # 4. Final Pruning Pass
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                selected_set.remove(c)
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
    
    # 5. Final Verification and Conversion
    final_cover = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cover[balls[c]] += 1
    
    if np.any(final_cover == 0):
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
            selected_set.add(m)
            final_cover[balls[m]] += 1
    
    # Convert to digit lists
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
