import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
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
    # balls[w, 0] = w, balls[w, 1:17] = neighbors
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
    
    # 1. Greedy Construction
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)
    
    # Main greedy loop
    while np.any(current_uncovered):
        if time.time() > deadline - 40.0:
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
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
        
    # If timed out during greedy, force completion
    if np.any(current_uncovered) and time.time() < deadline - 10.0:
        while np.any(current_uncovered):
            uncovered_indices = np.where(current_uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            best_idx = uncovered_indices[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
            if time.time() > deadline - 10.0:
                break
    
    # 2. Aggressive Pruning (Greedy Removal)
    changed = True
    while changed and time.time() < deadline - 20.0:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
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
                    changed = True
                else:
                    new_selected.append(c)
            selected = list(reversed(new_selected))

    # 3. Spectral-inspired Local Search: Swap-based improvement
    # Use a degree-based heuristic: prioritize swapping codewords with high "overlap"
    # (i.e., many neighbors already covered by other codewords) with words that
    # are "under-covered" (low cover count) to potentially enable removals.
    
    if time.time() < deadline - 5.0:
        selected_set = set(selected)
        improved = True
        max_iterations = 15  # Increased iterations for better convergence
        iter_count = 0
        while improved and time.time() < deadline - 5.0 and iter_count < max_iterations:
            improved = False
            iter_count += 1
            new_selected = list(selected)
            new_selected_set = set(new_selected)
            
            for i in range(len(new_selected)):
                if time.time() > deadline - 5.0:
                    break
                c = new_selected[i]
                
                # Compute "overlap score" for c: how many of its ball's words are
                # covered by at least 2 codewords (i.e., c is "redundant" for those)
                ball_c = balls[c]
                overlap_score = np.sum(cover_count[ball_c] >= 2)
                
                # Try to replace c with a neighbor w that has low cover count
                # (i.e., w is "under-covered" and might benefit from new coverage)
                best_w = None
                best_gain = 0
                for w in balls[c]:
                    if w in new_selected_set:
                        continue
                    
                    # Estimate gain: how many words in balls[w] are currently
                    # covered by only 1 codeword (would become 2 if we add w)
                    # and how many words in balls[c] are covered by only c
                    # (would become 0 if we remove c, unless w covers them)
                    
                    ball_w = balls[w]
                    # Words that would be uncovered if we remove c (only covered by c)
                    only_c = (cover_count[ball_c] == 1) & (ball_c != w)  # approximate
                    # Words that would gain coverage if we add w
                    gain_w = np.sum(cover_count[ball_w] == 1)
                    
                    # Net gain heuristic
                    net_gain = gain_w - np.sum(only_c)
                    
                    if net_gain > best_gain:
                        best_gain = net_gain
                        best_w = w
                
                if best_w is not None and best_gain > 0:
                    w = best_w
                    # Try swap: remove c, add w
                    cover_count[balls[c]] -= 1
                    if np.any(cover_count == 0):
                        cover_count[balls[c]] += 1  # Restore
                        continue
                    
                    cover_count[balls[w]] += 1
                    
                    # Check if we can remove another codeword d
                    removable_d = None
                    for j in range(len(new_selected)):
                        if j == i:
                            continue
                        d = new_selected[j]
                        if np.min(cover_count[balls[d]]) > 1:
                            removable_d = d
                            break
                    
                    if removable_d is not None:
                        cover_count[balls[removable_d]] -= 1
                        new_selected[i] = w
                        new_selected_set.remove(c)
                        new_selected_set.add(w)
                        new_selected.remove(removable_d)
                        new_selected_set.remove(removable_d)
                        improved = True
                        break
                    else:
                        cover_count[balls[w]] -= 1
                        cover_count[balls[c]] += 1
                
                if improved:
                    selected = new_selected
                    selected_set = new_selected_set
                    break
            
            # If no improvement in this pass, break

    # 4. Final Pruning Pass
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

    # 5. Final Verification and Conversion
    final_cover = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cover[balls[c]] += 1
    
    if np.any(final_cover == 0):
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
            final_cover[balls[m]] += 1

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
