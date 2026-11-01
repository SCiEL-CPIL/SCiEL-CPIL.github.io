import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_iterated(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (valid format, worst case)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_iterated(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls for all words
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
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
                
    best_size = total
    best_selected = None
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 10.0:
        
        # 1. Randomized Greedy Construction
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        num_uncovered = total
        current_uncovered = np.ones(total, dtype=bool) # Still needed for fast sum in greedy?
        # Actually, for greedy, we need counts of uncovered neighbors.
        # We can compute this using cover_count == 0.
        
        while num_uncovered > 0:
            if time.time() > deadline - 15.0:
                break
            
            # Count uncovered neighbors for each word
            # current_uncovered is (cover_count == 0)
            # We can avoid creating current_uncovered explicitly if we use cover_count
            # But np.sum(cover_count[balls] == 0, axis=1) is slow.
            # Let's stick to current_uncovered for the greedy phase, it's faster for the sum.
            # But we must update it.
            
            # Optimization: Instead of recomputing current_uncovered from scratch (O(N)),
            # we can update it incrementally?
            # When we add a codeword, some words go from 0 to 1.
            # We can just do: current_uncovered = (cover_count == 0)
            # This is O(N). 5000 * 6561 = 32M ops. Acceptable.
            
            counts = np.sum(current_uncovered[balls], axis=1)
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[rng.integers(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover_count and num_uncovered
            for w in balls[best_idx]:
                if cover_count[w] == 0:
                    num_uncovered -= 1
                cover_count[w] += 1
            
            # Update current_uncovered for next iteration
            current_uncovered = (cover_count == 0)
        
        # If timed out, complete
        if num_uncovered > 0:
            if time.time() > deadline - 10.0:
                break
            while num_uncovered > 0:
                # Find an uncovered word
                # We need to find index where cover_count == 0
                # Since current_uncovered is stale, we recompute or search
                # For completion, just pick random or first found
                # Efficient: np.where(cover_count == 0)
                uncovered_indices = np.where(cover_count == 0)[0]
                if len(uncovered_indices) == 0:
                    break
                best_idx = uncovered_indices[0]
                selected.append(best_idx)
                for w in balls[best_idx]:
                    if cover_count[w] == 0:
                        num_uncovered -= 1
                    cover_count[w] += 1
                if time.time() > deadline - 10.0:
                    break
        
        # 2. Aggressive Pruning
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if removing c is safe
                # min cover count in ball[c] must be > 1
                if np.min(cover_count[balls[c]]) > 1:
                    # Remove
                    for w in balls[c]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            num_uncovered += 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # 3. Enhanced Local Search
        improved = True
        max_swap_rounds = 50 # Increase rounds since it's faster now
        round_count = 0
        while improved and time.time() < deadline - 8.0 and round_count < max_swap_rounds:
            improved = False
            round_count += 1
            
            selected_set = set(selected)
            
            for i in range(len(selected)):
                if time.time() > deadline - 8.0:
                    break
                c = selected[i]
                
                for w in balls[c]:
                    if w in selected_set:
                        continue
                    
                    # Try replacing c with w
                    # 1. Remove c
                    # Check if we can remove c safely?
                    # We are doing a swap, so we assume c is removed and w is added.
                    # We need to ensure coverage is maintained.
                    
                    # Temporarily remove c
                    # We need to know if any word becomes uncovered.
                    # Words affected: balls[c] \ balls[w] ? No.
                    # Words that lose coverage: balls[c].
                    # Words that gain coverage: balls[w].
                    
                    # Safe removal condition:
                    # For all x in balls[c], if cover_count[x] == 1, then x must be in balls[w].
                    # i.e., balls[c] \ balls[w] must have cover_count > 1.
                    
                    # Optimization:
                    # Instead of modifying cover_count and checking num_uncovered,
                    # we can check the condition directly.
                    
                    # Let's stick to the modify-restore method but with O(1) check?
                    # No, modify is O(17).
                    
                    # Save old state of affected words?
                    # Just modify.
                    
                    # Remove c
                    for x in balls[c]:
                        cover_count[x] -= 1
                        if cover_count[x] == 0:
                            num_uncovered += 1
                    
                    # Check validity
                    if num_uncovered > 0:
                        # Restore
                        for x in balls[c]:
                            cover_count[x] += 1
                            if cover_count[x] == 1: # Was 0, now 1
                                num_uncovered -= 1
                        continue
                    
                    # Add w
                    for x in balls[w]:
                        if cover_count[x] == 0:
                            num_uncovered -= 1
                        cover_count[x] += 1
                    
                    # Try to remove redundant codewords
                    # We can do a quick pass?
                    # Or just accept the swap if it doesn't increase size?
                    # The goal is to reduce size. A swap keeps size same.
                    # But it might enable future removals.
                    # Let's try to remove redundancies immediately.
                    
                    removals_found = []
                    for j in range(len(selected)):
                        if j == i:
                            continue
                        d = selected[j]
                        if np.min(cover_count[balls[d]]) > 1:
                            removals_found.append(d)
                    
                    if removals_found:
                        # Perform removals
                        for d in removals_found:
                            for x in balls[d]:
                                cover_count[x] -= 1
                                if cover_count[x] == 0:
                                    num_uncovered += 1
                            selected_set.remove(d)
                        
                        # Update selected list
                        new_selected = []
                        for idx, codeword in enumerate(selected):
                            if idx == i:
                                new_selected.append(w)
                            elif codeword in removals_found:
                                continue
                            else:
                                new_selected.append(codeword)
                        
                        selected = new_selected
                        selected_set.add(w)
                        improved = True
                        break
                    else:
                        # No improvement, restore state
                        # Remove w
                        for x in balls[w]:
                            cover_count[x] -= 1
                            if cover_count[x] == 0:
                                num_uncovered += 1
                        # Add c back
                        for x in balls[c]:
                            if cover_count[x] == 0:
                                num_uncovered -= 1
                            cover_count[x] += 1
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            # Pruning after swap
            if improved:
                changed = True
                while changed and time.time() < deadline - 8.0:
                    changed = False
                    new_selected = []
                    for c in selected:
                        if np.min(cover_count[balls[c]]) > 1:
                            for x in balls[c]:
                                cover_count[x] -= 1
                                if cover_count[x] == 0:
                                    num_uncovered += 1
                            changed = True
                        else:
                            new_selected.append(c)
                    selected = new_selected

        # 4. Final Pruning
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    for x in balls[c]:
                        cover_count[x] -= 1
                        if cover_count[x] == 0:
                            num_uncovered += 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected

        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected.copy()
            if best_size <= 486:
                break

    if best_selected is None:
        best_selected = list(range(total))
        
    result = []
    for w in best_selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
