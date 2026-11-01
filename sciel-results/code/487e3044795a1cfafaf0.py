import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    import itertools
    if q == 3 and n >= 4:
        # Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute neighbors: neighbors[w] = list of 17 indices (w itself + 16 neighbors)
    neighbors = np.zeros((total, 17), dtype=np.int32)
    
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = 3 ** (n - 1 - pos)
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors[w, idx] = new_w
                idx += 1

    # Precompute ball_membership: ball_membership[w, c] = 1 if w is in ball(c)
    # Since distance is symmetric, w in ball(c) iff c in ball(w)
    # So ball_membership[w, c] = 1 iff c in neighbors[w]
    ball_membership = np.zeros((total, total), dtype=np.int8)
    for w in range(total):
        ball_membership[w, neighbors[w]] = 1

    best_size = total
    best_set = None
    
    while time.time() < deadline - 2.0:
        # 1. Randomized Greedy Construction
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total
        
        while uncovered_count > 0:
            # For each candidate c, count uncovered words in its ball
            counts = uncovered_mask[neighbors].sum(axis=1)
            
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            ball = neighbors[best_idx]
            was_uncovered = uncovered_mask[ball]
            num_newly_covered = np.sum(was_uncovered)
            uncovered_count -= num_newly_covered
            
            cover_count[ball] += 1
            uncovered_mask[ball] = False
            
            if time.time() > deadline - 5.0:
                break
        
        if uncovered_count > 0:
            continue
        
        # 2. Local Search with vectorized swap checking
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            
            # Pass 1: Removal - a codeword c is removable if min(cover_count[neighbors[c]]) > 1
            min_counts = np.min(cover_count[neighbors], axis=1)
            
            # Find all removable codewords in selected
            removable_mask = min_counts > 1
            removable_indices = [i for i, cw in enumerate(selected) if removable_mask[cw]]
            
            if removable_indices:
                # Remove all removable codewords at once
                for i in reversed(removable_indices):
                    cw = selected[i]
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                changed = True
                continue
            
            # No removals possible. Try swaps using ball_membership matrix.
            if time.time() < deadline - 8.0 and len(selected) > 486:
                # Build set of selected for quick lookup
                selected_set = set(selected)
                selected_set_mask = np.zeros(total, dtype=bool)
                for cw in selected:
                    selected_set_mask[cw] = True
                
                # For each selected codeword, find words uniquely covered by it
                # U_c = {w in neighbors[c] : cover_count[w] == 1}
                
                # We need to find c' not in selected such that U_c ⊆ neighbors[c']
                # Using ball_membership: for word w, candidates c' are those where ball_membership[w, c'] = 1
                # i.e., c' in neighbors[w]
                
                # Try random swaps
                found_swap = False
                num_attempts = 50
                for attempt in range(num_attempts):
                    if time.time() > deadline - 5.0:
                        break
                    
                    idx_to_swap = np.random.randint(len(selected))
                    cw_removed = selected[idx_to_swap]
                    
                    # Find words uniquely covered by cw_removed
                    ball_words = neighbors[cw_removed]
                    U = ball_words[cover_count[ball_words] == 1]
                    
                    if len(U) == 0:
                        # Actually removable
                        selected.pop(idx_to_swap)
                        cover_count[neighbors[cw_removed]] -= 1
                        changed = True
                        found_swap = True
                        break
                    
                    # Find candidate replacements: must cover all words in U
                    # Start with candidates that cover U[0]
                    candidates = neighbors[U[0]]
                    candidates = candidates[(candidates != cw_removed) & (~selected_set_mask[candidates])]
                    
                    if len(candidates) == 0:
                        continue
                    
                    # For each candidate, check if it covers all words in U
                    # Using ball_membership: ball_membership[U, cand] should all be 1
                    # ball_membership is (total, total), so ball_membership[U, candidates] is (|U|, |candidates|)
                    coverage_matrix = ball_membership[U, candidates]
                    # A candidate covers all of U if all entries in its column are 1
                    covers_all = coverage_matrix.all(axis=0)
                    
                    valid_candidates = candidates[covers_all]
                    
                    if len(valid_candidates) > 0:
                        # Pick one (preferably one that also covers other words to increase redundancy)
                        # Count how many additional words each valid candidate covers
                        # Actually, just pick the first one for simplicity
                        cw_new = valid_candidates[np.random.randint(len(valid_candidates))]
                        
                        selected[idx_to_swap] = cw_new
                        cover_count[neighbors[cw_removed]] -= 1
                        cover_count[neighbors[cw_new]] += 1
                        selected_set_mask[cw_removed] = False
                        selected_set_mask[cw_new] = True
                        found_swap = True
                        changed = True
                        break
                
                if not found_swap:
                    # Stuck, break out of local search
                    break
        # End Local Search

        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
        if best_size <= 486:
            break

    # Convert best_set to list of digit lists
    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n).ljust(n, '0')) for w in range(total)]
        
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
