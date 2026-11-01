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

    # Precompute reverse neighbors: for each word w, which codewords cover w?
    # reverse_neighbors[w] = list of codewords c such that w is in ball(c)
    # Since the relation is symmetric (w in ball(c) iff c in ball(w)), 
    # reverse_neighbors[w] = neighbors[w] as well!
    # Wait, that's only if the distance is symmetric, which it is.
    # So neighbors[w] gives us the words covered by codeword w.
    # For a given word w, the codewords that cover w are exactly {c : w in neighbors[c]}.
    # This is the same as {c : c in neighbors[w]} because distance is symmetric.
    # So reverse_neighbors[w] = neighbors[w]!
    
    best_size = total
    best_set = None
    
    while time.time() < deadline - 2.0:
        # 1. Randomized Greedy Construction
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total
        
        while uncovered_count > 0:
            # Count how many uncovered words each candidate covers
            # A candidate c covers word w if w is in neighbors[c], i.e., c is in neighbors[w].
            # So for each uncovered word w, it contributes to the count of all c in neighbors[w].
            # We want: for each c, count of w in uncovered such that w in neighbors[c]
            # = for each c, count of w in neighbors[c] that are uncovered
            # = sum over c of (uncovered_mask[neighbors[c]]).sum()
            
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
        
        # 2. Local Search
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            
            # Pass 1: Removal
            # A codeword c is removable if for all w in neighbors[c], cover_count[w] > 1
            # i.e., min(cover_count[neighbors[c]]) > 1
            
            min_counts = np.min(cover_count[neighbors], axis=1)
            
            # Find removable codewords that are in selected
            for i in range(len(selected)):
                cw = selected[i]
                if min_counts[cw] > 1:
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    changed = True
                    break
            
            if not changed:
                # No removals possible. Try swaps.
                # For each selected codeword c, check if we can replace it with another codeword c'
                # such that the coverage is maintained.
                # When we remove c, the words that become uncovered are those w in neighbors[c] 
                # where cover_count[w] == 1 (i.e., only c covers them).
                # Let U = {w in neighbors[c] : cover_count[w] == 1}
                # We need a codeword c' not in selected such that U is subset of neighbors[c'].
                
                if time.time() < deadline - 10.0 and len(selected) > 486:
                    # Try a swap
                    orig_selected = selected.copy()
                    orig_cover_count = cover_count.copy()
                    
                    # Pick a random codeword to try to replace
                    idx_to_swap = np.random.randint(len(selected))
                    cw_removed = selected[idx_to_swap]
                    
                    # Find words that would become uncovered
                    U = []
                    for w in neighbors[cw_removed]:
                        if cover_count[w] == 1:
                            U.append(w)
                    
                    U = np.array(U, dtype=np.int32)
                    
                    if len(U) == 0:
                        # This codeword is actually removable, which shouldn't happen
                        # since we just checked. But just in case:
                        selected.pop(idx_to_swap)
                        cover_count[neighbors[cw_removed]] -= 1
                        changed = True
                        continue
                    
                    # Find a replacement codeword that covers all words in U
                    # and is not already in selected
                    selected_set = set(selected)
                    replacement_found = False
                    
                    # Candidates must cover U[0], so they must be in neighbors[U[0]]
                    for cw2 in neighbors[U[0]]:
                        if cw2 == cw_removed:
                            continue
                        if cw2 in selected_set:
                            continue
                        
                        # Check if cw2 covers all words in U
                        covers_all = True
                        for w in U:
                            if w not in neighbors[cw2]:
                                covers_all = False
                                break
                        
                        if covers_all:
                            # Found a replacement
                            selected[idx_to_swap] = cw2
                            cover_count[neighbors[cw_removed]] -= 1
                            cover_count[neighbors[cw2]] += 1
                            replacement_found = True
                            changed = True
                            break
                    
                    if not replacement_found:
                        # Revert
                        selected = orig_selected
                        cover_count = orig_cover_count
                        # Try a few more random swaps before giving up
                        for _ in range(5):
                            if time.time() > deadline - 5.0:
                                break
                            idx_to_swap = np.random.randint(len(selected))
                            cw_removed = selected[idx_to_swap]
                            
                            U = []
                            for w in neighbors[cw_removed]:
                                if cover_count[w] == 1:
                                    U.append(w)
                            
                            U = np.array(U, dtype=np.int32)
                            
                            if len(U) == 0:
                                selected.pop(idx_to_swap)
                                cover_count[neighbors[cw_removed]] -= 1
                                changed = True
                                break
                            
                            selected_set = set(selected)
                            for cw2 in neighbors[U[0]]:
                                if cw2 == cw_removed:
                                    continue
                                if cw2 in selected_set:
                                    continue
                                
                                covers_all = True
                                for w in U:
                                    if w not in neighbors[cw2]:
                                        covers_all = False
                                        break
                                
                                if covers_all:
                                    selected[idx_to_swap] = cw2
                                    cover_count[neighbors[cw_removed]] -= 1
                                    cover_count[neighbors[cw2]] += 1
                                    replacement_found = True
                                    changed = True
                                    break
                            
                            if replacement_found:
                                break
                        if not replacement_found:
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
