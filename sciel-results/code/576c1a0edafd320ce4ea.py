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
    
    # Precompute neighbors (forward: word -> its 17 ball members)
    neighbors = np.zeros((total, 17), dtype=np.int32)
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for pos in range(n):
            weight = 3 ** (n - 1 - pos)
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors[w, idx] = new_w
                idx += 1

    # Precompute reverse neighbors: for each word, which codewords cover it
    rev_neighbors = [[] for _ in range(total)]
    for c in range(total):
        for w in neighbors[c]:
            rev_neighbors[w].append(c)

    best_size = total
    best_set = None
    
    while time.time() < deadline - 2.0:
        # 1. Dual-guided Greedy Construction with shadow prices
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total
        
        # Shadow price: for each uncovered word, count how many remaining candidates can cover it
        # Initially all 6561 words can cover each word, so shadow_price[w] = len(rev_neighbors[w]) = 17
        # As we select codewords, we update shadow prices for uncovered words
        shadow_prices = np.array([len(rev_neighbors[w]) for w in range(total)], dtype=np.float64)
        
        while uncovered_count > 0 and time.time() < deadline - 5.0:
            # For each candidate codeword c, compute weighted coverage:
            # sum over uncovered words w in ball(c) of shadow_prices[w]
            # This prioritizes covering words that are "scarce" (few remaining candidates)
            
            # Efficient computation: 
            # weighted_counts[c] = sum of shadow_prices[w] for w in neighbors[c] where uncovered_mask[w]
            # We can precompute this using numpy fancy indexing
            
            # Get shadow prices for all words in each candidate's ball
            ball_shadow = shadow_prices[neighbors]  # shape (total, 17)
            
            # Mask out covered words
            weighted_counts = np.where(uncovered_mask[neighbors], ball_shadow, 0.0).sum(axis=1)
            
            # Among candidates, pick the one with highest weighted coverage
            max_weighted = np.max(weighted_counts)
            if max_weighted == 0:
                break
            
            candidates = np.where(weighted_counts == max_weighted)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            ball = neighbors[best_idx]
            was_uncovered = uncovered_mask[ball]
            num_newly_covered = np.sum(was_uncovered)
            uncovered_count -= num_newly_covered
            
            cover_count[ball] += 1
            uncovered_mask[ball] = False
            
            # Update shadow prices: for each uncovered word w, 
            # if w was in ball(best_idx), decrement its shadow price
            # (because best_idx is no longer a candidate)
            if num_newly_covered > 0:
                # For words that were just covered, we don't need to update their shadow price
                # For words that remain uncovered, if they were in ball(best_idx), decrement
                # Actually, shadow_price[w] = number of remaining candidates that can cover w
                # When we select best_idx, for any uncovered word w in ball(best_idx),
                # we remove best_idx from its candidate set
                uncovered_in_ball = np.where(uncovered_mask[ball])[0]
                if len(uncovered_in_ball) > 0:
                    shadow_prices[uncovered_in_ball] -= 1
                    # Clamp to 0 (shouldn't go negative if logic is correct)
                    shadow_prices[uncovered_in_ball] = np.maximum(shadow_prices[uncovered_in_ball], 0)
        
        if uncovered_count > 0:
            continue
        
        # 2. Local Search: Removal and Swap
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            
            # Pass 1: Removal - find all codewords that are removable
            min_counts = np.min(cover_count[neighbors], axis=1)
            
            removable_in_selected = [c for c in selected if min_counts[c] > 1]
            
            if len(removable_in_selected) > 0:
                cw = removable_in_selected[0]
                selected.remove(cw)
                cover_count[neighbors[cw]] -= 1
                changed = True
                continue
            
            # No removals possible. Try Swaps.
            if time.time() < deadline - 10.0 and len(selected) > 486:
                orig_selected = selected.copy()
                orig_cover_count = cover_count.copy()
                
                idx_to_remove = np.random.randint(len(selected))
                cw_removed = selected.pop(idx_to_remove)
                cover_count[neighbors[cw_removed]] -= 1
                
                U = np.where(cover_count == 0)[0]
                
                if len(U) > 0:
                    candidate_set = set(rev_neighbors[U[0]])
                    
                    for w in U[1:]:
                        candidate_set &= set(rev_neighbors[w])
                        if len(candidate_set) == 0:
                            break
                    
                    valid_candidates = [c for c in candidate_set if c != cw_removed and c not in selected]
                    
                    replacement_found = False
                    if len(valid_candidates) > 0:
                        cw2 = valid_candidates[np.random.randint(len(valid_candidates))]
                        selected.append(cw2)
                        cover_count[neighbors[cw2]] += 1
                        replacement_found = True
                    else:
                        u_mask = np.zeros(total, dtype=bool)
                        u_mask[U] = True
                        u_counts = u_mask[neighbors].sum(axis=1)
                        for c in selected:
                            u_counts[c] = 0
                        u_counts[cw_removed] = 0
                        
                        best_cw = np.argmax(u_counts)
                        if u_counts[best_cw] > 0:
                            selected.append(best_cw)
                            cover_count[neighbors[best_cw]] += 1
                            remaining = np.where(cover_count == 0)[0]
                            while len(remaining) > 0 and time.time() < deadline - 3.0:
                                rem_mask = np.zeros(total, dtype=bool)
                                rem_mask[remaining] = True
                                rem_counts = rem_mask[neighbors].sum(axis=1)
                                for c in selected:
                                    rem_counts[c] = 0
                                bc = np.argmax(rem_counts)
                                if rem_counts[bc] == 0:
                                    break
                                selected.append(bc)
                                cover_count[neighbors[bc]] += 1
                                remaining = np.where(cover_count == 0)[0]
                            replacement_found = True
                        else:
                            pass
                    
                    if not replacement_found:
                        selected = orig_selected
                        cover_count = orig_cover_count
                        changed = False
                        break
                else:
                    pass
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
