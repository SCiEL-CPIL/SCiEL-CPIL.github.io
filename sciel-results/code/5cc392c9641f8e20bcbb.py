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
    
    # Precompute neighbors
    # We use a flat array representation for speed
    # neighbors[cw] = list of 17 indices (cw itself + 16 neighbors)
    neighbors = np.zeros((total, 17), dtype=np.int32)
    
    # Vectorized precomputation of neighbors
    # This is a one-time cost, so a loop is fine and likely faster than complex vectorization for 6561 items
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

    best_size = total
    best_set = None
    
    # We will run multiple iterations of Greedy + Local Search (Removal + Swap)
    # The local search with swaps is key to escaping local optima and reaching 486.
    
    while time.time() < deadline - 2.0:
        # 1. Randomized Greedy Construction
        # To get diversity, we can randomize the tie-breaking in greedy.
        # Or simply run greedy with a random starting point? No, greedy starts with empty set.
        # We can randomize the order of candidates if counts are equal.
        
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        
        # Optimization: Instead of checking np.any(uncovered_mask) which is O(N),
        # we can maintain a count of uncovered words.
        uncovered_count = total
        
        while uncovered_count > 0:
            # Count how many uncovered words each candidate covers
            # uncovered_mask[neighbors] shape (total, 17)
            # sum along axis 1
            counts = uncovered_mask[neighbors].sum(axis=1)
            
            # Find max count
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            # Find all candidates with max count
            candidates = np.where(counts == max_count)[0]
            
            # Pick one randomly to add diversity
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover counts and uncovered mask
            # We need to update uncovered_count carefully
            ball = neighbors[best_idx]
            # Words in ball that were uncovered
            was_uncovered = uncovered_mask[ball]
            num_newly_covered = np.sum(was_uncovered)
            uncovered_count -= num_newly_covered
            
            cover_count[ball] += 1
            uncovered_mask[ball] = False
            
            # Check time
            if time.time() > deadline - 5.0:
                break
        
        if uncovered_count > 0:
            # If we broke early due to time, this set might be invalid.
            # We should only consider it if it's valid.
            # Since we broke, it's likely invalid. Skip this iteration.
            continue
        
        # 2. Local Search: Removal and Swap
        # We maintain cover_count and the list of selected codewords.
        # Removal: If min(cover_count[neighbors[cw]]) > 1, remove cw.
        # Swap: If removing cw makes some words uncovered, try to add another cw2 that covers them.
        # A simpler swap: 
        #   1. Identify "critical" words (cover_count == 1).
        #   2. A codeword cw is "removable" if it doesn't cover any critical word.
        #   3. If no codeword is removable, try a swap:
        #      Pick a codeword cw that covers a critical word.
        #      Remove it (temporarily).
        #      Find the set of words that become uncovered (they were covered only by cw).
        #      Find a codeword cw2 (not in set) that covers all these words.
        #      If found, add cw2.
        
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            
            # Pass 1: Removal
            # We can vectorize the check for removability somewhat, but let's stick to loop for clarity and low constant overhead.
            # Actually, checking min(cover_count[neighbors[cw]]) for all cw is O(N * 17).
            # min_counts = np.min(cover_count[neighbors], axis=1)
            # removable_mask = min_counts > 1
            # This is very fast with NumPy.
            
            min_counts = np.min(cover_count[neighbors], axis=1)
            
            # We only care about indices in 'selected'
            # Since selected is a list, we can just iterate.
            # But if we remove many, we should update selected.
            
            removed_any = False
            # Iterate through selected in random order to break symmetry
            indices_to_check = list(range(len(selected)))
            np.random.shuffle(indices_to_check)
            
            for i in indices_to_check:
                cw = selected[i]
                if min_counts[cw] > 1:
                    # Remove it
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    changed = True
                    removed_any = True
                    break # Restart loop to recompute min_counts
            
            if not changed:
                # No removals possible. Try Swaps.
                # A swap is beneficial if we can replace 1 codeword with 1 codeword and reduce total size?
                # No, swap keeps size same. We need to find a sequence of swaps that enables a removal.
                # Or, we can look for a "2-for-1" improvement? Unlikely.
                
                # Standard approach: 
                # If stuck, try to remove a random codeword (even if it breaks coverage), 
                # then greedily repair. This is a perturbation.
                
                if time.time() < deadline - 10.0 and len(selected) > 486:
                    # Perturbation: Remove a random codeword, then repair.
                    # Save state
                    orig_selected = selected.copy()
                    orig_cover_count = cover_count.copy()
                    
                    # Pick a random codeword to remove
                    idx_to_remove = np.random.randint(len(selected))
                    cw_removed = selected.pop(idx_to_remove)
                    cover_count[neighbors[cw_removed]] -= 1
                    
                    # Now we have uncovered words (cover_count == 0)
                    # We need to add codewords to cover them.
                    # Greedy repair:
                    # While there are uncovered words:
                    #   Find best codeword to add.
                    
                    # This might increase size, but hopefully subsequent removals will fix it.
                    # Or, we can try to find a single codeword that covers all newly uncovered words?
                    # The set of newly uncovered words is exactly the ball of cw_removed that had count 1.
                    # Let U = {w | cover_count[w] == 0}
                    # We need a cw2 such that U is subset of ball(cw2).
                    # This is a strong condition.
                    
                    # Let's try the greedy repair.
                    # It's risky but can escape local minima.
                    
                    # Check if we can just add back cw_removed? That's a no-op.
                    # We want a different one.
                    
                    # Let's try to find a replacement.
                    # U = words with cover_count 0.
                    # For each candidate cw2 not in selected:
                    #   if U is subset of ball(cw2):
                    #       add cw2, break.
                    
                    U = np.where(cover_count == 0)[0]
                    
                    if len(U) > 0:
                        # Find candidates that cover all of U
                        # This is O(N * |U|) which is too slow if we do it for all N.
                        # But |U| is small (at most 17).
                        # We can check: for a candidate cw2, does it cover all words in U?
                        # We can precompute a "covers" bitset? Or just check.
                        
                        # Optimization:
                        # Only candidates that cover U[0] can possibly cover all of U.
                        # Let's get the list of candidates that cover U[0].
                        # We don't have reverse neighbors precomputed in a list form for fast lookup.
                        # We can compute it on the fly?
                        # Or just iterate through all candidates? 6561 * 17 checks = ~100k ops. Fast enough.
                        
                        replacement_found = False
                        for cw2 in range(total):
                            if cw2 == cw_removed:
                                continue
                            if cw2 in selected:
                                continue
                            
                            # Check if cw2 covers all words in U
                            # We can check if U is subset of neighbors[cw2]
                            # Efficient check:
                            # Create a set of neighbors[cw2]? No, slow.
                            # Just check if all w in U are in neighbors[cw2].
                            # Since |U| is small, this is fast.
                            
                            # We can use a boolean array for neighbors[cw2]?
                            # No, let's just loop.
                            
                            covers_all = True
                            for w in U:
                                # Check if w is in neighbors[cw2]
                                # We can use np.isin? Or just check if w == neighbors[cw2, 0] or w is in neighbors[cw2, 1:]
                                # Since neighbors[cw2] is a small array, a linear search or simple check is fine.
                                # Actually, we can precompute a "membership" matrix?
                                # membership[w, cw] = 1 if w in ball(cw).
                                # This is 6561 x 6561 matrix. 43 million bytes. 43MB.
                                # This might be acceptable and make swaps much faster.
                                # But let's avoid it for now to keep memory low.
                                
                                # Simple check:
                                # w is in ball(cw2) if w == cw2 or dist(w, cw2) == 1.
                                # We can compute dist quickly?
                                # Or just check if w is in the small list neighbors[cw2].
                                # neighbors[cw2] is an array of 17 ints.
                                # `if w in neighbors[cw2]:` is O(17).
                                
                                if w not in neighbors[cw2]:
                                    covers_all = False
                                    break
                            
                            if covers_all:
                                # Found a replacement
                                selected.append(cw2)
                                cover_count[neighbors[cw2]] += 1
                                replacement_found = True
                                break
                        
                        if not replacement_found:
                            # No single replacement found.
                            # Greedy repair: add best codewords to cover U.
                            # This might increase size.
                            # Let's just add the best one.
                            # Count coverage of U for all candidates.
                            # counts = (cover_count[neighbors] == 0).sum(axis=1) # No, we want to cover U specifically.
                            # We want to maximize coverage of U.
                            # u_mask = np.zeros(total, dtype=bool)
                            # u_mask[U] = True
                            # counts = u_mask[neighbors].sum(axis=1)
                            # best_cw = np.argmax(counts)
                            # if counts[best_cw] > 0:
                            #    add best_cw
                            #    repeat until U is empty.
                            
                            # This is getting complex. Let's just revert if no replacement found.
                            selected = orig_selected
                            cover_count = orig_cover_count
                            # Continue to next iteration of the while loop (which will break as changed is False)
                            # Actually, we should break out of the local search to try a new greedy run.
                            changed = False
                            break
                    else:
                        # U is empty, meaning the removal didn't break coverage?
                        # That means the codeword was removable, which contradicts the removal pass.
                        # Unless we removed it and it was redundant.
                        # In that case, we improved!
                        pass
        # End Local Search

        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
        if best_size <= 486:
            # We found a very good solution.
            # We can stop early.
            break

    # Convert best_set to list of digit lists
    if best_set is None:
        # Fallback
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
