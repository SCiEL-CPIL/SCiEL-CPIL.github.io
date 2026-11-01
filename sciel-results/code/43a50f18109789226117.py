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
    
    # Precompute reverse neighbors: for each word w, which codewords cover it?
    # reverse_neighbors[w] = list of codewords cw such that w is in ball(cw)
    # This is the same as neighbors but transposed conceptually.
    # Actually, since the ball relation is symmetric (w in ball(cw) iff cw in ball(w) for radius 1? No!)
    # Wait, Hamming distance is symmetric, so w in ball(cw) iff cw in ball(w).
    # So reverse_neighbors is just the same as neighbors! The adjacency is symmetric.
    # So we don't need a separate reverse neighbors structure.
    
    best_size = total
    best_set = None
    
    # We will run multiple iterations of Greedy + Local Search (Removal + Swap)
    # The local search with swaps is key to escaping local optima and reaching 486.
    
    while time.time() < deadline - 2.0:
        # 1. Randomized Greedy Construction
        # To get diversity, we can randomize the tie-breaking in greedy.
        
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
        
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            
            # Pass 1: Removal
            min_counts = np.min(cover_count[neighbors], axis=1)
            
            # Find removable codewords in selected
            removed = False
            for i in range(len(selected)):
                cw = selected[i]
                if min_counts[cw] > 1:
                    # Remove it
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    changed = True
                    removed = True
                    break  # Restart loop to recompute min_counts
            
            if not changed:
                # No removals possible. Try perturbation.
                if time.time() < deadline - 10.0 and len(selected) > 486:
                    # Perturbation: Remove a random codeword, then greedily repair.
                    orig_selected = selected.copy()
                    orig_cover_count = cover_count.copy()
                    
                    # Pick a random codeword to remove
                    idx_to_remove = np.random.randint(len(selected))
                    cw_removed = selected.pop(idx_to_remove)
                    cover_count[neighbors[cw_removed]] -= 1
                    
                    # Now we have uncovered words (cover_count == 0)
                    U = np.where(cover_count == 0)[0]
                    
                    if len(U) > 0:
                        # Try to find a single replacement first
                        replacement_found = False
                        for cw2 in range(total):
                            if cw2 == cw_removed:
                                continue
                            if cw2 in selected:
                                continue
                            
                            # Check if cw2 covers all words in U
                            covers_all = True
                            for w in U:
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
                            # Greedy repair: add best codewords to cover U
                            repair_count = 0
                            while repair_count < 5:  # Limit repairs
                                u_mask = np.zeros(total, dtype=bool)
                                u_mask[U] = True
                                counts = u_mask[neighbors].sum(axis=1)
                                best_cw = np.argmax(counts)
                                if counts[best_cw] == 0:
                                    break
                                if best_cw in selected:
                                    # No good candidate, revert
                                    break
                                selected.append(best_cw)
                                cover_count[neighbors[best_cw]] += 1
                                # Update U
                                U = np.where(cover_count == 0)[0]
                                if len(U) == 0:
                                    break
                                repair_count += 1
                            
                            # If we added more than we removed, revert
                            if len(selected) > len(orig_selected):
                                selected = orig_selected
                                cover_count = orig_cover_count
                                changed = False
                                break
                    else:
                        # U is empty, removal was safe
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
