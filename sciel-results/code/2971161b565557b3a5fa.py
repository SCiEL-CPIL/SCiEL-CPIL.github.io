import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    total_space = q ** n

    # Precompute positional powers for index conversion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    # Precompute neighbors for radius 1
    # neighbor_count = 1 + n * (q - 1)
    neighbor_count = 1 + n * (q - 1)
    neighbors = np.zeros((total_space, neighbor_count), dtype=np.int32)
    
    # Also precompute the mask (bitmask) for fast intersection checks if needed, 
    # but we will rely on the neighbors array for counts.
    # We need to map index to its neighbors.
    for i in range(total_space):
        # Get digits
        digits = [0] * n
        temp = i
        for pos in range(n):
            digits[pos] = (temp // powers[pos]) % q
            temp //= powers[pos]
        
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                # Calculate new index
                # index = sum(d_j * q^j)
                # new_index = i - orig * powers[pos] + new_val * powers[pos]
                new_idx = i - orig * powers[pos] + new_val * powers[pos]
                neighbors[i, count] = new_idx
                count += 1

    # Precompute reverse cover: for each point x, which words cover it?
    # Since the graph is regular and symmetric, reverse_cover[x] is the same set as neighbors[x] 
    # IF the distance metric is symmetric (which Hamming is).
    # So reverse_cover[x] = neighbors[x].
    # We can just use neighbors for both.
    
    # Initial Greedy Construction
    def run_greedy(randomized=False, alpha=1.0):
        selected = []
        selected_set = set()
        remaining = np.ones(total_space, dtype=np.uint8)
        
        while remaining.any() and time.time() < deadline - 10:
            # Gain: number of remaining points covered by each word
            # gain[w] = sum(remaining[neighbors[w]])
            gains = np.sum(remaining[neighbors], axis=1).astype(np.int32)
            
            # Mask out already selected
            if selected:
                sel_arr = np.array(selected, dtype=np.int32)
                gains[sel_arr] = -1
            
            if randomized:
                noise = np.random.normal(0, alpha, size=total_space).astype(np.int32)
                gains = gains + noise
            
            best_word = int(np.argmax(gains))
            if gains[best_word] <= 0 and not randomized:
                break
                
            selected.append(best_word)
            selected_set.add(best_word)
            
            # Update remaining
            remaining[neighbors[best_word]] = 0
            
        return selected

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively."""
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            counts[neighbors[w]] += 1

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                # Check if any point in w's neighborhood has count == 1
                removable = True
                for pos in neighbors[w]:
                    if counts[pos] == 1:
                        removable = False
                        break
                
                if removable:
                    for pos in neighbors[w]:
                        counts[pos] -= 1
                    selected.pop(idx)
                    changed = True

        return selected

    def local_search(selected, time_limit):
        """
        Local search with 1-for-1 swaps and 2-for-1 swaps.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            counts[neighbors[w]] += 1
        
        selected_set = set(selected)

        # 1-for-1 swap
        max_passes = 100
        for p in range(max_passes):
            if time.time() > time_limit:
                break
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                
                # Identify unique points covered by w
                unique_points = []
                for pos in neighbors[w]:
                    if counts[pos] == 1:
                        unique_points.append(pos)

                if not unique_points:
                    # w is redundant
                    for pos in neighbors[w]:
                        counts[pos] -= 1
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    break # Restart shuffle loop? Or just continue? 
                    # Breaking out of idx loop, but 'improved' is True so we restart passes.
                    # Actually, just continue to next idx or break?
                    # If we remove one, counts change. Better to restart the pass.
                    # But for speed, we can just continue and let polish clean up later?
                    # Let's break to restart the pass for consistency.
                    pass 

                if improved:
                    break

                # Find candidates that cover all unique_points
                if not unique_points:
                    continue

                candidates = None
                for pt in unique_points:
                    # Candidates must be in neighbors[pt]
                    if candidates is None:
                        candidates = set(neighbors[pt].tolist())
                    else:
                        candidates.intersection_update(neighbors[pt].tolist())
                        if not candidates:
                            break

                if not candidates:
                    continue

                # Remove w from candidates
                candidates.discard(w)
                if not candidates:
                    continue

                best_gain = -1
                best_word = -1
                
                # We want a word that covers the unique points AND adds new coverage
                # Gain = number of points covered by w2 that are NOT covered by w
                # Actually, since we are swapping, the "new" points are those in neighbors[best_word] 
                # that are not in neighbors[w].
                # But we also care about how much it overlaps with others to keep counts high?
                # Standard heuristic: Maximize (neighbors[best_word] & ~neighbors[w]).size
                w_neighbors_set = set(neighbors[w].tolist())
                
                for w2 in candidates:
                    # Calculate gain
                    # Vectorized way might be hard for single candidate in loop.
                    # Just count intersection size.
                    # gain = len(set(neighbors[w2]) - w_neighbors_set)
                    # To speed up, we can precompute neighbor sets? 
                    # No, just use numpy setdiff1d or manual count.
                    # Since neighbor_count is small (15), manual count is fast.
                    gain = 0
                    for pos in neighbors[w2]:
                        if pos not in w_neighbors_set:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w2

                if best_word >= 0 and best_gain > 0:
                    # Perform swap
                    for pos in neighbors[w]:
                        counts[pos] -= 1
                    for pos in neighbors[best_word]:
                        counts[pos] += 1
                    
                    selected[idx] = best_word
                    selected_set.remove(w)
                    selected_set.add(best_word)
                    improved = True
                    break # Restart pass

        # 2-for-1 swap
        # Try to remove 2 codewords and add 1
        if time.time() < time_limit - 5:
            for _ in range(50): # Max 50 attempts
                if time.time() > time_limit:
                    break
                improved = False
                if len(selected) < 2:
                    break
                
                # Pick two random distinct indices
                idx1 = random.randint(0, len(selected) - 1)
                idx2 = random.randint(0, len(selected) - 1)
                if idx1 == idx2:
                    continue
                
                w1 = selected[idx1]
                w2 = selected[idx2]
                
                # Calculate gap: points that would become uncovered if we remove w1 and w2
                # A point is in the gap if:
                # 1. It is in neighbors[w1] or neighbors[w2]
                # 2. After removing w1 and w2, its count becomes 0.
                
                # Temporarily decrement counts for w1 and w2
                for pos in neighbors[w1]:
                    counts[pos] -= 1
                for pos in neighbors[w2]:
                    counts[pos] -= 1
                
                gap_points = []
                # Check points in neighbors[w1]
                for pos in neighbors[w1]:
                    if counts[pos] == 0:
                        gap_points.append(pos)
                # Check points in neighbors[w2] that are not in neighbors[w1]
                # To avoid duplicates, we can just check all in w2 as well, 
                # but if a point is in both, we already checked it in w1 loop.
                # However, if a point is in both, counts was decremented twice.
                # If counts is 0, it's a gap.
                for pos in neighbors[w2]:
                    if counts[pos] == 0:
                        if pos not in gap_points: # Optimization: avoid dupes
                            gap_points.append(pos)
                
                if not gap_points:
                    # 2-for-0 move! We can just remove w1 and w2.
                    # Undo temporary decrements? No, we keep them.
                    # Just remove from selected.
                    # Ensure we remove larger index first to keep indices valid
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    selected.pop(idx2)
                    selected.pop(idx1)
                    selected_set.remove(w1)
                    selected_set.remove(w2)
                    improved = True
                    break
                else:
                    # Find w3 that covers all gap_points
                    candidates = None
                    for pt in gap_points:
                        if candidates is None:
                            candidates = set(neighbors[pt].tolist())
                        else:
                            candidates.intersection_update(neighbors[pt].tolist())
                            if not candidates:
                                break
                    
                    if candidates:
                        # Remove w1, w2 from candidates? 
                        # w3 must be different from w1, w2? 
                        # If w3 == w1, we just removed it. 
                        # So w3 must not be in selected_set? 
                        # Actually, w3 can be any word. If w3 is in selected, 
                        # we are just moving it? No, we are adding it.
                        # If w3 is already in selected, we are adding a duplicate? 
                        # Codewords must be distinct.
                        # So w3 must not be in selected_set.
                        candidates.difference_update(selected_set)
                        
                        if candidates:
                            # Pick best w3 (max gain? or random?)
                            # Let's pick one that covers most gap points (already ensured) 
                            # and adds new coverage?
                            # For now, random choice or max overlap with existing?
                            # Max overlap with existing might be safer.
                            # Let's just pick the first one for speed, or random.
                            w3 = random.choice(list(candidates))
                            
                            # Apply swap: Remove w1, w2, Add w3
                            # Counts for w1, w2 are already decremented.
                            for pos in neighbors[w3]:
                                counts[pos] += 1
                            
                            if idx1 > idx2:
                                idx1, idx2 = idx2, idx1
                            selected.pop(idx2)
                            selected.pop(idx1)
                            selected.append(w3)
                            
                            selected_set.remove(w1)
                            selected_set.remove(w2)
                            selected_set.add(w3)
                            
                            improved = True
                            break
                    # If no candidate, undo the temporary decrements
                    for pos in neighbors[w1]:
                        counts[pos] += 1
                    for pos in neighbors[w2]:
                        counts[pos] += 1

                if improved:
                    # Polish to clean up
                    selected = polish(selected, time_limit)
                    # Rebuild counts and set
                    counts = np.zeros(total_space, dtype=np.int32)
                    for w in selected:
                        counts[neighbors[w]] += 1
                    selected_set = set(selected)
                    break # Restart 2-for-1 loop

        return selected

    best_size = total_space
    best_selected = list(range(total_space))

    # Strategy 1: Standard greedy + polish + local search
    if time.time() < deadline - 20:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 20)
        ls = local_search(polished[:], deadline - 20)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Multiple randomized greedy restarts
    restarts = 0
    while time.time() < deadline - 30 and restarts < 50:
        restarts += 1
        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.5))
        if len(sol) < best_size + 10:
            polished = polish(sol[:], deadline - 30)
            ls = local_search(polished[:], deadline - 30)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Final polish and local search on the best solution found
    if time.time() < deadline - 5:
        polished = polish(best_selected[:], deadline - 5)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    if time.time() < deadline - 2:
        final_ls = local_search(best_selected[:], deadline - 2)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_selected = final_ls[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
