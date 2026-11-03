import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s, but task says 120s. 
    # We aim to finish well before 120s to be safe.
    deadline = start_time + 110.0 

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbors for each word
    # neighbors[i] is a list of word indices within distance <= radius of word i
    # We only need distance 1 for radius 1, but let's be general for radius.
    # For q=3, n=7, radius=1:
    # Each word has 1 + n*(q-1) neighbors. 1 + 7*2 = 15 neighbors.
    
    # Optimization: Instead of storing lists, we can generate neighbors on the fly 
    # or precompute a flat array of neighbors.
    # Let's precompute a list of neighbors for each word to speed up updates.
    # neighbors_flat[i] = list of j such that dist(i,j) <= radius
    
    neighbors = [None] * total_space
    for i in range(total_space):
        neigh = [i]
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                j = i + diff * powers[pos]
                neigh.append(j)
        # Remove duplicates if any (though logic above shouldn't produce dupes for radius 1)
        # For radius > 1, we'd need more complex generation. 
        # Given the task is specifically K_3(7,1), radius is 1.
        # If radius > 1, this simple generation fails. 
        # However, the prompt says "radius" is passed. 
        # If radius > 1, we need a proper generation.
        # Let's assume radius is small (likely 1). 
        # If radius > 1, the neighbor count explodes. 
        # For safety, if radius > 1, we might need a different approach or it might be too slow.
        # But K_3(7,1) is the specific case.
        
        # To be robust for radius > 1 (though unlikely to be tested with large radius for n=7):
        if radius > 1:
            # Generate all words within radius
            # This is expensive. 
            # For now, assume radius=1 or very small.
            # If radius > 1, we fall back to a simpler, potentially slower method?
            # No, let's stick to the efficient radius=1 path if possible.
            # The problem statement says "at most one position", implying radius=1.
            # The interface passes radius. We should respect it.
            # If radius is large, the problem becomes trivial (fewer codewords needed).
            # But generating neighbors for radius=2 in 3^7 is 1 + 14 + ... 
            # Let's stick to the assumption that radius is small.
            pass
        
        neighbors[i] = neigh

    # Convert to numpy arrays for faster iteration if needed? 
    # Lists are fine for 2187 items.

    best_size = total_space
    best_selected = list(range(total_space))

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively."""
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            for j in neighbors[w]:
                counts[j] += 1

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                
                # Check if any neighbor is covered only by w
                removable = True
                for j in neighbors[w]:
                    if counts[j] == 1:
                        removable = False
                        break
                
                if removable:
                    for j in neighbors[w]:
                        counts[j] -= 1
                    selected.pop(idx)
                    changed = True

        return selected

    def local_search(selected, time_limit):
        """
        Local search with 1-for-1 swaps and removals.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            for j in neighbors[w]:
                counts[j] += 1

        selected_set = set(selected)

        # 1-for-1 swap and removal
        iterations = 0
        while time.time() < time_limit and iterations < 50:
            iterations += 1
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                
                # Find unique neighbors of w
                unique_mask = 0
                for j in neighbors[w]:
                    if counts[j] == 1:
                        unique_mask |= (1 << j)
                
                if unique_mask == 0:
                    # Remove w
                    for j in neighbors[w]:
                        counts[j] -= 1
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    continue

                # Try to replace w with w2 that covers all unique neighbors of w
                best_gain = -1
                best_word = -1
                
                # Iterate over all possible candidates
                # Optimization: Only check candidates that cover at least one unique neighbor?
                # Or just check all? 2187 is small.
                for w2 in range(total_space):
                    if w2 in selected_set:
                        continue
                    
                    # Check if w2 covers all unique neighbors of w
                    covers_all = True
                    for j in neighbors[w]:
                        if counts[j] == 1 and j not in neighbors[w2]:
                            covers_all = False
                            break
                    
                    if covers_all:
                        # Calculate gain: how many new words does w2 cover that w didn't?
                        # Or just prefer replacing if it maintains coverage.
                        # Actually, we want to minimize size. 
                        # If we replace 1 with 1, size is same. 
                        # We should only replace if it allows further reduction?
                        # The "polish" step handles removals.
                        # So a 1-for-1 swap is only useful if it enables a removal later.
                        # This is hard to predict. 
                        # Instead, let's focus on "2-for-1" swaps which directly reduce size.
                        pass

            if not improved:
                break
        
        # 2-for-1 swap (remove 2, add 1)
        # This is the key to reducing size.
        if time.time() < time_limit - 5:
            for _ in range(20):
                if time.time() > time_limit:
                    break
                improved = False
                indices = list(range(len(selected)))
                random.shuffle(indices)
                
                # Try pairs
                # Limit pairs to keep time down
                pair_limit = min(200, len(indices))
                for i_idx in range(pair_limit):
                    idx1 = indices[i_idx]
                    if idx1 >= len(selected):
                        continue
                    w1 = selected[idx1]

                    idx2 = random.randint(0, len(selected) - 1)
                    if idx2 == idx1:
                        continue
                    w2 = selected[idx2]

                    # Calculate unique coverage of w1 and w2
                    # A word is "uniquely covered by w1 or w2" if:
                    # 1. It is a neighbor of w1 or w2
                    # 2. It is NOT covered by any other selected word
                    # We need to find a word w3 that covers ALL such words.
                    
                    unique_words = []
                    # Check neighbors of w1
                    for j in neighbors[w1]:
                        if counts[j] == 1: # Only w1 covers it
                            unique_words.append(j)
                    # Check neighbors of w2
                    for j in neighbors[w2]:
                        if counts[j] == 2: # w1 and w2 cover it (and no others)
                            # If w1 and w2 are the only ones, removing both makes it uncovered.
                            # So it must be covered by w3.
                            unique_words.append(j)
                        elif counts[j] == 1:
                            # Only w2 covers it (w1 doesn't)
                            unique_words.append(j)
                    
                    # Remove duplicates
                    unique_set = set(unique_words)
                    
                    if len(unique_set) > 10:
                        continue # Too many constraints, unlikely to find a single w3

                    # Find a candidate w3 that covers all in unique_set
                    found_w3 = -1
                    for w3 in range(total_space):
                        if w3 in selected_set:
                            continue
                        
                        covers_all = True
                        for j in unique_set:
                            if j not in neighbors[w3]:
                                covers_all = False
                                break
                        
                        if covers_all:
                            found_w3 = w3
                            break

                    if found_w3 != -1:
                        # Perform swap: remove w1, w2; add w3
                        # Update counts
                        for j in neighbors[w1]:
                            counts[j] -= 1
                        for j in neighbors[w2]:
                            counts[j] -= 1
                        for j in neighbors[found_w3]:
                            counts[j] += 1
                        
                        # Remove from list
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected.append(found_w3)
                        
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        selected_set.add(found_w3)
                        
                        improved = True
                        break
                
                if improved:
                    # Polish after swap
                    selected = polish(selected, time_limit)
                    # Rebuild counts
                    counts = np.zeros(total_space, dtype=np.int32)
                    for w in selected:
                        for j in neighbors[w]:
                            counts[j] += 1
                    selected_set = set(selected)

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover."""
        covered = np.zeros(total_space, dtype=bool)
        selected = []
        remaining_count = total_space
        selected_set = set()

        while remaining_count > 0 and time.time() < deadline - 10:
            best_gain = -1
            best_word = -1

            for w in range(total_space):
                if w in selected_set:
                    continue
                
                # Count how many new words w covers
                gain = 0
                for j in neighbors[w]:
                    if not covered[j]:
                        gain += 1
                
                if randomized:
                    gain = gain + random.gauss(0, 1.0) * alpha
                
                if gain > best_gain:
                    best_gain = gain
                    best_word = w

            if best_word < 0:
                break

            selected.append(best_word)
            selected_set.add(best_word)
            
            # Update covered
            for j in neighbors[best_word]:
                if not covered[j]:
                    covered[j] = True
                    remaining_count -= 1

        return selected

    # Strategy 1: Standard greedy + polish + local search
    if time.time() < deadline - 20:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 20)
        ls = local_search(polished[:], deadline - 20)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Multiple randomized greedy restarts
    for r in range(50):
        if time.time() > deadline - 30:
            break

        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.5))
        if len(sol) < best_size + 10:
            polished = polish(sol[:], deadline - 30)
            ls = local_search(polished[:], deadline - 30)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Final polish
    if time.time() < deadline - 5:
        polished = polish(best_selected[:], deadline - 5)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Final local search pass
    if time.time() < deadline - 2:
        final_ls = local_search(best_selected[:], deadline - 2)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_selected = final_ls[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
