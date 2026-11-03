import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 148.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor lists (word -> list of words within Hamming distance <= radius)
    # For radius 1, this is just the word itself + neighbors at distance 1
    neighbor_lists = []
    for i in range(total_space):
        neighbors = [i]
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                # Calculate index of neighbor
                # i = d_0 * 1 + d_1 * q + ...
                # neighbor = ... + new_val * powers[pos] + ...
                # diff = (new_val - digit_val) * powers[pos]
                neighbor_idx = i + (new_val - digit_val) * powers[pos]
                neighbors.append(neighbor_idx)
        neighbor_lists.append(neighbors)
    
    # For faster greedy, precompute masks? 
    # With total_space=2187, a mask is an int of ~2187 bits. 
    # Python big ints are okay, but let's try to use sets or numpy arrays for coverage if needed.
    # However, for greedy, we need to know how many NEW points a word covers.
    # We can maintain a `covered` set.
    
    best_size = total_space
    best_selected = list(range(total_space))

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def get_counts(selected):
        """Compute coverage counts for a list of selected codewords."""
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            for idx in neighbor_lists[w]:
                counts[idx] += 1
        return counts

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively."""
        selected = selected[:]
        if not selected:
            return selected

        counts = get_counts(selected)

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]

                removable = True
                for pos in neighbor_lists[w]:
                    if counts[pos] == 1:
                        removable = False
                        break

                if removable:
                    for pos in neighbor_lists[w]:
                        counts[pos] -= 1
                    selected.pop(idx)
                    changed = True

        return selected

    def local_search(selected, time_limit):
        """
        Local search with 1-for-1 swaps.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = get_counts(selected)
        selected_set = set(selected)

        # 1-for-1 swap and removal
        for _ in range(50): # Reduced iterations, each is fast
            if time.time() > time_limit:
                break
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]

                # Find unique positions: covered only by w
                unique_positions = []
                for pos in neighbor_lists[w]:
                    if counts[pos] == 1:
                        unique_positions.append(pos)

                if not unique_positions:
                    # Can remove w
                    for pos in neighbor_lists[w]:
                        counts[pos] -= 1
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    continue

                # Find best replacement w2 that covers all unique_positions
                # We need w2 such that for all p in unique_positions, p is in neighbor_lists[w2]
                # And w2 should cover some other points to be useful (gain > 0)
                
                # Optimization: Check candidates from a small set of "good" words
                # Instead of checking all 2187 words, we can check words that are neighbors of unique positions?
                # Actually, any word w2 must be in the intersection of the "coverers" of unique_positions.
                # Let's just iterate through all words but break early if possible?
                # 2187 is small.
                
                best_gain = -1
                best_word = -1
                
                # Pre-compute intersection of coverers? 
                # coverers[p] = {w | p in neighbor_lists[w]}
                # We can precompute coverers globally.
                
                # To avoid global precompute overhead in every call, we do it once outside.
                # But for now, let's just iterate.
                
                for w2 in range(total_space):
                    if w2 in selected_set:
                        continue
                    
                    # Check if w2 covers all unique positions
                    valid = True
                    for pos in unique_positions:
                        if pos not in neighbor_lists[w2]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Calculate gain: number of points covered by w2 that are NOT covered by w
                    # Gain = |N(w2) \ N(w)|
                    # We can approximate or calculate exactly.
                    # Exact calculation:
                    gain = 0
                    for pos in neighbor_lists[w2]:
                        if pos not in neighbor_lists[w]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w2

                if best_word >= 0 and best_gain > 0:
                    for pos in neighbor_lists[w]:
                        counts[pos] -= 1
                    for pos in neighbor_lists[best_word]:
                        counts[pos] += 1

                    selected[idx] = best_word
                    selected_set.remove(w)
                    selected_set.add(best_word)
                    improved = True

            if not improved:
                break

        return selected

    # Precompute coverers for faster local search candidate generation
    # coverers[p] = list of w such that p in neighbor_lists[w]
    coverers = [[] for _ in range(total_space)]
    for w in range(total_space):
        for p in neighbor_lists[w]:
            coverers[p].append(w)

    # Modify local_search to use coverers
    def local_search_fast(selected, time_limit):
        selected = selected[:]
        if not selected:
            return selected

        counts = get_counts(selected)
        selected_set = set(selected)

        for _ in range(50):
            if time.time() > time_limit:
                break
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]

                unique_positions = []
                for pos in neighbor_lists[w]:
                    if counts[pos] == 1:
                        unique_positions.append(pos)

                if not unique_positions:
                    for pos in neighbor_lists[w]:
                        counts[pos] -= 1
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    continue

                # Candidates must be in intersection of coverers[unique_positions]
                # Start with coverers of first unique position
                candidates = set(coverers[unique_positions[0]])
                for pos in unique_positions[1:]:
                    candidates &= set(coverers[pos])
                
                # Filter out already selected
                candidates -= selected_set
                
                best_gain = -1
                best_word = -1
                
                for w2 in candidates:
                    # Calculate gain
                    gain = 0
                    # Optimization: if len(candidates) is small, this is fast
                    for pos in neighbor_lists[w2]:
                        if pos not in neighbor_lists[w]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w2

                if best_word >= 0 and best_gain > 0:
                    for pos in neighbor_lists[w]:
                        counts[pos] -= 1
                    for pos in neighbor_lists[best_word]:
                        counts[pos] += 1

                    selected[idx] = best_word
                    selected_set.remove(w)
                    selected_set.add(best_word)
                    improved = True

            if not improved:
                break

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover."""
        covered = set()
        selected = []
        selected_set = set()

        while time.time() < deadline - 8:
            # Check if all covered
            if len(covered) == total_space:
                break
                
            best_gain = -1
            best_word = -1

            for w in range(total_space):
                if w in selected_set:
                    continue
                # Gain = number of new points covered
                gain = 0
                for pos in neighbor_lists[w]:
                    if pos not in covered:
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
            for pos in neighbor_lists[best_word]:
                covered.add(pos)

        return selected

    # Strategy 1: Standard greedy + polish + local search
    if time.time() < deadline - 20:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 20)
        ls = local_search_fast(polished[:], deadline - 20)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Iterated Local Search
    # Perturb best solution and repair
    current_best = best_selected[:]
    
    for _ in range(100):
        if time.time() > deadline - 10:
            break
            
        # Perturb: remove k random codewords
        k = random.randint(2, 5)
        perturbed = current_best[:]
        random.shuffle(perturbed)
        # Remove k
        for _ in range(k):
            if perturbed:
                perturbed.pop()
        
        # Repair: Greedy cover the remaining uncovered points
        # First, calculate current coverage of perturbed
        counts = get_counts(perturbed)
        covered = set(i for i in range(total_space) if counts[i] > 0)
        
        # Add greedy words to cover the rest
        selected_set = set(perturbed)
        
        while len(covered) < total_space and time.time() < deadline - 10:
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                if w in selected_set:
                    continue
                gain = 0
                for pos in neighbor_lists[w]:
                    if pos not in covered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word < 0:
                break
            perturbed.append(best_word)
            selected_set.add(best_word)
            for pos in neighbor_lists[best_word]:
                covered.add(pos)
                
        # Polish and Local Search
        polished = polish(perturbed[:], deadline - 10)
        ls = local_search_fast(polished[:], deadline - 10)
        
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]
            current_best = ls[:]
            
    # Final polish
    if time.time() < deadline - 2:
        polished = polish(best_selected[:], deadline - 2)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]
            
    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
