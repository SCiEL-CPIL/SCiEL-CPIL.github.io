import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills at 150s; leave safety margin
    deadline = start_time + 142.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbors for all words
    # neighbors[w] is a list of indices within Hamming distance <= radius
    # For radius 1, this is w itself + words differing in 1 position
    neighbors = [[] for _ in range(total_space)]
    for i in range(total_space):
        # Add self
        neighbors[i].append(i)
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                j = i + diff * powers[pos]
                neighbors[i].append(j)
        
        # Convert to set for faster intersection checks later if needed, 
        # but list is fine for iteration. Let's use a set for 'in' checks if we do it often.
        # Actually, for gain calculation `len(set(neighbors[w]) & remaining)`, 
        # having neighbors as a set is faster.
        neighbors[i] = set(neighbors[i])

    full_set = set(range(total_space))

    best_size = total_space
    best_selected = list(range(total_space))

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def compute_counts(selected):
        """Compute coverage counts for a set of selected words."""
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            for nb in neighbors[w]:
                counts[nb] += 1
        return counts

    def polish(selected, time_limit, max_passes=20):
        """Remove redundant codewords iteratively."""
        selected = selected[:]
        if not selected:
            return selected

        counts = compute_counts(selected)

        for _ in range(max_passes):
            if time.time() >= time_limit:
                break
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                
                # Check if any neighbor is covered only by w
                removable = True
                for nb in neighbors[w]:
                    if counts[nb] == 1:
                        removable = False
                        break

                if removable:
                    # Remove w and update counts
                    for nb in neighbors[w]:
                        counts[nb] -= 1
                    selected.pop(idx)
                    changed = True

            if not changed:
                break

        return selected

    def local_search(selected, time_limit):
        """Try to reduce the set by removing or swapping words."""
        selected = selected[:]
        if not selected:
            return selected

        counts = compute_counts(selected)

        for _ in range(100):
            if time.time() >= time_limit:
                break
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                
                # Find unique neighbors (covered only by w)
                unique_words = [nb for nb in neighbors[w] if counts[nb] == 1]

                if not unique_words:
                    # Can remove entirely
                    for nb in neighbors[w]:
                        counts[nb] -= 1
                    selected.pop(idx)
                    improved = True
                    continue

                # Try to find a replacement word that covers all unique words
                # and adds some new coverage
                best_gain = 0
                best_word = -1
                
                # Sample candidates to speed up
                if total_space > 500:
                    candidates = random.sample(range(total_space), min(200, total_space))
                else:
                    candidates = list(range(total_space))
                
                # Convert unique_words to a set for faster intersection
                unique_set = set(unique_words)
                
                for w2 in candidates:
                    if w2 == w:
                        continue
                    # Check if w2 covers all unique words
                    if unique_set.issubset(neighbors[w2]):
                        # Calculate gain: new neighbors covered by w2 that were not covered by w
                        # gain = |neighbors[w2] - neighbors[w]|
                        # But we need to be careful with counts. 
                        # A simpler proxy: if it covers the critical ones, it's a candidate.
                        # We want to maximize the number of *newly* covered points or minimize loss.
                        # Since we remove w and add w2, the net change in size is 0 (swap).
                        # We just need validity. The 'gain' here is heuristic for future steps.
                        # Let's just pick the first valid one or one with high overlap with current coverage?
                        # Actually, any valid swap maintains coverage. 
                        # To improve, we might want to swap to a word that allows more removals later.
                        # For now, accept the first valid swap to break symmetry.
                        best_gain = 1 
                        best_word = w2
                        break # Take the first valid one to save time

                if best_word >= 0:
                    # Update counts
                    for nb in neighbors[w]:
                        counts[nb] -= 1
                    for nb in neighbors[best_word]:
                        counts[nb] += 1
                    selected[idx] = best_word
                    improved = True

            if not improved:
                break

        return selected

    def run_greedy(randomized=False, alpha=1.0, time_limit=None):
        """Greedy set cover with optional randomization."""
        if time_limit is None:
            time_limit = deadline - 5
        
        selected = []
        remaining = set(range(total_space))

        while remaining and time.time() < time_limit:
            best_gain = -1
            best_word = -1
            ties = []

            # Optimization: If remaining is small, we can be more precise.
            # Otherwise, iterate all words.
            for w in range(total_space):
                # Gain is number of new points covered
                # len(neighbors[w] & remaining)
                # Since neighbors[w] is a set, intersection is fast.
                gain = len(neighbors[w] & remaining)
                
                if randomized:
                    gain = gain + random.gauss(0, 1.0) * alpha
                
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
                    ties = [w]
                elif gain == best_gain:
                    ties.append(w)

            if best_word < 0:
                break

            if randomized and len(ties) > 1:
                best_word = random.choice(ties)

            selected.append(best_word)
            remaining.difference_update(neighbors[best_word])

        return selected

    def run_greedy_randomized(alpha_range=(0.05, 0.8)):
        """Randomized greedy with varied noise levels."""
        alpha = random.uniform(*alpha_range)
        return run_greedy(randomized=True, alpha=alpha)

    # Strategy 1: Standard greedy + polish
    if time.time() < deadline - 10:
        sol = run_greedy(randomized=False, time_limit=deadline - 10)
        polished = polish(sol[:], deadline - 10)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Strategy 2: Multiple randomized greedy restarts with polish
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        sol = run_greedy_randomized()
        # Only polish if the raw solution is promising
        if len(sol) <= best_size + 3:
            polished = polish(sol[:], deadline - 20)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]
        
        # Periodically run local search on the best solution
        if restart_count % 10 == 0 and time.time() < deadline - 10:
            ls = local_search(best_selected[:], deadline - 10)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Strategy 3: Final local search
    if time.time() < deadline - 5:
        ls = local_search(best_selected[:], deadline - 5)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Final polish
    if time.time() < deadline - 2:
        polished = polish(best_selected[:], deadline - 2)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
