import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 142.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i - 1] * q

    # Precompute cover sets as frozensets
    cover_sets = []
    for i in range(total_space):
        neighbors = {i}
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                neighbors.add(i + diff * powers[pos])
        cover_sets.append(frozenset(neighbors))

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively."""
        if not selected:
            return selected

        # Count coverage
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            for nb in cover_sets[w]:
                counts[nb] += 1

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
                for nb in cover_sets[w]:
                    if counts[nb] == 1:
                        removable = False
                        break
                if removable:
                    for nb in cover_sets[w]:
                        counts[nb] -= 1
                    selected.pop(idx)
                    changed = True

        return selected

    def local_search(selected, time_limit):
        """
        Local search with 1-for-1 swaps and 2-for-1 swaps.
        """
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            for nb in cover_sets[w]:
                counts[nb] += 1

        selected_set = set(selected)

        def is_redundant(w):
            for nb in cover_sets[w]:
                if counts[nb] == 1:
                    return False
            return True

        def remove_word(w):
            for nb in cover_sets[w]:
                counts[nb] -= 1
            selected_set.remove(w)

        def add_word(w):
            for nb in cover_sets[w]:
                counts[nb] += 1
            selected_set.add(w)

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if time.time() > time_limit:
                    break
                if idx >= len(selected):
                    continue

                w_a = selected[idx]

                # 1. Try simple removal
                if is_redundant(w_a):
                    remove_word(w_a)
                    selected.pop(idx)
                    improved = True
                    continue

                # 2. Try 1-for-1 swap
                unique_a = [nb for nb in cover_sets[w_a] if counts[nb] == 1]

                if unique_a:
                    # Find best C that covers all unique_a
                    best_c = -1
                    best_gain = -1
                    for w_c in range(total_space):
                        if w_c in selected_set:
                            continue
                        covers_all = all(nb in cover_sets[w_c] for nb in unique_a)
                        if covers_all:
                            # Gain: how many new words does C cover?
                            gain = sum(1 for nb in cover_sets[w_c] if counts[nb] == 0)
                            if gain > best_gain:
                                best_gain = gain
                                best_c = w_c

                    if best_c != -1:
                        # Perform swap: remove A, add C
                        remove_word(w_a)
                        add_word(best_c)
                        selected[idx] = best_c
                        improved = True

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover with optional randomization."""
        covered = set()
        selected = []
        remaining = set(range(total_space))

        while remaining and time.time() < deadline - 5:
            best_gain = -1
            best_word = -1
            candidates = []

            for w in range(total_space):
                gain = len(cover_sets[w] & remaining)
                if gain == 0:
                    continue
                if randomized:
                    gain = gain + random.gauss(0, 1.0) * alpha
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
                    candidates = [w]
                elif gain == best_gain and not randomized:
                    candidates.append(w)

            if best_word == -1:
                break

            if randomized and len(candidates) > 1:
                best_word = random.choice(candidates)
            else:
                best_word = candidates[0]

            selected.append(best_word)
            covered |= cover_sets[best_word]
            remaining -= covered

        return selected

    best_size = total_space
    best_selected = list(range(total_space))

    # Strategy 1: Standard greedy + polish
    if time.time() < deadline - 10:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 10)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Strategy 2: Multiple randomized greedy restarts with polish
    num_restarts = 150
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break

        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.5))
        if len(sol) < best_size + 5:
            polished = polish(sol[:], deadline - 15)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    # Strategy 3: Local search on best solution
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
