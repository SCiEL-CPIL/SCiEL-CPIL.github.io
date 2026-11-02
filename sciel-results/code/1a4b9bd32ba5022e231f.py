import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor bitmasks for all words
    # masks[i] is a Python int with bits set for all words within distance 1 of i
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                m |= (1 << (i + diff * powers[pos]))
        masks[i] = m

    full_mask = (1 << total_space) - 1

    best_size = total_space
    best_selected = list(range(total_space))

    # Convert integer index to tuple
    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    # Polish: remove redundant words
    def polish(selected, time_limit):
        selected = selected[:]
        if not selected:
            return selected

        # Build counts: for each word position, how many selected codewords cover it
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            m = masks[w]
            # Extract set bits and increment counts
            mm = m
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        # Repeatedly remove redundant words
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                # Check if all words covered by w have count > 1
                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit

                if removable:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    changed = True

        return selected

    # Greedy set cover
    def run_greedy(randomized=False):
        covered = 0
        selected = []
        remaining = full_mask

        while remaining and time.time() < deadline - 5:
            best_gain = -1
            candidates = []

            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [w]
                elif gain == best_gain and randomized:
                    candidates.append(w)

            if not candidates:
                break

            if randomized and len(candidates) > 1:
                best_word = random.choice(candidates)
            else:
                best_word = candidates[0]

            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered

        return selected

    # Run standard greedy
    if time.time() < deadline - 10:
        sol = run_greedy(randomized=False)
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]
            polished = polish(best_selected[:], deadline - 5)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    # Run randomized greedy restarts
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break

        sol = run_greedy(randomized=True)
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]
            polished = polish(best_selected[:], deadline - 5)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    # Final polish on the best solution
    if time.time() < deadline - 3:
        polished = polish(best_selected[:], deadline - 2)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Convert to list of tuples
    best_words = [int_to_tuple(v) for v in best_selected]

    return best_words
