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

    # Precompute neighbor bitmasks (mask[w] = all words within radius of w)
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

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def polish(selected, time_limit):
        """Remove redundant codewords with O(1) unique-count tracking."""
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        # unique_count[w] = number of neighbors of w that only w covers
        unique_count = {}
        for w in selected:
            mm = masks[w]
            uc = 0
            while mm:
                lowbit = mm & -mm
                pos = lowbit.bit_length() - 1
                if counts[pos] == 1:
                    uc += 1
                mm ^= lowbit
            unique_count[w] = uc

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                if unique_count.get(w, 0) > 0:
                    continue
                # w is redundant: remove it
                mm = masks[w]
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    counts[pos] -= 1
                    mm ^= lowbit
                # update unique counts of neighbors (they only gain uniqueness)
                for other in selected:
                    if other == w:
                        continue
                    if (masks[other] & mm) == 0:
                        continue
                    uc = unique_count.get(other, 0)
                    if uc > 0:
                        continue
                    # other was unique-count 0; recompute
                    mm2 = masks[other]
                    new_uc = 0
                    while mm2:
                        lowbit = mm2 & -mm2
                        p = lowbit.bit_length() - 1
                        if counts[p] == 1:
                            new_uc += 1
                        mm2 ^= lowbit
                    unique_count[other] = new_uc
                selected.pop(idx)
                del unique_count[w]
                changed = True

        return selected

    def local_search(selected, time_limit):
        """1-for-1 swaps plus removal, then 2-for-1 swaps."""
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        selected_set = set(selected)

        for _ in range(80):
            if time.time() > time_limit:
                break
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                unique_mask = 0
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_mask |= lowbit
                    mm ^= lowbit

                if unique_mask == 0:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    continue

                best_gain = -1
                best_word = -1
                for w2 in range(total_space):
                    if w2 in selected_set:
                        continue
                    m2 = masks[w2]
                    if (m2 & unique_mask) == unique_mask:
                        gain = (m2 & ~m).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_word = w2

                if best_word >= 0 and best_gain > 0:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    mm = masks[best_word]
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit
                    selected[idx] = best_word
                    selected_set.remove(w)
                    selected_set.add(best_word)
                    improved = True

            if not improved:
                break

        # 2-for-1 swap
        if time.time() < time_limit - 8:
            for _ in range(15):
                if time.time() > time_limit:
                    break
                improved = False
                indices = list(range(len(selected)))
                random.shuffle(indices)

                for i_idx in range(min(40, len(indices))):
                    idx1 = indices[i_idx]
                    if idx1 >= len(selected):
                        continue
                    w1 = selected[idx1]
                    idx2 = random.randint(0, len(selected) - 1)
                    if idx2 == idx1:
                        continue
                    w2 = selected[idx2]

                    m1 = masks[w1]
                    m2 = masks[w2]

                    u_ab = 0
                    inter = m1 & m2
                    mm = inter
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 2:
                            u_ab |= lowbit
                        mm ^= lowbit
                    diff1 = m1 & ~m2
                    mm = diff1
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u_ab |= lowbit
                        mm ^= lowbit
                    diff2 = m2 & ~m1
                    mm = diff2
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u_ab |= lowbit
                        mm ^= lowbit

                    if u_ab == 0:
                        mm = m1
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        mm = m2
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        improved = True
                        break

                    if u_ab.bit_count() > 8:
                        continue

                    best_word = -1
                    for w3 in range(total_space):
                        if w3 in selected_set:
                            continue
                        m3 = masks[w3]
                        if (m3 & u_ab) == u_ab:
                            best_word = w3
                            break

                    if best_word >= 0:
                        mm = m1
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        mm = m2
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        mm = masks[best_word]
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] += 1
                            mm ^= lowbit
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected.append(best_word)
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        selected_set.add(best_word)
                        improved = True
                        break

                if improved:
                    selected = polish(selected, time_limit)
                    counts = np.zeros(total_space, dtype=np.int32)
                    for w in selected:
                        mm = masks[w]
                        while mm:
                            lowbit = mm & -mm
                            idx = lowbit.bit_length() - 1
                            counts[idx] += 1
                            mm ^= lowbit
                    selected_set = set(selected)

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover using a maintained new-coverage count array."""
        remaining = np.ones(total_space, dtype=np.int32)
        selected = []
        selected_set = set()

        while remaining.any() and time.time() < deadline - 8:
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                if w in selected_set:
                    continue
                mm = masks[w]
                gain = 0
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    gain += remaining[pos]
                    mm ^= lowbit
                if gain == 0:
                    continue
                if randomized:
                    gain = gain + random.gauss(0, 1.0) * alpha
                if gain > best_gain:
                    best_gain = gain
                    best_word = w

            if best_word < 0:
                break

            selected.append(best_word)
            selected_set.add(best_word)
            mm = masks[best_word]
            while mm:
                lowbit = mm & -mm
                pos = lowbit.bit_length() - 1
                remaining[pos] = 0
                mm ^= lowbit

        return selected

    # Strategy 1: Standard greedy + polish + local search
    if time.time() < deadline - 15:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 15)
        ls = local_search(polished[:], deadline - 15)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Multiple randomized greedy restarts
    for r in range(200):
        if time.time() > deadline - 25:
            break
        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.6))
        if len(sol) < best_size + 8:
            polished = polish(sol[:], deadline - 25)
            ls = local_search(polished[:], deadline - 25)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Strategy 3: Two-phase greedy
    if time.time() < deadline - 30:
        remaining = np.ones(total_space, dtype=np.int32)
        covered = 0
        selected = []
        selected_set = set()
        phase1_limit = int(total_space * 0.5)
        while remaining.any() and len(selected) < phase1_limit and time.time() < deadline - 30:
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                if w in selected_set:
                    continue
                mm = masks[w]
                gain = 0
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    gain += remaining[pos]
                    mm ^= lowbit
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word < 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            covered |= masks[best_word]
            mm = masks[best_word]
            while mm:
                lowbit = mm & -mm
                pos = lowbit.bit_length() - 1
                remaining[pos] = 0
                mm ^= lowbit

        while remaining.any() and time.time() < deadline - 30:
            best_score = -1
            best_word = -1
            for w in range(total_space):
                if w in selected_set:
                    continue
                mm = masks[w]
                new_cover = 0
                overlap = 0
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if remaining[pos]:
                        new_cover += 1
                    else:
                        overlap += 1
                    mm ^= lowbit
                if new_cover == 0:
                    continue
                score = new_cover * 10 - overlap
                if score > best_score:
                    best_score = score
                    best_word = w
            if best_word < 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            covered |= masks[best_word]
            mm = masks[best_word]
            while mm:
                lowbit = mm & -mm
                pos = lowbit.bit_length() - 1
                remaining[pos] = 0
                mm ^= lowbit

        polished = polish(selected[:], deadline - 30)
        ls = local_search(polished[:], deadline - 30)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Final polish
    if time.time() < deadline - 3:
        polished = polish(best_selected[:], deadline - 3)
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
