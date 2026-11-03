import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 142.0

    total_space = q ** n
    cover_count = 1 + n * (q - 1)

    # Positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i - 1] * q

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    # Neighbor indices (self + radius-1 neighbors) as a compact int32 matrix
    neighbors = np.zeros((total_space, cover_count), dtype=np.int32)
    for i in range(total_space):
        w = [0] * n
        for pos in range(n):
            w[pos] = (i // powers[pos]) % q
        j = 0
        neighbors[i, j] = i
        j += 1
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                diff = new_val - orig
                neighbors[i, j] = i + diff * powers[pos]
                j += 1

    # Reverse cover: for each word, the set of codewords whose radius-1 ball covers it
    reverse_cover = [[] for _ in range(total_space)]
    for i in range(total_space):
        for k in range(cover_count):
            reverse_cover[neighbors[i, k]].append(i)
    rev_counts = np.array([len(rc) for rc in reverse_cover], dtype=np.int32)
    max_rev = int(rev_counts.max())
    reverse_cover_arr = np.full((total_space, max_rev), -1, dtype=np.int32)
    for i in range(total_space):
        rc = reverse_cover[i]
        reverse_cover_arr[i, :len(rc)] = rc

    # Bitmask representation (fast for polish / swap logic)
    masks = [0] * total_space
    for i in range(total_space):
        m = 0
        for k in range(cover_count):
            m |= (1 << neighbors[i, k])
        masks[i] = m

    best_size = total_space
    best_selected = list(range(total_space))

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively."""
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

    def local_search(selected, time_limit):
        """1-for-1 swaps (via reverse_cover) and 2-for-1 swaps."""
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

        # ---- 1-for-1 swap / removal ----
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

                # words uniquely covered by w
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

                # candidate replacement: must cover every uniquely-covered word
                cand = None
                mm = unique_mask
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    rc = reverse_cover_arr[pos, :rev_counts[pos]]
                    s = set(rc.tolist())
                    cand = s if cand is None else (cand & s)
                    mm ^= lowbit
                    if cand is not None and not cand:
                        break
                if cand is None or not cand:
                    continue
                cand.discard(w)
                if not cand:
                    continue
                # choose the candidate with maximal net new coverage
                best_word = -1
                best_gain = 0
                for w2 in cand:
                    gain = (masks[w2] & ~m).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w2
                if best_word < 0 or best_gain <= 0:
                    continue

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

        # ---- 2-for-1 swap ----
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

                    # replacement covering all u_ab words
                    cand = None
                    mm = u_ab
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        rc = reverse_cover_arr[pos, :rev_counts[pos]]
                        s = set(rc.tolist())
                        cand = s if cand is None else (cand & s)
                        mm ^= lowbit
                        if cand is not None and not cand:
                            break
                    if cand is None or not cand:
                        continue
                    cand.discard(w1)
                    cand.discard(w2)
                    if not cand:
                        continue
                    best_word = random.choice(sorted(cand))

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
        """Vectorized greedy set cover using the compact neighbor matrix."""
        uncovered = np.ones(total_space, dtype=np.bool_)
        selected = []
        selected_set = set()
        while uncovered.any() and time.time() < deadline - 8:
            gains = uncovered[neighbors].sum(axis=1).astype(np.float64)
            for w in selected_set:
                gains[w] = -1.0
            if randomized:
                gains += np.random.normal(0, alpha, size=total_space)
            best_word = int(np.argmax(gains))
            if gains[best_word] <= 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            uncovered[neighbors[best_word, :]] = False
        return selected

    # Strategy 1: deterministic greedy + polish + local search
    if time.time() < deadline - 15:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 15)
        ls = local_search(polished[:], deadline - 15)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: many randomized greedy restarts
    max_restarts = 400
    for r in range(max_restarts):
        if time.time() > deadline - 25:
            break
        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.6))
        if len(sol) < best_size + 8:
            polished = polish(sol[:], deadline - 25)
            ls = local_search(polished[:], deadline - 25)
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

    return [int_to_tuple(v) for v in best_selected]
