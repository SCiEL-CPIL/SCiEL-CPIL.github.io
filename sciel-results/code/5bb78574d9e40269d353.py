import time
import random
import heapq
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0

    total_space = q ** n

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # masks[i] = bitmask of all words within Hamming distance 1 of i
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

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def lazy_greedy(max_time, seed=None):
        """Submodular greedy with lazy heap: each step picks the word with max
        marginal gain among remaining, using a heap with stale-entry skipping."""
        rng = random.Random(seed)
        covered = 0
        remaining = full_mask
        remaining_count = total_space
        selected = []

        # lazy heap: (neg_gain, seq, word). seq breaks ties for diversity.
        heap = []
        seq = 0
        for w in range(total_space):
            gain = masks[w].bit_count()
            if seed is not None:
                gain = gain + rng.uniform(0, 0.4)
            heapq.heappush(heap, (-gain, seq, w))
            seq += 1

        while remaining_count > 0 and time.time() < max_time:
            # pop until we find a word whose stored gain matches current gain
            while heap:
                ng, _, w = heapq.heappop(heap)
                g = -ng
                # recompute true marginal gain
                true_gain = (masks[w] & remaining).bit_count()
                # stored gain was (base + noise); true base is bit_count of mask & remaining
                # accept if stored approx equals current base gain (within noise tolerance)
                base = (masks[w] & remaining).bit_count()
                if abs(g - base) <= 0.41:
                    break
            else:
                break

            selected.append(w)
            covered |= masks[w]
            new_remaining = remaining & ~masks[w]
            remaining_count = new_remaining.bit_count()
            remaining = new_remaining

            if remaining_count == 0:
                break

            # lazily push updated gains for words whose gain may have increased
            # (we can't easily know which; instead push all words that are still
            #  "useful" -- but that's expensive. Instead, we re-evaluate only the
            #  top few candidates that we popped earlier. Simpler: push a fresh
            #  entry for every word only occasionally.)
            # To keep it cheap, we push updated gains for a random sample.
            if len(selected) % 3 == 0:
                sample = rng.sample(range(total_space), min(200, total_space))
                for w in sample:
                    g = (masks[w] & remaining).bit_count()
                    if g > 0:
                        heapq.heappush(heap, (-g, seq, w))
                        seq += 1

        return selected

    def polish(selected, time_limit):
        """Remove redundant codewords (any word whose coverage is fully covered by others)."""
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

    def two_for_one_search(selected, time_limit):
        """Try to drop codeword i if its unique points are covered by codeword j.
        Then try 2-for-1: drop i and k if a single word j covers all their unique points."""
        selected = selected[:]
        if len(selected) < 3:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        def unique_points(w):
            m = masks[w]
            pts = []
            mm = m
            while mm:
                lowbit = mm & -mm
                pos = lowbit.bit_length() - 1
                if counts[pos] == 1:
                    pts.append(pos)
                mm ^= lowbit
            return pts

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            L = len(selected)
            # 1-for-0: drop if no unique points
            for i in range(L):
                if time.time() > time_limit:
                    break
                w = selected[i]
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
                    selected.pop(i)
                    improved = True
                    break

            if improved or time.time() > time_limit:
                continue

            # 1-for-1 swap: drop i, add j if j covers i's unique points and j not selected
            sel_set = set(selected)
            for i in range(L):
                if time.time() > time_limit:
                    break
                w = selected[i]
                uw = unique_points(w)
                if not uw:
                    continue
                # find j not in sel_set covering all uw, with max extra gain
                best_j = -1
                best_gain = 0
                for j in range(total_space):
                    if j in sel_set:
                        continue
                    m2 = masks[j]
                    ok = True
                    for p in uw:
                        if not (m2 >> p) & 1:
                            ok = False
                            break
                    if not ok:
                        continue
                    gain = (m2 & ~masks[w]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                if best_j >= 0:
                    # swap
                    mm = masks[w]
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    mm = masks[best_j]
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit
                    selected[i] = best_j
                    sel_set.add(best_j)
                    improved = True
                    break

            if improved or time.time() > time_limit:
                continue

            # 2-for-1: drop i and k if some j covers union of their unique points
            L = len(selected)
            found = False
            for i in range(L):
                if time.time() > time_limit:
                    break
                wi = selected[i]
                uwi = unique_points(wi)
                if not uwi:
                    continue
                for k in range(i+1, L):
                    if time.time() > time_limit:
                        break
                    wk = selected[k]
                    uwk = unique_points(wk)
                    if not uwk:
                        continue
                    union = set(uwi) | set(uwk)
                    # find j not in {wi,wk} covering all union points
                    best_j = -1
                    for j in range(total_space):
                        if j == wi or j == wk:
                            continue
                        m2 = masks[j]
                        ok = True
                        for p in union:
                            if not (m2 >> p) & 1:
                                ok = False
                                break
                        if ok:
                            best_j = j
                            break
                    if best_j >= 0:
                        # drop wi and wk, add best_j
                        mm = masks[wi]
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        mm = masks[wk]
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        mm = masks[best_j]
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] += 1
                            mm ^= lowbit
                        selected.pop(k)
                        selected.pop(i)
                        selected.append(best_j)
                        found = True
                        improved = True
                        break
                if found:
                    break
        return selected

    best_size = total_space
    best_selected = list(range(total_space))

    def update_best(sol):
        nonlocal best_size, best_selected
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]

    # Phase 1: lazy greedy with several seeds
    for seed in [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 20:
            break
        sol = lazy_greedy(deadline - 20, seed)
        if len(sol) < best_size + 8:
            polished = polish(sol[:], deadline - 20)
            update_best(polished)

    # Phase 2: 2-for-1 local search on best
    if time.time() < deadline - 5:
        ls = two_for_one_search(best_selected[:], deadline - 5)
        update_best(ls)

    # Phase 3: final polish
    if time.time() < deadline - 2:
        polished = polish(best_selected[:], deadline - 2)
        update_best(polished)

    # Phase 4: one more 2-for-1 pass
    if time.time() < deadline - 1:
        ls = two_for_one_search(best_selected[:], deadline - 1)
        update_best(ls)

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
