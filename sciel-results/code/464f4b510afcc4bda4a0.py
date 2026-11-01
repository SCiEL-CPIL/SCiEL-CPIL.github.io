"""Improved construction for K_3(7,1) using bitset-based greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    return _construct_general(q, n, radius, deadline)


def _construct_3_7_1(deadline):
    """Optimized construction for K_3(7,1)."""
    N = 2187

    # Precompute cover masks and neighbor masks
    covers = [0] * N
    for i in range(N):
        digits = [0] * 7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        mask = 1 << i
        for pos in range(7):
            for val in range(3):
                if val != digits[pos]:
                    diff = val - digits[pos]
                    new_i = i + diff * (3 ** pos)
                    mask |= (1 << new_i)
        covers[i] = mask

    full_mask = (1 << N) - 1

    # Precompute cover index lists for faster numpy operations
    cover_indices = []
    for i in range(N):
        mask = covers[i]
        idxs = []
        m = mask
        while m:
            lsb = m & -m
            idxs.append(lsb.bit_length() - 1)
            m ^= lsb
        cover_indices.append(idxs)

    all_indices = list(range(N))

    def greedy_cover(randomize=False, seed=None):
        """Greedy set cover with optional randomization."""
        if seed is not None:
            random.seed(seed)

        uncovered = full_mask
        selected = []

        while uncovered:
            best_count = -1
            best_idx = -1
            best_indices = []

            for i in all_indices:
                cnt = (uncovered & covers[i]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                    best_indices = [i]
                elif cnt == best_count:
                    best_indices.append(i)

            if best_count == 0:
                break

            if randomize and len(best_indices) > 1:
                idx = random.choice(best_indices)
            else:
                idx = best_idx

            selected.append(idx)
            uncovered &= ~covers[idx]

            if time.time() > deadline - 5.0:
                break

        return selected, uncovered

    def local_search_remove(code, deadline):
        """Remove redundant codewords."""
        if not code:
            return code

        counts = np.zeros(N, dtype=np.int32)
        for i in code:
            counts[cover_indices[i]] += 1

        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            indices_to_try = list(range(len(code)))
            random.shuffle(indices_to_try)

            for idx_in_code in indices_to_try:
                if idx_in_code >= len(code):
                    continue
                c = code[idx_in_code]

                if np.all(counts[cover_indices[c]] > 1):
                    counts[cover_indices[c]] -= 1
                    code.pop(idx_in_code)
                    changed = True

        return code

    def local_search_swap(code, deadline):
        """Swap codewords to enable further reductions."""
        if not code:
            return code

        code_set = set(code)
        counts = np.zeros(N, dtype=np.int32)
        for i in code:
            counts[cover_indices[i]] += 1

        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            indices_to_try = list(range(len(code)))
            random.shuffle(indices_to_try)

            for idx_in_code in indices_to_try:
                if idx_in_code >= len(code):
                    continue
                c = code[idx_in_code]

                for _ in range(15):
                    if time.time() > deadline - 5.0:
                        break
                    d = random.choice(all_indices)
                    if d in code_set:
                        continue

                    c_set = set(cover_indices[c])
                    d_set = set(cover_indices[d])
                    only_c = c_set - d_set
                    if only_c:
                        if np.any(counts[list(only_c)] == 1):
                            continue

                    code[idx_in_code] = d
                    code_set.remove(c)
                    code_set.add(d)
                    counts[cover_indices[c]] -= 1
                    counts[cover_indices[d]] += 1
                    changed = True
                    break

        return code

    def two_opt_swap(code, deadline):
        """Swap pairs of codewords."""
        if len(code) < 2:
            return code

        code_set = set(code)
        counts = np.zeros(N, dtype=np.int32)
        for i in code:
            counts[cover_indices[i]] += 1

        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            indices_to_try = list(range(len(code)))
            random.shuffle(indices_to_try)

            for i in range(min(8, len(indices_to_try))):
                if indices_to_try[i] >= len(code):
                    continue
                c1 = code[indices_to_try[i]]

                for j in range(i + 1, min(i + 8, len(indices_to_try))):
                    if indices_to_try[j] >= len(code):
                        continue
                    c2 = code[indices_to_try[j]]

                    for _ in range(8):
                        if time.time() > deadline - 5.0:
                            break
                        d1 = random.choice(all_indices)
                        d2 = random.choice(all_indices)
                        if d1 in code_set or d2 in code_set or d1 == d2:
                            continue

                        c1_set = set(cover_indices[c1])
                        c2_set = set(cover_indices[c2])
                        d1_set = set(cover_indices[d1])
                        d2_set = set(cover_indices[d2])

                        only_c1 = c1_set - c2_set - d1_set - d2_set
                        if only_c1:
                            if np.any(counts[list(only_c1)] == 1):
                                continue

                        only_c2 = c2_set - c1_set - d1_set - d2_set
                        if only_c2:
                            if np.any(counts[list(only_c2)] == 1):
                                continue

                        code[indices_to_try[i]] = d1
                        code[indices_to_try[j]] = d2
                        code_set.remove(c1)
                        code_set.remove(c2)
                        code_set.add(d1)
                        code_set.add(d2)
                        counts[cover_indices[c1]] -= 1
                        counts[cover_indices[c2]] -= 1
                        counts[cover_indices[d1]] += 1
                        counts[cover_indices[d2]] += 1
                        changed = True
                        break
                    if changed:
                        break
                if changed:
                    break

        return code

    def full_polish(code, deadline):
        """Apply all local search steps in sequence."""
        code = local_search_remove(code, deadline)
        code = local_search_swap(code, deadline)
        code = two_opt_swap(code, deadline)
        code = local_search_remove(code, deadline)
        code = local_search_swap(code, deadline)
        code = local_search_remove(code, deadline)
        return code

    best_code = None
    best_size = N

    # 1. Deterministic Greedy
    code, unc = greedy_cover(randomize=False)
    if not unc:
        code = full_polish(code, deadline)
        best_code = code
        best_size = len(code)

    # 2. Multiple randomized attempts
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        code, unc = greedy_cover(randomize=True, seed=attempts)
        if not unc:
            code = full_polish(code, deadline)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code

    if best_code is None:
        code, unc = greedy_cover(randomize=False)
        if unc:
            while unc:
                lsb = unc & -unc
                idx = lsb.bit_length() - 1
                code.append(idx)
                unc &= ~covers[idx]
        best_code = code
        best_size = len(best_code)

    result = []
    for idx in best_code:
        word = [0] * 7
        tmp = idx
        for pos in range(7):
            word[pos] = tmp % 3
            tmp //= 3
        result.append(word)

    return result


def _construct_general(q, n, radius, deadline):
    """General fallback."""
    import itertools

    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)

    uncovered = set(range(total))
    selected = []

    while uncovered and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for i in range(total):
            cnt = 0
            for j in uncovered:
                dist = sum(1 for a, b in zip(all_words[i], all_words[j]) if a != b)
                if dist <= radius:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i

        if best_idx == -1:
            break

        selected.append(best_idx)
        to_remove = []
        for j in uncovered:
            dist = sum(1 for a, b in zip(all_words[best_idx], all_words[j]) if a != b)
            if dist <= radius:
                to_remove.append(j)
        for j in to_remove:
            uncovered.discard(j)

    if uncovered:
        for j in uncovered:
            selected.append(j)

    return [list(all_words[i]) for i in selected]
# EVOLVE-BLOCK-END
