import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    total_space = q ** n

    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    all_words = [idx_to_word(i) for i in range(total_space)]

    # Build cover matrix using NumPy for vectorized operations
    cover_matrix = np.zeros((total_space, total_space), dtype=np.uint8)
    np.fill_diagonal(cover_matrix, 1)

    if radius >= 1:
        powers = [q ** (n - 1 - pos) for pos in range(n)]
        for i in range(total_space):
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                power = powers[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    j = i + (val - orig_val) * power
                    cover_matrix[i, j] = 1

    # Precompute cover_masks as Python integers for bitset operations in local search
    cover_masks = [0] * total_space
    for i in range(total_space):
        mask = 1 << i
        if radius >= 1:
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                power = q ** (n - 1 - pos)
                for val in range(q):
                    if val == orig_val:
                        continue
                    j = i + (val - orig_val) * power
                    mask |= (1 << j)
        cover_masks[i] = mask

    full_mask = (1 << total_space) - 1

    # Precompute reverse cover
    reverse_cover = [[] for _ in range(total_space)]
    for cw_idx in range(total_space):
        m = cover_masks[cw_idx]
        while m:
            low = m & (-m)
            w_idx = low.bit_length() - 1
            reverse_cover[w_idx].append(cw_idx)
            m ^= low

    # NumPy-based greedy with vectorized gain computation
    def greedy_cover_randomized():
        uncovered = np.ones(total_space, dtype=np.uint8)
        code = []

        while uncovered.any():
            gains = cover_matrix.dot(uncovered)
            max_gain = gains.max()
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            code.append(best_idx)
            uncovered[cover_matrix[best_idx]] = 0

        return code

    # Bitset-based local search (from current program, refined)
    def local_search_reduce(code):
        if not code:
            return code

        current_code = code[:]
        current_code_set = set(current_code)

        cover_count = [0] * total_space
        for idx in current_code:
            m = cover_masks[idx]
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                cover_count[w_idx] += 1
                m ^= low

        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)

            # Phase 1: Removal
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                idx = current_code[i]
                mask = cover_masks[idx]
                removable = True
                m = mask
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    if cover_count[w_idx] <= 1:
                        removable = False
                        break
                    m ^= low
                if removable:
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] -= 1
                        m ^= low
                    current_code.pop(i)
                    current_code_set.remove(idx)
                    improved = True
                    break

            # Phase 2: Swaps
            if not improved:
                max_unique_to_cover = 4
                for i in range(n_cw):
                    if time.time() > deadline:
                        return current_code
                    idx = current_code[i]
                    mask = cover_masks[idx]
                    unique_mask = 0
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        if cover_count[w_idx] == 1:
                            unique_mask |= (1 << w_idx)
                        m ^= low
                    if unique_mask == 0:
                        m = mask
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        current_code.pop(i)
                        current_code_set.remove(idx)
                        improved = True
                        break
                    if unique_mask.bit_count() > max_unique_to_cover:
                        continue
                    words_in_unique = []
                    m = unique_mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        words_in_unique.append(w_idx)
                        m ^= low
                    words_in_unique.sort(key=lambda w: len(reverse_cover[w]))
                    candidates = set(reverse_cover[words_in_unique[0]])
                    for w in words_in_unique[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(reverse_cover[w])
                    candidates.discard(idx)
                    candidates -= current_code_set
                    if candidates:
                        new_idx = random.choice(list(candidates))
                        m = cover_masks[idx]
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        m = cover_masks[new_idx]
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] += 1
                            m ^= low
                        current_code[i] = new_idx
                        current_code_set.remove(idx)
                        current_code_set.add(new_idx)
                        improved = True
                        break

            # Phase 3: Remove if another codeword covers all unique words
            if not improved and len(current_code) > 2:
                n_cw = len(current_code)
                for i in range(n_cw):
                    if time.time() > deadline:
                        return current_code
                    idx = current_code[i]
                    mask = cover_masks[idx]
                    unique_mask = 0
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        if cover_count[w_idx] == 1:
                            unique_mask |= (1 << w_idx)
                        m ^= low
                    if unique_mask == 0:
                        m = mask
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        current_code.pop(i)
                        current_code_set.remove(idx)
                        improved = True
                        break
                    if unique_mask.bit_count() > 6:
                        continue
                    # Check if any single other codeword covers all unique words
                    for j in range(n_cw):
                        if j == i:
                            continue
                        other_mask = cover_masks[current_code[j]]
                        if not (unique_mask & ~other_mask):
                            m = mask
                            while m:
                                low = m & (-m)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] -= 1
                                m ^= low
                            current_code.pop(i)
                            current_code_set.remove(idx)
                            improved = True
                            break
                    if improved:
                        break

        return current_code

    # Initial greedy
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)
    best_size = len(best_code)

    # Multiple randomized restarts
    if time.time() < deadline - 15.0:
        for _ in range(30):
            if time.time() > deadline - 5.0:
                break
            cand_code = greedy_cover_randomized()
            cand_code = local_search_reduce(cand_code)
            if len(cand_code) < best_size:
                best_code = cand_code
                best_size = len(cand_code)

    # Final polish: aggressive local search on best solution
    if time.time() < deadline - 3.0:
        best_code = local_search_reduce(best_code)

    return [all_words[i] for i in best_code]
