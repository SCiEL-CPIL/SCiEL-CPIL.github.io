import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n
    cover_count = 1 + n * (q - 1)  # For q=3,n=7: 1+14=15

    # Precompute neighbor indices as a fixed-size array
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        w = []
        tmp = i
        for _ in range(n):
            w.append(tmp % q)
            tmp //= q
        w.reverse()
        neighbors_arr[i, 0] = i
        k = 1
        for pos in range(n):
            orig_val = w[pos]
            power = q ** (n - 1 - pos)
            for val in range(q):
                if val == orig_val:
                    continue
                j = i + (val - orig_val) * power
                neighbors_arr[i, k] = j
                k += 1

    # Precompute reverse mapping: for each word, which codewords cover it
    reverse_cover = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in range(cover_count):
            w = neighbors_arr[i, j]
            reverse_cover[w].append(i)

    reverse_cover_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    reverse_cover_counts = np.zeros(total_words, dtype=np.int32)
    for w in range(total_words):
        rc = reverse_cover[w]
        for j, ci in enumerate(rc):
            reverse_cover_arr[w, j] = ci
        reverse_cover_counts[w] = len(rc)

    def idx_to_word(idx):
        w = []
        tmp = idx
        for _ in range(n):
            w.append(tmp % q)
            tmp //= q
        w.reverse()
        return w

    def greedy_cover_randomized():
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []

        while True:
            if time.time() > deadline - 5.0:
                break

            # Vectorized gain: for each candidate, count how many of its neighbors are uncovered
            gains = uncovered[neighbors_arr].sum(axis=1)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates) - 1)]

            code.append(int(best_idx))
            uncovered[neighbors_arr[best_idx]] = False

        return code

    def find_candidates_covering(unique_indices, exclude_set):
        if len(unique_indices) == 0:
            return np.arange(total_words, dtype=np.int64)
        lists = [reverse_cover_arr[w, :reverse_cover_counts[w]] for w in unique_indices]
        lists.sort(key=len)
        cur = lists[0].copy()
        for lst in lists[1:]:
            cur = np.intersect1d(cur, lst)
            if cur.size == 0:
                break
        if cur.size == 0:
            return cur
        if exclude_set:
            mask = np.ones(total_words, dtype=bool)
            for j in exclude_set:
                mask[j] = False
            cur = cur[mask[cur]]
        return cur

    def local_search_reduce(code, max_time=None):
        if not code:
            return code

        if max_time is None:
            max_time = deadline - time.time()

        current_code = list(code)

        cover_count_arr = np.zeros(total_words, dtype=np.int32)
        for idx in current_code:
            cover_count_arr[neighbors_arr[idx]] += 1

        improved = True
        while improved:
            if time.time() > max_time:
                break
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)

            for i in range(n_cw):
                if time.time() > max_time:
                    return current_code

                idx = current_code[i]
                mask = neighbors_arr[idx]

                # Check if removable
                min_count = np.min(cover_count_arr[mask])

                if min_count > 1:
                    cover_count_arr[mask] -= 1
                    current_code.pop(i)
                    improved = True
                    break
                else:
                    unique_mask = (cover_count_arr[mask] == 1)
                    if np.any(unique_mask):
                        unique_indices = mask[unique_mask]
                        exclude_set = set(current_code)
                        candidates = find_candidates_covering(unique_indices, exclude_set)

                        if len(candidates) > 0:
                            new_idx = candidates[random.randint(0, len(candidates) - 1)]
                            cover_count_arr[mask] -= 1
                            cover_count_arr[neighbors_arr[new_idx]] += 1
                            current_code[i] = int(new_idx)
                            improved = True
                            break
                        else:
                            # 2-for-1 swap
                            for j in range(n_cw):
                                if time.time() > max_time:
                                    return current_code
                                if j == i:
                                    continue
                                idx2 = current_code[j]
                                mask2 = neighbors_arr[idx2]
                                unique_mask2 = (cover_count_arr[mask2] == 1)
                                if not np.any(unique_mask2):
                                    continue
                                unique_indices2 = mask2[unique_mask2]
                                combined = np.unique(np.concatenate([unique_indices, unique_indices2]))
                                exclude_set2 = set(current_code)
                                candidates2 = find_candidates_covering(combined, exclude_set2)
                                if len(candidates2) > 0:
                                    new_idx = candidates2[random.randint(0, len(candidates2) - 1)]
                                    cover_count_arr[mask] -= 1
                                    cover_count_arr[mask2] -= 1
                                    cover_count_arr[neighbors_arr[new_idx]] += 1
                                    if j < i:
                                        current_code.pop(i)
                                        current_code.pop(j)
                                    else:
                                        current_code.pop(j)
                                        current_code.pop(i)
                                    current_code.append(int(new_idx))
                                    improved = True
                                    break
                            if improved:
                                break

        return current_code

    # Initial greedy construction
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)

    # Run restarts if time permits
    while time.time() < deadline - 15.0:
        cand_code = greedy_cover_randomized()
        cand_code = local_search_reduce(cand_code)
        if len(cand_code) < len(best_code):
            best_code = cand_code

    # Final polish
    best_code = local_search_reduce(best_code)

    # Convert indices back to words
    return [idx_to_word(i) for i in best_code]
