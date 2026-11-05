import numpy as np
import random
import time
import heapq

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 145.0

    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n  # 2187
        ball_size = 1 + n * (q - 1)  # 15

        def int_to_base3(num, length):
            res = [0] * length
            for i in range(length - 1, -1, -1):
                res[i] = num % 3
                num //= 3
            return res

        def base3_to_int(digits):
            res = 0
            for d in digits:
                res = res * 3 + d
            return res

        # Precompute balls as lists of ints
        balls = []
        for i in range(N):
            digits = int_to_base3(i, n)
            neighbors = [i]
            for pos in range(n):
                original = digits[pos]
                for new_val in range(q):
                    if new_val != original:
                        new_digits = digits[:]
                        new_digits[pos] = new_val
                        neighbors.append(base3_to_int(new_digits))
            balls.append(neighbors)

        # Precompute cover_map: for each word, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        # Precompute ball sets for fast membership
        ball_sets = [set(b) for b in balls]
        # Precompute ball indices as numpy arrays for vectorized ops
        ball_indices = [np.array(b, dtype=np.int32) for b in balls]

        # Precompute neighbor index matrix for vectorized gain
        neighbor_mat = np.empty((N, ball_size), dtype=np.int32)
        for c in range(N):
            neighbor_mat[c] = balls[c]

        # Precompute cover_map as numpy arrays
        cover_map_arr = [np.array(cm, dtype=np.int32) for cm in cover_map]
        cover_map_sets = [set(cm) for cm in cover_map]

        def greedy_cover(seed, start_code=None):
            rng = random.Random(seed)
            covered = np.zeros(N, dtype=bool)
            cnt = np.full(N, ball_size, dtype=np.int32)
            code = []

            if start_code:
                for c in start_code:
                    code.append(c)
                    for w in balls[c]:
                        if not covered[w]:
                            covered[w] = True
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1

            # Use a priority queue
            heap = [(-ball_size, i) for i in range(N)]
            heapq.heapify(heap)

            # Recompute cnt for heap entries that weren't in start_code
            # Actually we need to be careful: if start_code covers some words,
            # cnt for those codewords should be reduced. Let's recompute properly.
            # For simplicity, recompute cnt from scratch if start_code is large
            if start_code and len(start_code) > 50:
                covered = np.zeros(N, dtype=bool)
                cnt = np.full(N, ball_size, dtype=np.int32)
                for c in start_code:
                    for w in balls[c]:
                        covered[w] = True
                for c in range(N):
                    cnt[c] = np.sum(~covered[ball_indices[c]])
                heap = [(-int(cnt[i]), i) for i in range(N)]
                heapq.heapify(heap)

            uncovered_count = int(np.sum(~covered))

            while uncovered_count > 0:
                # Find best candidate from heap
                while heap:
                    neg_val, i = heapq.heappop(heap)
                    if cnt[i] == -neg_val and not covered[i]:
                        break
                else:
                    # Heap exhausted, rebuild
                    cnt_valid = cnt.copy()
                    cnt_valid[covered] = 0
                    heap = [(-int(cnt_valid[i]), i) for i in range(N) if not covered[i]]
                    heapq.heapify(heap)
                    if not heap:
                        break
                    neg_val, i = heapq.heappop(heap)
                    if cnt[i] != -neg_val:
                        continue

                c = i
                code.append(c)
                covered[c] = True  # mark as used (not a word to cover)

                # Update counts
                ball_c = ball_indices[c]
                new_uncovered_mask = ~covered[ball_c]
                if np.any(new_uncovered_mask):
                    newly_covered = ball_c[new_uncovered_mask]
                    uncovered_count -= len(newly_covered)
                    covered[ball_c] = True

                    # Update cnt for all codewords that cover any of the newly covered words
                    # Use vectorized approach: for each newly covered word, decrement cnt of its coverers
                    # This is expensive in Python, so do it efficiently
                    for w in newly_covered:
                        for c2 in cover_map[w]:
                            if not covered[c2]:  # c2 not already used
                                cnt[c2] -= 1
                                if cnt[c2] > 0:
                                    heapq.heappush(heap, (-int(cnt[c2]), c2))

            # Remove the 'covered' flag we set on codewords themselves
            # Actually we don't need to since we track code separately
            return code

        def greedy_cover_fast(seed, start_code=None):
            """Faster greedy using numpy for gain computation."""
            rng = random.Random(seed)
            uncovered = np.ones(N, dtype=bool)
            in_code = np.zeros(N, dtype=bool)
            code = []

            if start_code:
                for c in start_code:
                    code.append(c)
                    in_code[c] = True
                    uncovered[ball_indices[c]] = False

            uncovered_count = int(np.sum(uncovered))

            # Precompute gain for initial state
            # gain[i] = number of uncovered words in ball[i]
            # Use matrix multiplication: gain = neighbor_mat @ uncovered (but neighbor_mat is 0/1)
            # Actually neighbor_mat has the actual word indices, so we need:
            # gain[i] = sum(uncovered[neighbor_mat[i]])
            # This is: for each i, sum over its 15 neighbors

            while uncovered_count > 0 and time.time() < deadline - 10:
                # Compute gain for all candidates
                # gain[i] = sum(uncovered[neighbor_mat[i, :]])
                # This is a matrix-vector product where neighbor_mat is used as indices
                # Use: gain = np.zeros(N); for each i, gain[i] = uncovered[neighbor_mat[i]].sum()
                # Vectorized: we can do this with bincount trick
                # For each position in the ball, count how many uncovered
                # gain[i] = sum over j of uncovered[neighbor_mat[i,j]]
                # = for each word w, count how many codewords i have w in their ball AND uncovered[w]
                # Alternative: gain = (neighbor_mat as boolean) @ uncovered
                # But neighbor_mat contains indices, not 0/1

                # Let's just do it in a loop but with numpy sum
                gains = np.zeros(N, dtype=np.int32)
                for i in range(N):
                    if in_code[i]:
                        continue
                    gains[i] = int(np.sum(uncovered[ball_indices[i]]))

                # Find max gain among non-code candidates
                gains[in_code] = -1
                if gains.max() <= 0:
                    break

                c = int(np.argmax(gains))
                code.append(c)
                in_code[c] = True

                # Update uncovered
                newly = ball_indices[c]
                was_uncovered = uncovered[newly]
                n_new = int(np.sum(was_uncovered))
                if n_new > 0:
                    uncovered[newly[was_uncovered]] = False
                    uncovered_count -= n_new

            return code

        def refine(code):
            """Remove codewords that don't uniquely cover any word."""
            if not code:
                return code

            cover_count = np.zeros(N, dtype=np.int32)
            for c in code:
                cover_count[ball_indices[c]] += 1

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    # Check if any word is covered only by this codeword
                    if cover_count[ball_indices[c]].min() > 1:
                        # Safe to remove
                        cover_count[ball_indices[c]] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            """1-swap and 2-swap local search."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = np.zeros(N, dtype=np.int32)
            for c in code_list:
                cover_count[ball_indices[c]] += 1

            in_code = np.zeros(N, dtype=bool)
            for c in code_list:
                in_code[c] = True

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)

                # 1-swap phase
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]

                    # Find words that would become uncovered
                    newly_uncovered = []
                    bi = ball_indices[c_remove]
                    old_counts = cover_count[bi].copy()
                    cover_count[bi] -= 1
                    for j in range(len(bi)):
                        if old_counts[j] == 1:  # was covered only once
                            newly_uncovered.append(int(bi[j]))

                    if not newly_uncovered:
                        # Safe removal
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break

                    # Try to find a replacement
                    found = -1
                    if len(newly_uncovered) <= 8:
                        # Sort by number of candidates (most constrained first)
                        crit_sorted = sorted(newly_uncovered, key=lambda w: len(cover_map[w]))
                        cand_pool = None
                        for w in crit_sorted:
                            candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove]
                            if cand_pool is None:
                                cand_pool = set(candidates_w)
                            else:
                                cand_pool.intersection_update(candidates_w)
                            if not cand_pool:
                                break
                        if cand_pool:
                            needed_set = set(newly_uncovered)
                            for c_add in cand_pool:
                                if needed_set <= ball_sets[c_add]:
                                    found = c_add
                                    break

                    if found >= 0:
                        code_list[i] = found
                        in_code[c_remove] = False
                        in_code[found] = True
                        cover_count[ball_indices[found]] += 1
                        improved = True
                        break
                    else:
                        # Restore
                        cover_count[bi] += 1

                # 2-swap phase
                if time.time() < time_limit and len(code_list) >= 3:
                    n_codes = len(code_list)
                    pairs = list(range(n_codes))
                    rng.shuffle(pairs)
                    for i2 in pairs:
                        if time.time() > time_limit:
                            break
                        if i2 >= n_codes:
                            continue
                        for j2 in range(i2 + 1, n_codes):
                            if time.time() > time_limit:
                                break
                            c_rem1 = code_list[i2]
                            c_rem2 = code_list[j2]

                            # Try removing both
                            bi1 = ball_indices[c_rem1]
                            bi2 = ball_indices[c_rem2]
                            old_counts1 = cover_count[bi1].copy()
                            cover_count[bi1] -= 1
                            old_counts2 = cover_count[bi2].copy()
                            cover_count[bi2] -= 1

                            uncov = set()
                            for j in range(len(bi1)):
                                if old_counts1[j] == 1:
                                    uncov.add(int(bi1[j]))
                            for j in range(len(bi2)):
                                if old_counts2[j] == 1 and cover_count[bi2[j]] == 0:
                                    uncov.add(int(bi2[j]))
                                elif old_counts2[j] > 1 and cover_count[bi2[j]] == 0:
                                    uncov.add(int(bi2[j]))

                            # Rebuild uncov properly
                            uncov = set()
                            for j in range(len(bi1)):
                                if cover_count[bi1[j]] == 0:
                                    uncov.add(int(bi1[j]))
                            for j in range(len(bi2)):
                                if cover_count[bi2[j]] == 0:
                                    uncov.add(int(bi2[j]))

                            if uncov and len(uncov) <= 10:
                                cand_pool = None
                                for w in sorted(uncov, key=lambda x: len(cover_map[x])):
                                    candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_rem1 and c2 != c_rem2]
                                    if cand_pool is None:
                                        cand_pool = set(candidates_w)
                                    else:
                                        cand_pool.intersection_update(candidates_w)
                                    if not cand_pool:
                                        break
                                found2 = -1
                                if cand_pool:
                                    needed_set = set(uncov)
                                    for c_add in cand_pool:
                                        if needed_set <= ball_sets[c_add]:
                                            found2 = c_add
                                            break
                                if found2 >= 0:
                                    code_list[j2] = found2
                                    in_code[c_rem2] = False
                                    in_code[found2] = True
                                    cover_count[ball_indices[found2]] += 1
                                    code_list.pop(i2)
                                    in_code[c_rem1] = False
                                    improved = True
                                    break
                                else:
                                    # Restore
                                    cover_count[bi1] += 1
                                    cover_count[bi2] += 1
                            else:
                                # Restore
                                cover_count[bi1] += 1
                                cover_count[bi2] += 1

                            if improved:
                                break
                        if improved:
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        # Multi-start with greedy + refine + local search
        num_restarts = 40
        for i in range(num_restarts):
            if time.time() > deadline - 20:
                break

            code = greedy_cover_fast(i)
            refined = refine(code)

            if len(refined) < best_len:
                best_len = len(refined)
                best_code = list(refined)

            if time.time() > deadline - 15:
                continue

            ls_time = min(deadline - 10, time.time() + 3.0)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = list(ls_code)

        # Kick restarts: remove some codewords and re-greedy
        if best_code is not None:
            kick_sizes = [3, 5, 4, 6, 5, 4, 6, 3, 5, 4]
            for kick_iter, k in enumerate(kick_sizes):
                if time.time() > deadline - 15:
                    break
                rng_k = random.Random(1000 + kick_iter)
                k_actual = min(k, len(best_code))
                kick_set = set(rng_k.sample(best_code, k_actual))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover_fast(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = list(re_refined)
                if time.time() > deadline - 12:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 10, time.time() + 2.5))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = list(ls_code)

        # Final polish
        if best_code is not None and time.time() < deadline - 5:
            ls_code = local_search(best_code, 42, deadline - 3)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = list(ls_code)
            final_code = refine(best_code)
            for polish_iter in range(3):
                if time.time() > deadline - 2:
                    break
                if len(final_code) <= 155:
                    break
                ls_code2 = local_search(final_code, 99 + polish_iter * 13, deadline - 1)
                if len(ls_code2) < len(final_code):
                    final_code = refine(ls_code2)
                else:
                    break
        elif best_code is not None:
            final_code = refine(best_code)
        else:
            final_code = []

        result = []
        for c in final_code:
            result.append(int_to_base3(c, n))
        return result

    # Fallback for other parameters
    N = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((N, nb_count), dtype=np.int32)
    for idx in range(N):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    uncovered = np.ones(N, dtype=bool)
    code = []
    while np.any(uncovered):
        if time.time() > deadline:
            break
        gains = np.array([np.sum(uncovered[neighbors[i]]) for i in range(N)], dtype=np.int32)
        gains[code] = -1
        if gains.max() <= 0:
            break
        best_idx = int(np.argmax(gains))
        code.append(best_idx)
        uncovered[neighbors[best_idx]] = False

    return [idx_to_word(i) for i in code]
