import numpy as np
import random
import time

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin for 150s limit

    # For the specific case of q=3, n=7, radius=1, we use specialized code
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

        # Precompute balls (neighbors including self)
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

        # Precompute cover_map: for each word w, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        # Precompute ball sets for fast membership (swap search)
        ball_sets = [set(b) for b in balls]
        # Precompute ball indices as numpy arrays for faster iteration in local search
        ball_indices = [np.array(b, dtype=np.int32) for b in balls]

        # Precompute gain matrix row-sums for greedy: gain[c] = number of
        # currently-uncovered words in ball c.  We track it incrementally.
        # gain[c] = sum over w in ball c of (1 - covered[w])
        gain = [ball_size] * N
        # For quick update: when w becomes covered, decrement gain[c] for c in cover_map[w]

        import heapq

        def greedy_cover(seed, start_code=None):
            """Greedy set cover with heap-based candidate selection.
            If start_code is given, start from that partial cover and continue
            covering the remaining words (kick / re-greedy)."""
            rng = random.Random(seed)
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N

            # Build heap: (-cnt[i], i) for max-heap behavior
            heap = [(-ball_size, i) for i in range(N)]
            heapq.heapify(heap)
            in_heap = [True] * N

            if start_code:
                for c in start_code:
                    code.append(c)
                    for w in balls[c]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1
                                if cnt[c2] > 0:
                                    heapq.heappush(heap, (-cnt[c2], c2))

            while uncovered_count > 0:
                # Pop valid candidates from heap
                while heap:
                    neg_val, i = heapq.heappop(heap)
                    if cnt[i] == -neg_val and not covered[i]:
                        break
                else:
                    break

                c = i
                code.append(c)

                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0:
                                heapq.heappush(heap, (-cnt[c2], c2))

            return code

        def refine(code):
            """Remove redundant codewords (vectorized)."""
            if not code:
                return code

            # Vectorized cover count
            cover_count = np.zeros(N, dtype=np.int32)
            code_arr = np.array(code, dtype=np.int32)
            for c in code_arr:
                cover_count[ball_indices[c]] += 1

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    # Vectorized check: is any word in ball c covered exactly once?
                    if np.min(cover_count[ball_indices[c]]) > 1:
                        cover_count[ball_indices[c]] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            """Vectorized local search with deletion, 1-swap, and 2-remove-1-add moves."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            # Vectorized cover count
            cover_count = np.zeros(N, dtype=np.int32)
            code_arr = np.array(code_list, dtype=np.int32)
            for c in code_arr:
                cover_count[ball_indices[c]] += 1

            in_code = np.zeros(N, dtype=bool)
            for c in code_list:
                in_code[c] = True

            # Precompute cover_map as numpy arrays for faster candidate lookup
            cover_map_arr = [np.array(cm, dtype=np.int32) for cm in cover_map]
            # Precompute ball_indices as a 2D array for vectorized operations
            ball_indices_2d = np.array(balls, dtype=np.int32)

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]
                    bi = ball_indices_2d[c_remove]

                    # Try deletion first (vectorized)
                    new_cc = cover_count.copy()
                    new_cc[bi] -= 1
                    if np.min(new_cc[bi]) > 0:
                        cover_count = new_cc
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break

                    # Try 1-swap (vectorized)
                    found = -1
                    newly_uncovered_mask = new_cc == 0
                    newly_uncovered_count = np.sum(newly_uncovered_mask)
                    if newly_uncovered_count <= 8:
                        newly_uncovered_list = np.where(newly_uncovered_mask)[0]
                        # Sort by cover_map size (most constrained first)
                        newly_uncovered_list = sorted(newly_uncovered_list, key=lambda w: len(cover_map[w]))
                        w0 = newly_uncovered_list[0]
                        cand_arr = cover_map_arr[w0]
                        valid_mask = ~in_code[cand_arr] & (cand_arr != c_remove)
                        candidates = cand_arr[valid_mask]
                        if len(candidates) > 0:
                            # Vectorized check: does each candidate cover all newly_uncovered?
                            for c_add in candidates:
                                bi_add = ball_indices_2d[c_add]
                                # Check if all newly_uncovered are in ball c_add
                                ok = np.all(np.isin(newly_uncovered_list, bi_add))
                                if ok:
                                    found = c_add
                                    break

                    if found >= 0:
                        cover_count = new_cc
                        cover_count[ball_indices_2d[found]] += 1
                        code_list[i] = found
                        in_code[c_remove] = False
                        in_code[found] = True
                        improved = True
                        break
                    else:
                        # Restore cover_count
                        cover_count = new_cc
                        cover_count[bi] += 1

                # 2-remove-1-add (net -1) - vectorized
                if time.time() < time_limit and len(code_list) >= 3:
                    n_codes = len(code_list)
                    for i2 in range(n_codes):
                        if time.time() > time_limit:
                            break
                        for j2 in range(i2 + 1, n_codes):
                            if time.time() > time_limit:
                                break
                            c_rem1 = code_list[i2]
                            c_rem2 = code_list[j2]
                            bi1 = ball_indices_2d[c_rem1]
                            bi2 = ball_indices_2d[c_rem2]

                            new_cc = cover_count.copy()
                            new_cc[bi1] -= 1
                            new_cc[bi2] -= 1
                            uncov_mask = new_cc == 0
                            uncov_count = np.sum(uncov_mask)
                            if uncov_count > 0 and uncov_count <= 10:
                                uncov_list = np.where(uncov_mask)[0]
                                uncov_list = sorted(uncov_list, key=lambda x: len(cover_map[x]))
                                w0 = uncov_list[0]
                                cand_arr = cover_map_arr[w0]
                                valid_mask = ~in_code[cand_arr] & (cand_arr != c_rem1) & (cand_arr != c_rem2)
                                candidates = cand_arr[valid_mask]
                                found2 = -1
                                if len(candidates) > 0:
                                    for c_add in candidates:
                                        bi_add = ball_indices_2d[c_add]
                                        ok = np.all(np.isin(uncov_list, bi_add))
                                        if ok:
                                            found2 = c_add
                                            break
                                if found2 >= 0:
                                    cover_count = new_cc
                                    cover_count[ball_indices_2d[found2]] += 1
                                    code_list[j2] = found2
                                    in_code[c_rem2] = False
                                    in_code[found2] = True
                                    code_list.pop(i2)
                                    in_code[c_rem1] = False
                                    improved = True
                                    break
                            # Restore if not improved
                            if not improved:
                                pass  # new_cc is local, no restore needed
                        if improved:
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        # Interleave greedy restarts and kick restarts for better exploration
        num_restarts = 100
        kick_sizes = [3, 4, 5, 6, 7, 4, 5, 6, 3, 5, 4, 6, 8, 5, 7, 4, 6, 5, 3, 4, 5, 6, 7, 5, 4, 6, 3, 5, 7, 4]
        kick_idx = 0
        for i in range(num_restarts):
            if time.time() > deadline - 18:
                break

            if i % 2 == 0:
                # Greedy restart
                code = greedy_cover(i)
                refined = refine(code)
                if len(refined) < best_len:
                    best_len = len(refined)
                    best_code = refined
                if time.time() > deadline - 12:
                    continue
                ls_time = min(deadline - 10, time.time() + 3.5)
                ls_code = local_search(refined, i * 7 + 1, ls_time)
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code
            else:
                # Kick restart on best solution
                if best_code is None:
                    code = greedy_cover(i)
                    refined = refine(code)
                    if len(refined) < best_len:
                        best_len = len(refined)
                        best_code = refined
                    continue
                k = kick_sizes[kick_idx % len(kick_sizes)]
                kick_idx += 1
                rng_k = random.Random(1000 + kick_idx)
                k_actual = min(k, len(best_code))
                kick_set = set(rng_k.sample(best_code, k_actual))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(2000 + kick_idx, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 10:
                    continue
                ls_code = local_search(re_refined, 3000 + kick_idx, min(deadline - 8, time.time() + 3.5))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Final polish
        if best_code is not None and time.time() < deadline - 5:
            ls_code = local_search(best_code, 42, deadline - 3)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            final_code = refine(best_code)
            for polish_iter in range(6):
                if time.time() > deadline - 2:
                    break
                if len(final_code) <= 150:
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

    # Generic case (shouldn't be called for this task, but included for completeness)
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

    # Generic greedy
    uncovered = np.ones(N, dtype=bool)
    code = []
    while np.any(uncovered):
        if time.time() > deadline:
            break
        best_idx = -1
        max_gain = 0
        for i in range(N):
            gain = np.sum(uncovered[neighbors[i]])
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if max_gain == 0:
            break
        code.append(best_idx)
        uncovered[neighbors[best_idx]] = False

    return [idx_to_word(i) for i in code]
