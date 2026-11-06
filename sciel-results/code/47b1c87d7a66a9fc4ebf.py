import numpy as np
import random
import time
import heapq

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0

    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n
        ball_size = 1 + n * (q - 1)

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

        # Precompute balls
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

        # Precompute cover_map
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        ball_sets = [set(b) for b in balls]
        ball_indices = [np.array(b, dtype=np.int32) for b in balls]

        def greedy_cover_conflict_aware(seed, start_code=None):
            """Greedy set-cover with conflict-aware scoring.
            Score = marginal coverage - lambda * conflict (overlap with already-selected codewords).
            This encourages selecting codewords that are far from each other."""
            rng = random.Random(seed)
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N
            in_code = [False] * N
            
            # Precompute pairwise conflict (Hamming distance overlap) - use ball intersection size
            # conflict[i][j] = number of words in ball_i that are also in ball_j
            # For efficiency, precompute as a sparse structure or just compute on the fly
            
            lambda_conflict = 0.15  # penalty for overlap with already selected

            if start_code:
                for c in start_code:
                    code.append(c)
                    in_code[c] = True
                    for w in balls[c]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1

                # Build heap from remaining candidates
                heap = []
                for i in range(N):
                    if not covered[i] and cnt[i] > 0 and not in_code[i]:
                        heapq.heappush(heap, (-cnt[i], i))
            else:
                initial_pool = rng.sample(range(N), min(N, 500))
                heap = [(-ball_size, i) for i in initial_pool]
                heapq.heapify(heap)

            while uncovered_count > 0:
                best_score = -1
                best_cand = -1
                heap_exhausted = False

                while heap:
                    neg_val, i = heapq.heappop(heap)
                    if cnt[i] != -neg_val or covered[i] or in_code[i]:
                        continue
                    break
                else:
                    # Heap exhausted; rebuild
                    heap = []
                    for i in range(N):
                        if not covered[i] and cnt[i] > 0 and not in_code[i]:
                            heapq.heappush(heap, (-cnt[i], i))
                    if not heap:
                        break
                    neg_val, i = heapq.heappop(heap)
                    if cnt[i] != -neg_val or covered[i] or in_code[i]:
                        continue

                # Evaluate top candidates from heap with conflict-aware scoring
                # Pull a few candidates and score them
                candidates = []
                current = i
                candidates.append(current)
                try:
                    for _ in range(4):
                        neg_val2, i2 = heapq.heappop(heap)
                        if cnt[i2] == -neg_val2 and not covered[i2] and not in_code[i2]:
                            candidates.append(i2)
                except IndexError:
                    pass

                # Score each candidate: coverage_gain - lambda * total_conflict_with_selected
                for c in candidates:
                    gain = cnt[c]
                    conflict = 0
                    for sel in code:
                        # Count overlap between ball_c and ball_sel
                        # Use set intersection for accuracy
                        overlap = len(ball_sets[c] & ball_sets[sel])
                        conflict += overlap
                    score = gain - lambda_conflict * conflict
                    if score > best_score:
                        best_score = score
                        best_cand = c

                if best_cand < 0:
                    break

                c = best_cand
                code.append(c)
                in_code[c] = True

                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0 and not in_code[c2] and not covered[c2]:
                                heapq.heappush(heap, (-cnt[c2], c2))

            return code

        def refine(code):
            if not code:
                return code

            cover_count = [0] * N
            for c in code:
                for w in balls[c]:
                    cover_count[w] += 1

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    safe = True
                    for w in balls[c]:
                        if cover_count[w] == 1:
                            safe = False
                            break
                    if safe:
                        for w in balls[c]:
                            cover_count[w] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = [0] * N
            for c in code_list:
                for w in ball_indices[c]:
                    cover_count[w] += 1

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]

                    newly_uncovered = []
                    for w in ball_indices[c_remove]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            newly_uncovered.append(w)

                    if not newly_uncovered:
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break

                    found = -1
                    if len(newly_uncovered) <= 6:
                        uncov_arr = np.array(newly_uncovered, dtype=np.int32)
                        best_cand = -1
                        for w in sorted(newly_uncovered, key=lambda w: len(cover_map[w])):
                            candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove]
                            if not candidates_w:
                                found = -1
                                break
                            if best_cand == -1:
                                for c2 in candidates_w:
                                    if all(w2 in ball_sets[c2] for w2 in newly_uncovered):
                                        best_cand = c2
                                        break
                            else:
                                if not all(w2 in ball_sets[best_cand] for w2 in newly_uncovered):
                                    best_cand = -1
                                    for c2 in candidates_w:
                                        if all(w2 in ball_sets[c2] for w2 in newly_uncovered):
                                            best_cand = c2
                                            break
                            if best_cand == -1 and len(candidates_w) == 0:
                                break
                        if best_cand >= 0:
                            found = best_cand

                    if found >= 0:
                        code_list[i] = found
                        in_code[c_remove] = False
                        in_code[found] = True
                        for w in ball_indices[found]:
                            cover_count[w] += 1
                        improved = True
                        break
                    else:
                        for w in ball_indices[c_remove]:
                            cover_count[w] += 1

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
                            uncov = set()
                            for w in ball_indices[c_rem1]:
                                cover_count[w] -= 1
                                if cover_count[w] == 0:
                                    uncov.add(w)
                            for w in ball_indices[c_rem2]:
                                cover_count[w] -= 1
                                if cover_count[w] == 0:
                                    uncov.add(w)
                            if uncov and len(uncov) <= 8:
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
                                    for c_add in cand_pool:
                                        bs = ball_sets[c_add]
                                        if all(w in bs for w in uncov):
                                            found2 = c_add
                                            break
                                if found2 >= 0:
                                    code_list[j2] = found2
                                    in_code[c_rem2] = False
                                    in_code[found2] = True
                                    for w in ball_indices[found2]:
                                        cover_count[w] += 1
                                    code_list.pop(i2)
                                    in_code[c_rem1] = False
                                    improved = True
                                    break
                                else:
                                    for w in ball_indices[c_rem1]:
                                        cover_count[w] += 1
                                    for w in ball_indices[c_rem2]:
                                        cover_count[w] += 1
                            else:
                                for w in ball_indices[c_rem1]:
                                    cover_count[w] += 1
                                for w in ball_indices[c_rem2]:
                                    cover_count[w] += 1
                            if improved:
                                break
                        if improved:
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        num_restarts = 80
        for i in range(num_restarts):
            if time.time() > deadline - 15:
                break
            # Alternate between standard and conflict-aware greedy
            if i % 2 == 0:
                code = greedy_cover_conflict_aware(i)
            else:
                code = greedy_cover_conflict_aware(i, start_code=None)
            
            refined = refine(code)
            if time.time() > deadline - 10:
                if len(refined) < best_len:
                    best_len = len(refined)
                    best_code = refined
                continue
            ls_time = min(deadline - 8, time.time() + 3.5)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        if best_code is not None:
            kick_sizes = [3, 4, 5, 6, 7, 4, 5, 6, 3, 5, 4, 6, 8, 7, 5, 4]
            for kick_iter, k in enumerate(kick_sizes):
                if time.time() > deadline - 12:
                    break
                rng_k = random.Random(1000 + kick_iter)
                k_actual = min(k, len(best_code))
                kick_set = set(rng_k.sample(best_code, k_actual))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover_conflict_aware(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 10:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 8, time.time() + 3.0))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        if best_code is not None and time.time() < deadline - 5:
            ls_code = local_search(best_code, 42, deadline - 3)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            final_code = refine(best_code)
            for polish_iter in range(5):
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
