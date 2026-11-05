import numpy as np
import random
import time
import heapq


def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 112.0  # safety margin for 150s limit

    # Specialized, fast path for the known hard instance q=3, n=7, r=1.
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n  # 2187
        ball_size = 1 + n * (q - 1)  # 15

        # ---- vectorized ball / neighbor construction ----
        powers = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int64)
        idx = np.arange(N, dtype=np.int64)
        digits = np.zeros((N, n), dtype=np.int64)
        for i in range(n):
            digits[:, i] = (idx // powers[i]) % 3

        base = idx.reshape(N, 1)
        # For each position i and each replacement value v != current digit:
        # neighbor index = idx + (v - digit_i) * power_i
        delta_list = []
        for i in range(n):
            dcol = digits[:, i].reshape(N, 1)
            p = powers[i]
            for v in range(3):
                delta_list.append((v - dcol) * p)
        # neighbors: shape (N, 15), first column = self
        neigh = np.concatenate([idx.reshape(N, 1)] +
                               [d.reshape(N, 1) + base for d in delta_list],
                               axis=1)
        neigh = neigh.astype(np.int64)  # (N, 15)

        # ---- cover_map: for each word w, the list of codewords whose ball contains w ----
        # Build by inverting the neighbor relation.
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in neigh[c]:
                cover_map[w].append(c)
        # Precompute candidate arrays (exclude self) for swap pools.
        cover_map_arr = [np.array(cm, dtype=np.int64) for cm in cover_map]

        def int_to_base3(num, length):
            res = [0] * length
            for i in range(length - 1, -1, -1):
                res[i] = num % 3
                num //= 3
            return res

        def greedy_cover(seed, start_code=None, time_limit=deadline):
            """Vectorized greedy set cover (max uncovered gain)."""
            rng = random.Random(seed)
            covered = np.zeros(N, dtype=bool)
            code = []

            if start_code:
                for c in start_code:
                    code.append(int(c))
                    covered[neigh[c]] = True
                gain = (neigh[~covered].sum(axis=1) if False else None)

            while True:
                if time.time() > time_limit:
                    break
                if covered.all():
                    break
                # gain[c] = number of currently uncovered words in ball c
                gain = neigh[~covered]
                gain = gain.sum(axis=1) if gain.size else np.zeros(N, dtype=np.int64)
                best = int(np.argmax(gain))
                if gain[best] <= 0:
                    break
                code.append(best)
                covered[neigh[best]] = True

            return code

        def refine(code):
            """Remove redundant codewords using cover counts."""
            code_list = list(code)
            if not code_list:
                return code_list
            cover_count = np.zeros(N, dtype=np.int64)
            for c in code_list:
                cover_count[neigh[c]] += 1

            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    if (cover_count[neigh[c]] > 1).all():
                        cover_count[neigh[c]] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            """Local search: deletions, 1-swaps, limited 2-swaps."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            in_code = np.zeros(N, dtype=bool)
            for c in code_list:
                in_code[c] = True

            cover_count = np.zeros(N, dtype=np.int64)
            for c in code_list:
                cover_count[neigh[c]] += 1

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)

                # Pass 1: deletions + 1-swaps
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]
                    ball = neigh[c_remove]

                    # Try deletion
                    cover_count[ball] -= 1
                    newly = ball[cover_count[ball] == 0]
                    if newly.size == 0:
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break

                    # Try 1-swap
                    found = -1
                    if newly.size <= 10:
                        hole_mask = np.zeros(N, dtype=bool)
                        hole_mask[newly] = True
                        crit = newly[np.argsort(
                            [len(cover_map_arr[int(w)]) for w in newly])]
                        cand_pool = None
                        for w in crit:
                            w = int(w)
                            cands = cover_map_arr[w]
                            mask = (~in_code[cands]) & (cands != c_remove)
                            cands = cands[mask]
                            if cand_pool is None:
                                cand_pool = set(int(x) for x in cands)
                            else:
                                cand_pool.intersection_update(int(x) for x in cands)
                            if not cand_pool:
                                break
                        if cand_pool:
                            for c_add in cand_pool:
                                if hole_mask[neigh[c_add]].all():
                                    found = c_add
                                    break

                    if found >= 0:
                        code_list[i] = found
                        in_code[c_remove] = False
                        in_code[found] = True
                        cover_count[neigh[found]] += 1
                        improved = True
                        break
                    else:
                        cover_count[ball] += 1

                # Pass 2: limited 2-swaps
                if not improved and time.time() < time_limit and len(code_list) >= 3:
                    n_codes = len(code_list)
                    num_pairs = min(80, n_codes * (n_codes - 1) // 2)
                    pairs = set()
                    while len(pairs) < num_pairs:
                        i2 = rng.randint(0, n_codes - 1)
                        j2 = rng.randint(0, n_codes - 1)
                        if i2 != j2:
                            if i2 > j2:
                                i2, j2 = j2, i2
                            pairs.add((i2, j2))
                    for i2, j2 in pairs:
                        if time.time() > time_limit:
                            break
                        c1 = code_list[i2]
                        c2 = code_list[j2]
                        b1 = neigh[c1]
                        b2 = neigh[c2]
                        cover_count[b1] -= 1
                        cover_count[b2] -= 1
                        uncov = np.concatenate([b1, b2])[cover_count[np.concatenate([b1, b2])] == 0]
                        if uncov.size and uncov.size <= 12:
                            hole_mask = np.zeros(N, dtype=bool)
                            hole_mask[uncov] = True
                            crit = uncov[np.argsort(
                                [len(cover_map_arr[int(w)]) for w in uncov])]
                            cand_pool = None
                            for w in crit:
                                w = int(w)
                                cands = cover_map_arr[w]
                                mask = (~in_code[cands]) & (cands != c1) & (cands != c2)
                                cands = cands[mask]
                                if cand_pool is None:
                                    cand_pool = set(int(x) for x in cands)
                                else:
                                    cand_pool.intersection_update(int(x) for x in cands)
                                if not cand_pool:
                                    break
                            found2 = -1
                            if cand_pool:
                                for c_add in cand_pool:
                                    if hole_mask[neigh[c_add]].all():
                                        found2 = c_add
                                        break
                            if found2 >= 0:
                                code_list[j2] = found2
                                in_code[c2] = False
                                in_code[found2] = True
                                cover_count[neigh[found2]] += 1
                                code_list.pop(i2)
                                in_code[c1] = False
                                improved = True
                                break
                        cover_count[b1] += 1
                        cover_count[b2] += 1
                        if improved:
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        # ---- Phase A: many fast greedy restarts + refine + short local search ----
        # The vectorized greedy is cheap, so we can afford many restarts.
        num_restarts = 40
        for i in range(num_restarts):
            if time.time() > deadline - 25:
                break
            code = greedy_cover(i, time_limit=deadline - 25)
            refined = refine(code)
            if time.time() > deadline - 12:
                if len(refined) < best_len:
                    best_len = len(refined)
                    best_code = refined
                continue
            ls_code = local_search(refined, i * 7 + 1, min(deadline - 10, time.time() + 2.5))
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # ---- Phase B: kick / re-greedy from the best code found so far ----
        if best_code is not None:
            kick_sizes = [3, 4, 5, 6, 7, 8, 5, 6, 4, 7, 5, 8, 6, 4, 7, 5]
            for kick_iter, k in enumerate(kick_sizes):
                if time.time() > deadline - 14:
                    break
                rng_k = random.Random(1000 + kick_iter)
                k_actual = min(k, len(best_code))
                kick_set = set(rng_k.sample(best_code, k_actual))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(2000 + kick_iter, start_code=remaining,
                                       time_limit=deadline - 14)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 12:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter,
                                       min(deadline - 10, time.time() + 3.0))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # ---- Phase C: final deep local search polish ----
        if best_code is not None and time.time() < deadline - 6:
            ls_code = local_search(best_code, 42, deadline - 4)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            final_code = refine(best_code)
            for polish_iter in range(4):
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

        result = [int_to_base3(c, n) for c in final_code]
        return result

    # ---- Generic fallback (greedy, not optimized) ----
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
    neighbors = np.empty((N, nb_count), dtype=np.int64)
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
        gain = neighbors[uncovered].sum(axis=1) if np.any(uncovered) else np.zeros(N)
        best_idx = int(np.argmax(gain))
        if gain[best_idx] <= 0:
            break
        code.append(best_idx)
        uncovered[neighbors[best_idx]] = False

    return [idx_to_word(i) for i in code]
