import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    start_time = time.time()
    deadline = start_time + 140  # safety margin vs 150s kill

    # ---------- index <-> tuple ----------
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # ---------- neighbors of each word (radius-1 ball) ----------
    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        w = idx_to_tuple(i)
        col = 0
        neighbors_arr[i, col] = i
        col += 1
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                neighbors_arr[i, col] = tuple_to_idx(tuple(new_w))
                col += 1

    # ---------- reverse cover: which words are covered by each word ----------
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

    # ---------- per-codeword critical-word list (words it uniquely covers) ----------
    # critical[c] = list of words whose only cover (in the current code) is c
    # We recompute lazily inside local search; here we just precompute the
    # static set of words each codeword covers for fast reference.

    # ---------- greedy construction (weighted) ----------
    def greedy_construct(random_seed, weighted):
        rng = random.Random(random_seed)
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []

        if weighted:
            # static "rarity" weight: words with fewer covers are harder
            rarity = 1.0 / (reverse_cover_counts.astype(np.float64) + 1.0)
            while uncovered.any():
                gains = np.zeros(total_words, dtype=np.float64)
                # sum of rarity over the covered set
                gains += np.sum(rarity[neighbors_arr], axis=1)
                valid = gains > 0
                if not valid.any():
                    break
                # break ties randomly
                max_gain = gains[valid].max()
                best = np.where((gains >= max_gain - 1e-9) & valid)[0]
                if len(best) > 1:
                    chosen = best[rng.randrange(len(best))]
                else:
                    chosen = best[0]
                code.append(chosen)
                uncovered[neighbors_arr[chosen, :]] = False
        else:
            while uncovered.any():
                uncovered_expanded = uncovered[neighbors_arr]
                gains = uncovered_expanded.sum(axis=1)
                valid = gains > 0
                if not valid.any():
                    break
                max_gain = gains[valid].max()
                best = np.where((gains == max_gain) & valid)[0]
                if len(best) > 1:
                    chosen = best[rng.randrange(len(best))]
                else:
                    chosen = best[0]
                code.append(chosen)
                uncovered[neighbors_arr[chosen, :]] = False
        return code

    # ---------- reduction: drop redundant codewords ----------
    def reduce_code_fast(code):
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    # ---------- vectorized local search ----------
    def local_search_improve(code, deadline):
        code = list(code)
        rng = random.Random()
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1

        # Precompute, for each word, the list of words it uniquely covers
        # as a function of current counts — but counts change, so we compute
        # critical words on the fly per candidate.

        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            rng.shuffle(indices)

            for i in indices:
                if time.time() >= deadline:
                    break

                c_remove = code[i]
                counts[neighbors_arr[c_remove, :]] -= 1

                if np.any(counts == 0):
                    counts[neighbors_arr[c_remove, :]] += 1

                    # critical words: covered only by c_remove
                    cov = neighbors_arr[c_remove, :]
                    critical_words = cov[counts[cov] == 1]

                    if len(critical_words) == 0:
                        continue

                    # candidates = intersection of reverse_cover over critical words
                    candidate_set = None
                    for cw in critical_words:
                        rc = reverse_cover_arr[cw, :reverse_cover_counts[cw]]
                        if candidate_set is None:
                            candidate_set = set(rc.tolist())
                        else:
                            candidate_set &= set(rc.tolist())

                    if candidate_set is None or len(candidate_set) == 0:
                        continue

                    candidate_set.discard(c_remove)
                    if len(candidate_set) == 0:
                        continue

                    # pick a candidate that minimizes new overlaps (heuristic:
                    # prefer one that covers the most already-covered words so
                    # we don't create new redundancy)
                    best_w = None
                    best_score = -1
                    cand_list = list(candidate_set)
                    # sample if too many
                    if len(cand_list) > 40:
                        cand_list = rng.sample(cand_list, 40)
                    for cw in cand_list:
                        # gain: how many of its covered words are already covered
                        s = np.sum(counts[neighbors_arr[cw, :]] > 0)
                        if s > best_score:
                            best_score = s
                            best_w = cw
                    if best_w is None:
                        best_w = cand_list[0]

                    code[i] = best_w
                    counts[neighbors_arr[c_remove, :]] -= 1
                    counts[neighbors_arr[best_w, :]] += 1
                    improved = True
                    break
                else:
                    code.pop(i)
                    improved = True
                    break

        return code

    # ---------- main loop: many restarts ----------
    best_code = None
    best_size = float('inf')

    seed = 0
    while time.time() < deadline - 12:
        # alternate weighted / plain greedy
        weighted = (seed % 2 == 0)
        code = greedy_construct(seed, weighted)
        code = reduce_code_fast(code)

        # polish if we have time
        if time.time() < deadline - 12:
            code = local_search_improve(code, deadline - 8)
            code = reduce_code_fast(code)

        if len(code) < best_size:
            best_size = len(code)
            best_code = code

        seed += 1

    # final polish on best
    if best_code is not None and time.time() < deadline - 3:
        best_code = local_search_improve(best_code, deadline - 1)
        best_code = reduce_code_fast(best_code)

    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))

    return result
