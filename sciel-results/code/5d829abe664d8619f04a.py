import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before the 120s wall-clock kill

    # Fallback for non-specific parameters (not expected in this task, but robust)
    if q != 3 or n != 7 or radius != 1:
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}

        def get_neighbors(w):
            nbs = [w]
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    new_w = list(w)
                    new_w[pos] = new_val
                    nbs.append(tuple(new_w))
            return nbs

        neighbors = [get_neighbors(w) for w in all_words]
        neighbors_idx = [[word_to_idx[t] for t in nbs] for nbs in neighbors]

        uncovered = set(range(N))
        code = []
        while uncovered:
            best_c = -1
            max_cov = 0
            for c in range(N):
                if c in code:
                    continue
                cnt = 0
                for nb in neighbors_idx[c]:
                    if nb in uncovered:
                        cnt += 1
                if cnt > max_cov:
                    max_cov = cnt
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            for nb in neighbors_idx[best_c]:
                uncovered.discard(nb)
        return [list(all_words[c]) for c in code]

    # ---------- Specific optimization for q=3, n=7, radius=1 ----------
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Dense neighbor matrix: neighbors_arr[i] = the 15 indices covered by word i
    neighbors_arr = np.zeros((N, 1 + n * (q - 1)), dtype=np.int32)
    for i in range(N):
        w = idx_to_tuple(i)
        neighbors_arr[i, 0] = i
        col = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = w.copy()
                new_w[pos] = new_val
                neighbors_arr[i, col] = tuple_to_idx(new_w)
                col += 1

    # Fast membership sets for swap candidate search
    neighbor_sets = [set(neighbors_arr[i].tolist()) for i in range(N)]

    # ---------- Phase 1: Greedy Set-Cover with vectorized gain computation ----------
    coverage = np.zeros(N, dtype=np.int32)
    uncovered_mask = np.ones(N, dtype=bool)
    code_set = set()
    code_list = []

    while uncovered_mask.any() and time.time() < deadline:
        # Gain of candidate c = number of uncovered words it covers.
        # Vectorized gather + sum (fast for N=2187, 15 neighbors each).
        counts = np.take(uncovered_mask, neighbors_arr).sum(axis=1)
        for c in code_set:
            counts[c] = 0
        best_c = int(np.argmax(counts))
        best_count = counts[best_c]
        if best_count == 0:
            break
        code_list.append(best_c)
        code_set.add(best_c)
        coverage[neighbors_arr[best_c]] += 1
        uncovered_mask[neighbors_arr[best_c]] = False

    best_size = len(code_set)
    best_code_set = set(code_set)

    # ---------- Phase 2: Local search ----------
    random.seed(42)
    no_improvement_counter = 0

    def try_swap(c):
        """Try to replace non-removable codeword c with a better-positioned word.
        Returns True if a swap was made (coverage/code_set mutated in place)."""
        if np.min(coverage[neighbors_arr[c]]) > 1:
            return False
        # Critical points of c: neighbors whose only cover is c.
        crit_points = [int(i) for i in neighbors_arr[c] if coverage[i] == 1]
        if not crit_points:
            return False

        # Candidate words are neighbors of a critical point (must cover it).
        first = crit_points[0]
        cands = [w for w in neighbor_sets[first] if w not in code_set]

        # Strategy 1: a word covering all critical points (pure removal-quality swap).
        for w in cands:
            if all(k in neighbor_sets[w] for k in crit_points):
                score = int(np.sum(coverage[neighbors_arr[w]] == 1))
                code_set.discard(c)
                coverage[neighbors_arr[c]] -= 1
                code_set.add(w)
                coverage[neighbors_arr[w]] += 1
                return True

        # Strategy 2: relocation - also cover points that would become uncovered.
        temp_counts = coverage.copy()
        temp_counts[neighbors_arr[c]] -= 1
        would_be_uncovered = [int(i) for i in neighbors_arr[c] if temp_counts[i] == 0]
        if not would_be_uncovered:
            return False
        all_needed = set(crit_points) | set(would_be_uncovered)
        for w in cands:
            if all(k in neighbor_sets[w] for k in all_needed):
                code_set.discard(c)
                coverage[neighbors_arr[c]] -= 1
                code_set.add(w)
                coverage[neighbors_arr[w]] += 1
                return True
        return False

    def greedy_repair(uncovered_set):
        """Add codewords to cover uncovered_set; mutate coverage/code_set in place."""
        temp_uncovered = set(uncovered_set)
        while temp_uncovered and time.time() < deadline:
            best_c = -1
            max_gain = 0
            for c in range(N):
                if c in code_set:
                    continue
                gain = 0
                for nb in neighbors_arr[c]:
                    if nb in temp_uncovered:
                        gain += 1
                        if gain > max_gain:
                            max_gain = gain
                            best_c = c
                            if gain == len(temp_uncovered):
                                break
                if gain > max_gain:
                    max_gain = gain
                    best_c = c
                    if max_gain == len(temp_uncovered):
                        break
            if best_c == -1 or max_gain == 0:
                return False
            code_set.add(best_c)
            coverage[neighbors_arr[best_c]] += 1
            for nb in neighbors_arr[best_c]:
                temp_uncovered.discard(nb)
        return len(temp_uncovered) == 0

    while time.time() < deadline:
        # (a) Remove redundant codewords (all their neighbors have coverage > 1).
        removed_any = False
        for c in list(code_set):
            if np.min(coverage[neighbors_arr[c]]) > 1:
                code_set.discard(c)
                coverage[neighbors_arr[c]] -= 1
                removed_any = True
                best_size = len(code_set)
                best_code_set = set(code_set)
                no_improvement_counter = 0
                break
        if removed_any:
            continue

        # (b) Try swap/relocation moves on non-removable codewords.
        swapped_any = False
        for c in list(code_set):
            if try_swap(c):
                swapped_any = True
                best_size = len(code_set)
                best_code_set = set(code_set)
                no_improvement_counter = 0
                break
        if swapped_any:
            continue

        # (c) Kick: remove k random codewords, repair, accept if smaller.
        k = random.choice([1, 2, 3, 4, 5])
        if len(code_set) <= k:
            break
        removed_indices = random.sample(list(code_set), k)

        temp_coverage = coverage.copy()
        temp_code_set = set(code_set)
        for c in removed_indices:
            temp_coverage[neighbors_arr[c]] -= 1
            temp_code_set.discard(c)

        uncovered_idx = np.where(temp_coverage == 0)[0].tolist()
        if len(uncovered_idx) == 0:
            code_set = temp_code_set
            coverage = temp_coverage
            best_size = len(code_set)
            best_code_set = set(code_set)
            no_improvement_counter = 0
            continue

        # Greedy repair with a cap to keep each kick fast.
        current_uncovered = set(uncovered_idx)
        max_add = len(removed_indices) + 2
        added = 0
        while current_uncovered and added < max_add and time.time() < deadline:
            best_c = -1
            max_gain = 0
            for c in range(N):
                if c in temp_code_set:
                    continue
                gain = 0
                for nb in neighbors_arr[c]:
                    if nb in current_uncovered:
                        gain += 1
                        if gain > max_gain:
                            max_gain = gain
                            best_c = c
                            if gain == len(current_uncovered):
                                break
                if gain > max_gain:
                    max_gain = gain
                    best_c = c
                    if max_gain == len(current_uncovered):
                        break
            if best_c == -1 or max_gain == 0:
                break
            temp_code_set.add(best_c)
            temp_coverage[neighbors_arr[best_c]] += 1
            for nb in neighbors_arr[best_c]:
                current_uncovered.discard(nb)
            added += 1

        # Accept only if it is a valid cover AND strictly smaller.
        if len(current_uncovered) == 0 and len(temp_code_set) < len(code_set):
            code_set = temp_code_set
            coverage = temp_coverage
            best_size = len(code_set)
            best_code_set = set(code_set)
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if no_improvement_counter > 250:
            break

    # Final: use the best set found.
    if len(best_code_set) < len(code_set):
        final_code_set = best_code_set
    else:
        final_code_set = code_set

    # Fast full-coverage guard: ensure the returned set truly covers every word.
    final_cov = np.zeros(N, dtype=np.int32)
    for c in final_code_set:
        final_cov[neighbors_arr[c]] += 1
    if np.any(final_cov == 0):
        # Repair any gap (should not happen) to guarantee validity.
        uncovered = set(np.where(final_cov == 0)[0].tolist())
        for c in range(N):
            if c in final_code_set:
                continue
            if any(nb in uncovered for nb in neighbors_arr[c]):
                final_code_set.add(c)
                for nb in neighbors_arr[c]:
                    final_cov[nb] += 1
                    if nb in uncovered:
                        uncovered.discard(nb)
            if not uncovered:
                break

    result = [list(idx_to_tuple(c)) for c in final_code_set]
    return result
