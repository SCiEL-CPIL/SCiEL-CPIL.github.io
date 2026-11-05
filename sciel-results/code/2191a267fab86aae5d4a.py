import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin (150s limit, keep 32s buffer)

    if q != 3 or n != 6 or radius != 1:
        # Fallback: return all words (valid but large)
        all_words = []
        for i in range(q ** n):
            word = [(i // (q ** p)) % q for p in range(n)]
            all_words.append(word)
        return all_words

    total = 3 ** 6
    powers = [3 ** p for p in range(6)]

    # Precompute neighbors as numpy array (13 neighbors: self + 12 at dist 1)
    neighbors = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        d = [(i // powers[p]) % 3 for p in range(6)]
        neighbors[i, 0] = i
        k = 1
        for p in range(6):
            for alt in range(3):
                if alt != d[p]:
                    new_i = i + (alt - d[p]) * powers[p]
                    neighbors[i, k] = new_i
                    k += 1

    # Precompute neighbor sets for fast subset checks
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(total)]

    # Helper to convert index to list
    def idx_to_list(idx):
        return [(idx // powers[p]) % 3 for p in range(6)]

    # ---------- Initial solutions ----------
    # (A) Tetracode direct sum: 9 * 3^2 = 81 words, valid covering code
    def tetracode_direct_sum():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        words = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                words.append(h + list(tail))
        return words

    # (B) Greedy set cover
    def greedy_set_cover():
        uncovered = np.ones(total, dtype=np.int8)
        codewords = []
        while uncovered.sum() > 0:
            if time.time() > deadline - 60:
                break
            counts = np.sum(uncovered[neighbors], axis=1)
            best_c = int(np.argmax(counts))
            if counts[best_c] == 0:
                break
            codewords.append(best_c)
            uncovered[neighbors[best_c]] = 0
        if uncovered.sum() != 0:
            while uncovered.sum() > 0:
                c = int(np.random.randint(total))
                codewords.append(c)
                uncovered[neighbors[c]] = 0
        return codewords

    def words_to_indices(words):
        idxs = []
        for w in words:
            idx = 0
            for p in range(n):
                idx += w[p] * powers[p]
            idxs.append(idx)
        return idxs

    # Build initial candidates from both constructions
    tc_words = tetracode_direct_sum()
    tc_indices = words_to_indices(tc_words)

    greedy_indices = greedy_set_cover()

    # Choose the smaller valid initial solution
    random.seed(42)
    np.random.seed(42)

    initial_options = [tc_indices, greedy_indices]
    # Validate each (all must cover whole space)
    def is_valid(indices):
        covered = np.zeros(total, dtype=np.int8)
        for c in indices:
            covered[neighbors[c]] = 1
        return covered.sum() == total

    valid_options = [opt for opt in initial_options if is_valid(opt)]
    if not valid_options:
        valid_options = [tc_indices]
    initial = min(valid_options, key=len)

    codewords = list(initial)
    # Ensure uniqueness
    seen = set()
    unique = []
    for c in codewords:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    codewords = unique

    # Initialize coverage counts
    coverage_counts = np.zeros(total, dtype=np.int32)
    for c in codewords:
        coverage_counts[neighbors[c]] += 1

    in_set = set(codewords)
    best_codewords = list(codewords)
    best_size = len(codewords)

    # ---------- Local search: redundant removal + swaps ----------
    def try_refine(codewords, coverage_counts, in_set, budget):
        nonlocal best_codewords, best_size
        T = 1.0
        T_min = 0.005
        cooling = 0.9995
        while time.time() < budget:
            # Find redundant words
            redundant_list = []
            for idx, c in enumerate(codewords):
                if coverage_counts[neighbors[c]].min() >= 2:
                    redundant_list.append(idx)
            if redundant_list:
                if len(redundant_list) > 1:
                    slacks = [coverage_counts[neighbors[codewords[i]]].min() for i in redundant_list]
                    max_slack = max(slacks)
                    best_candidates = [i for i, s in zip(redundant_list, slacks) if s == max_slack]
                    idx_to_remove = random.choice(best_candidates)
                else:
                    idx_to_remove = redundant_list[0]
                c_to_remove = codewords[idx_to_remove]
                codewords.pop(idx_to_remove)
                coverage_counts[neighbors[c_to_remove]] -= 1
                in_set.discard(c_to_remove)
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                T = max(T * cooling, T_min)
                continue
            else:
                # No redundant words. Try swaps.
                found_swap = False
                num_tries = 4
                for _ in range(num_tries):
                    idx_to_remove = random.randint(0, len(codewords) - 1)
                    c_to_remove = codewords[idx_to_remove]
                    crit_mask = (coverage_counts[neighbors[c_to_remove]] == 1)
                    critical_words = neighbors[c_to_remove][crit_mask]
                    if len(critical_words) == 0:
                        continue
                    w0 = int(critical_words[0])
                    candidates = neighbors[w0]
                    valid_candidates = []
                    for cand in candidates:
                        if cand == c_to_remove:
                            continue
                        if all(w in neighbor_sets[int(cand)] for w in critical_words):
                            valid_candidates.append(int(cand))
                    if not valid_candidates:
                        continue
                    # Score: prefer adding coverage to less-covered areas
                    best_cand = None
                    best_score = -1
                    for cand in valid_candidates:
                        score = -coverage_counts[neighbors[cand]].sum()
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                    c_to_add = best_cand if best_cand is not None else random.choice(valid_candidates)
                    codewords[idx_to_remove] = c_to_add
                    coverage_counts[neighbors[c_to_remove]] -= 1
                    coverage_counts[neighbors[c_to_add]] += 1
                    in_set.discard(c_to_remove)
                    in_set.add(c_to_add)
                    found_swap = True
                    break
                T = max(T * cooling, T_min)
        return codewords, coverage_counts, in_set

    # Multi-start refinement: run a few independent refinement passes with
    # different random seeds, keeping the best.
    num_starts = 4
    per_start_budget = (deadline - 6) / num_starts
    for s in range(num_starts):
        random.seed(100 + s)
        np.random.seed(200 + s)
        # Restart from the current best
        cur = list(best_codewords)
        cc = np.zeros(total, dtype=np.int32)
        for c in cur:
            cc[neighbors[c]] += 1
        in_s = set(cur)
        budget = start_time + (s + 1) * per_start_budget
        if budget > deadline - 3:
            budget = deadline - 3
        cur, cc, in_s = try_refine(cur, cc, in_s, budget)
        # Final cleanup
        improved = True
        while improved:
            improved = False
            for idx in range(len(cur)):
                c = cur[idx]
                if cc[neighbors[c]].min() >= 2:
                    cur.pop(idx)
                    cc[neighbors[c]] -= 1
                    improved = True
                    break
        if len(cur) < best_size:
            best_size = len(cur)
            best_codewords = list(cur)

    final_indices = best_codewords
    return [idx_to_list(idx) for idx in final_indices]

# EVOLVE-BLOCK-END
