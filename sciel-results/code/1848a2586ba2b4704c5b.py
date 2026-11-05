import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 147.0  # Safety margin before 150s kill

    if q != 3 or n != 6 or radius != 1:
        all_words = []
        for i in range(q ** n):
            word = [(i // (q ** p)) % q for p in range(n)]
            all_words.append(word)
        return all_words

    total = 3 ** 6
    powers = [3 ** p for p in range(6)]

    # Precompute neighbors: self + 12 at Hamming distance 1 (13 total)
    neighbors = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        d = [(i // powers[p]) % 3 for p in range(6)]
        neighbors[i, 0] = i
        k = 1
        for p in range(6):
            for alt in range(3):
                if alt != d[p]:
                    neighbors[i, k] = i + (alt - d[p]) * powers[p]
                    k += 1

    # Precompute neighbor sets for O(1) membership / subset checks
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(total)]

    def idx_to_list(idx):
        return [(idx // powers[p]) % 3 for p in range(6)]

    def greedy_init(rng):
        """Greedy set cover with a touch of randomization to diversify starts."""
        uncovered = np.ones(total, dtype=np.int8)
        codewords = []
        while uncovered.sum() > 0:
            counts = np.sum(uncovered[neighbors], axis=1)
            # Among the top few, pick randomly to diversify
            max_count = counts.max()
            if max_count == 0:
                break
            top = np.where(counts == max_count)[0]
            if len(top) > 1 and rng.random() < 0.3:
                best_c = int(rng.choice(top))
            else:
                best_c = int(np.argmax(counts))
            codewords.append(best_c)
            uncovered[neighbors[best_c]] = 0
        if uncovered.sum() != 0:
            while uncovered.sum() > 0:
                c = int(rng.randint(total))
                codewords.append(c)
                uncovered[neighbors[c]] = 0
        return codewords

    def reduce_redundant(codewords, coverage_counts, in_set):
        """Remove any redundant codewords (min coverage of neighbors >= 2)."""
        improved = True
        while improved:
            improved = False
            for idx in range(len(codewords)):
                c = codewords[idx]
                if coverage_counts[neighbors[c]].min() >= 2:
                    codewords.pop(idx)
                    coverage_counts[neighbors[c]] -= 1
                    in_set.discard(c)
                    improved = True
                    break
        return codewords, coverage_counts, in_set

    def local_search(codewords, coverage_counts, in_set, rng, time_limit):
        """Simulated-annealing-style local search: remove redundant, then swap."""
        best_size = len(codewords)
        best_codewords = list(codewords)

        T = 1.0
        T_min = 0.005
        cooling = 0.9997
        end = time.time() + time_limit

        while time.time() < end:
            # 1) Remove a redundant word if any exists
            redundant_idx = None
            best_slack = -1
            for idx, c in enumerate(codewords):
                mn = coverage_counts[neighbors[c]].min()
                if mn >= 2:
                    if mn > best_slack:
                        best_slack = mn
                        redundant_idx = idx
            if redundant_idx is not None:
                c_to_remove = codewords[redundant_idx]
                codewords.pop(redundant_idx)
                coverage_counts[neighbors[c_to_remove]] -= 1
                in_set.discard(c_to_remove)
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                T = max(T * cooling, T_min)
                continue

            # 2) No redundant: try swaps (remove one, add one covering its critical words)
            found = False
            for _ in range(4):
                idx_to_remove = rng.randint(0, len(codewords) - 1)
                c_to_remove = codewords[idx_to_remove]
                crit = neighbors[c_to_remove][coverage_counts[neighbors[c_to_remove]] == 1]
                if len(crit) == 0:
                    continue
                w0 = int(crit[0])
                candidates = neighbors[w0]
                valid = []
                for cand in candidates:
                    cand = int(cand)
                    if cand == c_to_remove:
                        continue
                    if all(int(w) in neighbor_sets[cand] for w in crit):
                        # Score: prefer candidates that add most redundancy
                        # (higher coverage on neighbors not in crit)
                        add_cov = coverage_counts[neighbors[cand]].sum()
                        valid.append((add_cov, cand))
                if not valid:
                    continue
                # Weighted choice toward higher redundancy
                valid.sort(reverse=True)
                # Pick from top with some randomness
                k = min(len(valid), 3)
                chosen = valid[rng.randint(0, k - 1)][1]
                c_to_add = chosen
                codewords[idx_to_remove] = c_to_add
                coverage_counts[neighbors[c_to_remove]] -= 1
                coverage_counts[neighbors[c_to_add]] += 1
                in_set.discard(c_to_remove)
                in_set.add(c_to_add)
                found = True
                break
            T = max(T * cooling, T_min)

        # Final polish
        codewords, coverage_counts, in_set = reduce_redundant(codewords, coverage_counts, in_set)
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = list(codewords)
        return best_codewords

    # Global best across restarts
    global_best = None

    # Use multiple restarts with diverse seeds to escape local minima
    base_seed = 42
    rng = random.Random(base_seed)
    np.random.seed(base_seed)

    # First run: greedy + local search with a generous budget
    codewords = greedy_init(rng)
    coverage_counts = np.zeros(total, dtype=np.int32)
    for c in codewords:
        coverage_counts[neighbors[c]] += 1
    in_set = set(codewords)
    codewords, coverage_counts, in_set = reduce_redundant(codewords, coverage_counts, in_set)

    # Budget for first local search
    first_budget = min(40.0, max(5.0, deadline - (time.time() - start_time) - 5.0))
    best = local_search(codewords, coverage_counts, in_set, rng, first_budget)
    global_best = best

    # Additional restarts if time remains, with diverse randomization
    restart_budget = 25.0
    max_restarts = 6
    for r in range(max_restarts):
        if time.time() > deadline - restart_budget - 5.0:
            break
        seed = base_seed + 1000 * (r + 1)
        rng2 = random.Random(seed)
        np.random.seed(seed)
        codewords = greedy_init(rng2)
        coverage_counts = np.zeros(total, dtype=np.int32)
        for c in codewords:
            coverage_counts[neighbors[c]] += 1
        in_set = set(codewords)
        codewords, coverage_counts, in_set = reduce_redundant(codewords, coverage_counts, in_set)
        # Local search for the remaining budget
        remaining = deadline - (time.time() - start_time) - 5.0
        budget = min(restart_budget, max(3.0, remaining / (max_restarts - r)))
        best_r = local_search(codewords, coverage_counts, in_set, rng2, budget)
        if global_best is None or len(best_r) < len(global_best):
            global_best = best_r

    return [idx_to_list(c) for c in global_best]
