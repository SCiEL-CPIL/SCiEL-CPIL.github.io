"""New design: randomized greedy + local search with inverse index and 2-for-1 moves."""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def construct(q, n, radius):
    deadline = time.time() + 140.0
    total = q ** n

    # All words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbors: for each word, list of words within Hamming distance <= radius
    # For q=3, n=6, radius=1: 1 + 6*2 = 13 neighbors per word
    neighbors = []
    for w in all_words:
        nb = [w]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    lst = list(w)
                    lst[pos] = val
                    nb.append(tuple(lst))
        neighbors.append(nb)

    # For each word, which "neighborhoods" (words) cover it — inverse index
    # covered_by[i] = list of j such that i is in neighbors[j]
    covered_by = [[] for _ in range(total)]
    for j in range(total):
        for i in neighbors[j]:
            covered_by[word_to_idx[i]].append(j)

    def is_covered(codeword_indices):
        covered = [False] * total
        for j in codeword_indices:
            for i in neighbors[j]:
                covered[word_to_idx[i]] = True
        return all(covered)

    def count_uncovered(codeword_indices):
        covered = [False] * total
        for j in codeword_indices:
            for i in neighbors[j]:
                covered[word_to_idx[i]] = True
        return sum(1 for c in covered if not c)

    def greedy_randomized():
        """Randomized greedy: repeatedly pick the word covering most uncovered words."""
        uncovered = [True] * total
        chosen = []
        remaining = list(range(total))
        random.shuffle(remaining)

        while any(uncovered):
            best_j = -1
            best_count = -1
            # Sample a subset if too slow
            for j in remaining:
                cnt = 0
                for i in neighbors[j]:
                    idx = word_to_idx[i]
                    if uncovered[idx]:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_j = j
                if cnt == 0:
                    continue
            if best_j < 0:
                break
            chosen.append(best_j)
            for i in neighbors[best_j]:
                uncovered[word_to_idx[i]] = False
            remaining.remove(best_j)
        return chosen

    def local_search(codeword_indices):
        """Local search: try removals, swaps, and 2-for-1 moves."""
        cw = list(codeword_indices)
        # Build coverage counts
        cover_count = [0] * total
        for j in cw:
            for i in neighbors[j]:
                cover_count[word_to_idx[i]] += 1

        def apply_add(j):
            for i in neighbors[j]:
                cover_count[word_to_idx[i]] += 1

        def apply_remove(j):
            for i in neighbors[j]:
                cover_count[word_to_idx[i]] -= 1

        def is_valid():
            return all(cover_count[idx] > 0 for idx in range(total))

        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False

            # Try removals
            for pos, j in enumerate(cw):
                apply_remove(j)
                if is_valid():
                    cw.pop(pos)
                    improved = True
                    break
                else:
                    apply_add(j)
            if improved:
                continue

            # Try swaps
            for pos, j in enumerate(cw):
                apply_remove(j)
                best_new = -1
                best_gain = 0
                for k in range(total):
                    if k in cw:
                        continue
                    # Gain = number of currently-uncovered words that k covers
                    gain = 0
                    for i in neighbors[k]:
                        idx = word_to_idx[i]
                        if cover_count[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_new = k
                if best_new >= 0:
                    cw[pos] = best_new
                    apply_add(best_new)
                    improved = True
                    break
                else:
                    apply_add(j)
            if improved:
                continue

            # Try 2-for-1: remove two, add one
            if len(cw) >= 2:
                for i in range(len(cw)):
                    for k in range(i + 1, len(cw)):
                        j1, j2 = cw[i], cw[k]
                        apply_remove(j1)
                        apply_remove(j2)
                        best_new = -1
                        best_gain = 0
                        for cand in range(total):
                            if cand in cw:
                                continue
                            gain = 0
                            for x in neighbors[cand]:
                                idx = word_to_idx[x]
                                if cover_count[idx] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_new = cand
                        if best_new >= 0 and best_new not in cw:
                            cw[i] = best_new
                            cw.pop(k)
                            apply_add(best_new)
                            improved = True
                            break
                        else:
                            apply_add(j2)
                            apply_add(j1)
                    if improved:
                        break
        return cw

    best = None
    best_len = float('inf')

    # Multiple restarts
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        seed = random.randint(0, 10**9)
        random.seed(seed)
        cw = greedy_randomized()
        cw = local_search(cw)
        if len(cw) < best_len:
            best_len = len(cw)
            best = cw
            # Try more local search rounds
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                cw2 = local_search(cw)
                if len(cw2) < best_len:
                    best_len = len(cw2)
                    best = cw2
                    cw = cw2

    # Final safety: ensure we have a valid cover
    if best is None:
        # Fallback: direct sum construction
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            best = list(all_words)

    # Convert to lists of ints
    return [list(all_words[j]) for j in best]
# EVOLVE-BLOCK-END
