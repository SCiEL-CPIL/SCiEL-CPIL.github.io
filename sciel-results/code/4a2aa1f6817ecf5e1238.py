"""Ternary covering code K_3(7,1): minimize number of radius-1 covering words."""
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin under the 120s/150s limit


def _time_ok():
    return time.time() < _DEADLINE


def _neighbors_of(idx, q, n):
    """All words within Hamming distance 1 of the word with linear index idx."""
    word = np.unravel_index(idx, (q,) * n)
    out = [idx]
    for pos in range(n):
        for v in range(q):
            if v == word[pos]:
                continue
            w = list(word)
            w[pos] = v
            out.append(int(np.ravel_multi_index(tuple(w), (q,) * n)))
    return out


def _greedy_cover(q, n, r, neighbor_sets, covered, deadline):
    """Greedy set cover: repeatedly add the word covering the most uncovered words."""
    remaining = [i for i in range(len(covered)) if not covered[i]]
    chosen = []
    while remaining:
        if not _time_ok():
            break
        best_i, best_gain = -1, 0
        for i in remaining:
            gain = 0
            for j in neighbor_sets[i]:
                if not covered[j]:
                    gain += 1
                    if gain > best_gain:
                        pass
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i < 0:
            break
        chosen.append(best_i)
        covered[best_i] = True
        for j in neighbor_sets[best_i]:
            covered[j] = True
        remaining = [i for i in remaining if not covered[i]]
        if not remaining:
            break
    return chosen


def _remove_redundant(q, n, r, neighbor_sets, chosen):
    """Remove any chosen word whose covered words are all covered by the others."""
    changed = True
    while changed:
        changed = False
        if not _time_ok():
            break
        m = len(chosen)
        for i in range(m):
            if not _time_ok():
                break
            wi = neighbor_sets[chosen[i]]
            ok = True
            for w in wi:
                # is w covered by some other chosen word?
                found = False
                for j in range(m):
                    if j == i:
                        continue
                    if w in neighbor_sets[chosen[j]]:
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if ok:
                chosen.pop(i)
                changed = True
                break
    return chosen


def _try_swap(q, n, r, neighbor_sets, chosen):
    """1-for-1 swap: replace chosen[i] with c not in chosen, if c covers all words
    that only i covers, and c adds at least one newly-covered word."""
    m = len(chosen)
    in_set = [False] * (q ** n)
    for c in chosen:
        in_set[c] = True
    for i in range(m):
        if not _time_ok():
            return chosen
        wi = neighbor_sets[chosen[i]]
        # words uniquely covered by i
        unique = []
        for w in wi:
            only = True
            for j in range(m):
                if j == i:
                    continue
                if w in neighbor_sets[chosen[j]]:
                    only = False
                    break
            if only:
                unique.append(w)
        if len(unique) > 60:  # unlikely to be swappable
            continue
        # candidate pool: words that cover all unique words
        for c in range(len(neighbor_sets)):
            if in_set[c]:
                continue
            nc = neighbor_sets[c]
            covers_all = all(w in nc for w in unique)
            if not covers_all:
                continue
            # gain: words covered by c not covered by anyone else (besides i)
            gain = 0
            for w in nc:
                covered_by_other = False
                for j in range(m):
                    if j == i:
                        continue
                    if w in neighbor_sets[chosen[j]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    gain += 1
            if gain > 0:
                chosen[i] = c
                in_set[c] = True
                return chosen
    return chosen


def construct(q, n, radius):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 110.0

    if radius != 1:
        # fallback: full space
        import itertools
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    total = q ** n
    # Precompute neighbor sets (each word + its radius-1 ball)
    neighbor_sets = []
    for idx in range(total):
        neighbor_sets.append(_neighbors_of(idx, q, n))

    # Greedy cover
    covered = np.zeros(total, dtype=bool)
    chosen = _greedy_cover(q, n, radius, neighbor_sets, covered, _DEADLINE)

    # Removal polish (loop while time allows)
    while _time_ok():
        before = len(chosen)
        chosen = _remove_redundant(q, n, radius, neighbor_sets, chosen)
        if len(chosen) == before:
            break

    # Swap polish
    improved = True
    while improved and _time_ok():
        improved = False
        before = len(chosen)
        chosen = _try_swap(q, n, radius, neighbor_sets, chosen)
        if len(chosen) < before:
            improved = True
        # re-run removal after swaps
        chosen = _remove_redundant(q, n, radius, neighbor_sets, chosen)
        if len(chosen) < before:
            improved = True

    # Ensure validity; if invalid (shouldn't be), fall back to full space
    cov = np.zeros(total, dtype=bool)
    for c in chosen:
        for w in neighbor_sets[c]:
            cov[w] = True
    if not cov.all():
        import itertools
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Return as lists of ints
    out = []
    for c in chosen:
        out.append(list(np.unravel_index(c, (q,) * n)))
    return out
