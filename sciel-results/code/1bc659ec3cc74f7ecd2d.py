"""Fresh design: direct-sum seed + fast counter-based reduction + local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

_DEADLINE = 110.0


def _elapsed(start):
    return time.time() - start


def _precompute(q, n):
    """Precompute all words and their neighbor lists (distance <=1) as index arrays."""
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_idx = {tuple(w): i for i, w in enumerate(all_words)}
    N = len(all_words)
    neigh = []
    for w in all_words:
        nbs = [word_idx[tuple(w)]]
        for i in range(n):
            for v in range(q):
                if v != w[i]:
                    nw = list(w)
                    nw[i] = v
                    nbs.append(word_idx[tuple(nw)])
        neigh.append(nbs)
    return all_words, neigh, N


def _is_covered(all_words, neigh, N, codeword_idx_set):
    """Return True if every word is within distance 1 of some codeword index."""
    # Count coverage: a word is covered if any of its neighbors (incl itself) is a codeword
    # Use a coverage count array
    cov = [0] * N
    for ci in codeword_idx_set:
        for nb in neigh[ci]:
            cov[nb] += 1
    for i in range(N):
        if cov[i] == 0:
            return False
    return True


def _direct_sum_tetracode(q, n):
    if q != 3:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n == 4:
        return list(tetracode)
    if n > 4:
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    return None


def _greedy_cover(q, n, radius, rng, deadline_start, all_words, neigh, N):
    if radius != 1:
        return None
    uncovered = [True] * N
    remaining = [True] * N
    cover_count = [0] * N
    for i in range(N):
        for j in neigh[i]:
            if uncovered[j]:
                cover_count[i] += 1
    chosen = []
    while any(uncovered):
        if _elapsed(deadline_start) > _DEADLINE - 5:
            return None
        best = -1
        best_count = -1
        for i in range(N):
            if remaining[i] and cover_count[i] > best_count:
                best_count = cover_count[i]
                best = i
        if best == -1 or best_count == 0:
            break
        chosen.append(best)
        for j in neigh[best]:
            if uncovered[j]:
                uncovered[j] = False
                for k in range(N):
                    if remaining[k] and j in neigh[k]:
                        cover_count[k] -= 1
        remaining[best] = False
    return chosen


def _try_reduce(q, n, radius, rng, deadline_start, all_words, neigh, N, codeword_idx):
    if radius != 1:
        return codeword_idx
    codeword_idx = list(codeword_idx)
    changed = True
    while changed:
        if _elapsed(deadline_start) > _DEADLINE - 5:
            break
        changed = False
        for idx in range(len(codeword_idx)):
            cand = [c for i, c in enumerate(codeword_idx) if i != idx]
            if _is_covered(all_words, neigh, N, cand):
                codeword_idx = cand
                changed = True
                break
    return codeword_idx


def _local_search(q, n, radius, rng, deadline_start, all_words, neigh, N, codeword_idx):
    if radius != 1:
        return codeword_idx
    best = list(codeword_idx)
    while _elapsed(deadline_start) < _DEADLINE - 10:
        cur = list(best)
        # Try removing one random codeword
        idx = rng.randrange(len(cur))
        cand = [c for i, c in enumerate(cur) if i != idx]
        if _is_covered(all_words, neigh, N, cand):
            best = cand
            continue
        # Try replacing one codeword with a random word index
        idx = rng.randrange(len(cur))
        new_idx = rng.randrange(N)
        if new_idx in cur:
            continue
        cand = cur[:idx] + [new_idx] + cur[idx+1:]
        if len(set(cand)) < len(cand):
            continue
        if _is_covered(all_words, neigh, N, cand):
            best = cand
    return best


def construct(q, n, radius):
    start = time.time()
    rng = random.Random(12345)
    all_words, neigh, N = _precompute(q, n)
    word_idx = {tuple(w): i for i, w in enumerate(all_words)}
    initial = _direct_sum_tetracode(q, n)
    if initial is not None:
        initial_idx = [word_idx[tuple(w)] for w in initial]
    else:
        initial_idx = _greedy_cover(q, n, radius, rng, start, all_words, neigh, N)
        if initial_idx is None:
            return []
    reduced = _try_reduce(q, n, radius, rng, start, all_words, neigh, N, initial_idx)
    improved = _local_search(q, n, radius, rng, start, all_words, neigh, N, reduced)
    return [all_words[i] for i in improved]
# EVOLVE-BLOCK-END
