"""Football pool covering code: minimize codewords so every word is within Hamming distance radius."""
import time
import numpy as np

_DEADLINE = 112.0  # safety margin under 120s (evaluator kills at 150s)


def _balls(q, n, r):
    """Return list of balls (each a numpy int64 array of word indices) for every word in {0..q-1}^n."""
    words = np.array(list(_itertools_product(range(q), repeat=n)), dtype=np.int64).reshape(-1, n)
    N = q ** n
    balls = []
    for w in words:
        idx = _ball_indices(w, q, n, r)
        balls.append(idx)
    return words, balls


def _ball_indices(center, q, n, r):
    """All words within Hamming distance r of center (a length-n int array)."""
    from itertools import combinations
    out = [center.tolist()]
    for d in range(1, r + 1):
        for pos in combinations(range(n), d):
            for vals in _itertools_product(range(q), repeat=d):
                w = center.tolist()
                for p, v in zip(pos, vals):
                    if v != center[p]:
                        w[p] = v
                    else:
                        break
                else:
                    out.append(w)
    return np.array(out, dtype=np.int64)


def _itertools_product(iterable, repeat):
    import itertools
    return itertools.product(iterable, repeat=repeat)


def construct(q, n, radius):
    start = time.time()
    N = q ** n
    words, balls = _balls(q, n, radius)

    # covered[i] = True if word i is within radius of some chosen codeword
    covered = np.zeros(N, dtype=bool)
    chosen = []
    chosen_idx = set()

    # Greedy set cover: repeatedly pick the ball covering the most uncovered words.
    while not covered.all():
        if time.time() - start > _DEADLINE:
            break
        best = -1
        best_count = -1
        for i in range(N):
            if i in chosen_idx:
                continue
            c = np.count_nonzero(balls[i] & ~covered)
            if c > best_count:
                best_count = c
                best = i
                if best_count == 0:
                    break
        if best < 0:
            break
        covered[balls[best]] = True
        chosen.append(words[best].tolist())
        chosen_idx.add(best)

    # If we ran out of time before full coverage, verify and (if needed) brute-force patch.
    if not covered.all():
        # Patch remaining uncovered words by adding a codeword at each uncovered position.
        for i in np.flatnonzero(~covered):
            chosen.append(words[i].tolist())
            covered[balls[i]] = True

    # Deduplicate (shouldn't happen but safe).
    seen = set()
    out = []
    for c in chosen:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            out.append(c)
    return out
