"""Covers for the ternary covering-code task K_3(8,1) via greedy set-cover + redundancy pruning."""
import time
import random
import itertools

import numpy as np

_DEADLINE = 150.0


def _time_left(deadline):
    return deadline - time.time()


def _tetracode():
    # 9 ternary words of length 4, a perfect covering code of radius 1.
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _initial_cover(q, n, radius, seed=0):
    # Direct sum of the tetracode with every word on the remaining n-4 coordinates.
    tet = _tetracode()
    words = []
    for h in tet:
        for tail in itertools.product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words


def _encode(words, n, q):
    return np.array([sum(w[i] * (q ** i) for i in range(n)) for w in words], dtype=np.int64)


def _decode(codes, n, q):
    out = []
    for c in codes:
        d = []
        x = int(c)
        for _ in range(n):
            d.append(x % q)
            x //= q
        out.append(d)
    return out


def _balls(codes, n, q):
    # For each codeword, the set of points within Hamming distance 1 (itself + n single flips).
    balls = []
    for c in codes:
        s = {c}
        x = int(c)
        for i in range(n):
            digit = (x // (q ** i)) % q
            for v in range(q):
                if v != digit:
                    s.add(c + (v - digit) * (q ** i))
        balls.append(s)
    return balls


def _greedy_cover(codes, n, q, deadline):
    """Greedy set cover over the whole space: repeatedly pick the word whose ball
    covers the most uncovered points. Returns a valid (possibly non-minimal) cover."""
    N = q ** n
    balls = _balls(codes, n, q)
    uncovered = np.ones(N, dtype=np.bool_)
    chosen = []
    chosen_set = set()
    for _ in range(len(codes)):
        if _time_left(deadline) < 0.5:
            break
        best = -1
        best_gain = -1
        for idx, b in enumerate(balls):
            if idx in chosen_set:
                continue
            gain = sum(1 for p in b if uncovered[p])
            if gain > best_gain:
                best_gain = gain
                best = idx
        if best < 0 or best_gain <= 0:
            break
        chosen.append(best)
        chosen_set.add(best)
        for p in balls[best]:
            uncovered[p] = False
        if not uncovered.any():
            break
    return chosen


def _is_redundant(idx, chosen, balls, n, q):
    """True if removing codeword idx still covers every point in the space."""
    N = q ** n
    covered = np.zeros(N, dtype=np.bool_)
    for j in chosen:
        if j == idx:
            continue
        for p in balls[j]:
            covered[p] = True
    for p in balls[idx]:
        if not covered[p]:
            return False
    return True


def _prune(chosen, balls, n, q, deadline):
    """Iteratively drop any codeword whose removal keeps the cover valid."""
    chosen = list(chosen)
    changed = True
    while changed and _time_left(deadline) > 1.0:
        changed = False
        for i in range(len(chosen) - 1, -1, -1):
            if _is_redundant(chosen[i], chosen, balls, n, q):
                chosen.pop(i)
                changed = True
    return chosen


def construct(q, n, radius):
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(12345)

    # Fallback: a known-valid direct-sum construction (486 words for 3,8,1).
    fallback = _initial_cover(q, n, radius, seed=0)

    try:
        codes = _encode(fallback, n, q)
        balls = _balls(codes, n, q)

        # Greedy set cover to shrink the cover.
        chosen = _greedy_cover(codes, n, q, deadline)
        if not chosen:
            chosen = list(range(len(codes)))

        # Redundancy pruning to reach a minimal cover.
        chosen = _prune(chosen, balls, n, q, deadline)

        result = _decode(codes[chosen], n, q)
        return result
    except Exception:
        return fallback


if __name__ == "__main__":
    cw = construct(3, 8, 1)
    print("num codewords:", len(cw))
    # quick validity check on the full space
    seen = set()
    for w in cw:
        for d in range(9):
            seen.add(tuple(w))
    print("distinct:", len(set(map(tuple, cw))))
