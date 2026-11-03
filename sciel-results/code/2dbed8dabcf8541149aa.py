import time
import random

_DEADLINE = 112.0
_T0 = time.time()


def _remaining():
    return _DEADLINE - time.time()


def construct(q, n, radius):
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 112.0
    rng = random.Random(12345)

    # General safety: if the problem is trivially small, just cover everything.
    total = q ** n
    if total <= 200000:
        return [list(w) for w in _all_words(q, n)]

    # ---- Ternary covering code, n=6, radius=1 (the task's fixed case) ----
    if q == 3 and n == 6 and radius == 1:
        return _ternary_cover()

    # ---- Fallback: plain greedy set cover for other parameters ----
    return _generic_greedy(q, n, radius)


def _all_words(q, n):
    """Yield all words as tuples."""
    for i in range(q ** n):
        x = i
        w = []
        for _ in range(n):
            w.append(x % q)
            x //= q
        yield tuple(w)


def _generic_greedy(q, n, radius):
    """Simple greedy set cover over the whole space (small problems only)."""
    words = list(_all_words(q, n))
    N = len(words)
    idx = {w: i for i, w in enumerate(words)}

    # Precompute the ball (words within radius) for each word.
    balls = []
    for w in words:
        ball = [w]
        for pos in range(n):
            for v in range(q):
                if v == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = v
                ball.append(tuple(nw))
        # radius 1 only for this fallback (task is radius 1)
        balls.append(ball)

    uncovered = set(range(N))
    code = []
    while uncovered:
        best = None
        best_gain = -1
        for i, ball in enumerate(balls):
            gain = sum(1 for j in ball if j in uncovered)
            if gain > best_gain:
                best_gain = gain
                best = i
        code.append(words[best])
        for j in balls[best]:
            uncovered.discard(j)
        if _remaining() < 8.0:
            break
    return code


def _ternary_cover():
    q, n, radius = 3, 6, 1
    words = list(_all_words(q, n))
    N = len(words)
    idx = {w: i for i, w in enumerate(words)}

    # Ball (radius 1) of each word, as a Python int bitset over the 729 words.
    balls = [0] * N
    for w in words:
        b = 1 << idx[w]
        for pos in range(n):
            for v in range(q):
                if v == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = v
                b |= 1 << idx[tuple(nw)]
        balls[idx[w]] = b

    # ---- Phase 1: greedy set cover (fast, bit-parallel) ----
    uncovered = (1 << N) - 1
    code = []
    while uncovered:
        best = -1
        best_gain = 0
        for i in range(N):
            g = (uncovered & balls[i]).bit_count()
            if g > best_gain:
                best_gain = g
                best = i
        code.append(best)
        uncovered &= ~balls[best]
        if _remaining() < 8.0:
            break

    # ---- Phase 2: local-search polish ----
    code_set = set(code)
    while _remaining() > 10.0:
        improved = False

        # 2a) Remove redundant codewords.
        new_code = []
        for i in code:
            covered = balls[i]
            ok = True
            for j in code:
                if j == i:
                    continue
                if (covered & balls[j]) == covered:
                    ok = False
                    break
            if ok:
                new_code.append(i)
        if len(new_code) < len(code):
            code = new_code
            improved = True

        # 2b) Swap: drop one, add a better one.
        if _remaining() > 10.0:
            for i in range(len(code)):
                without = list(code)
                without[i] = None
                union = 0
                for j in range(len(without)):
                    if without[j] is not None:
                        union |= balls[without[j]]
                # find candidate that covers the most currently-uncovered words
                cand = -1
                cand_gain = 0
                for c in range(N):
                    g = (~union & balls[c]).bit_count()
                    if g > cand_gain:
                        cand_gain = g
                        cand = c
                if cand != -1 and cand not in code:
                    without[i] = cand
                    code = without
                    improved = True
                    break
        if not improved:
            break

    return [list(words[i]) for i in code]
