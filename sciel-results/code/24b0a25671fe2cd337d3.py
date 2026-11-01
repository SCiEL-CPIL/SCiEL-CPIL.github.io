"""Structural rewrite: bit-parallel greedy set-cover + 1-exchange local search."""
import time
import random


def construct(q, n, radius):
    start = time.monotonic()
    deadline = start + 115.0  # safety margin under 120s

    if radius < 0:
        return []
    if n == 0:
        return [[]]

    N = q ** n

    def encode(w):
        v = 0
        for x in w:
            v = v * q + x
        return v

    def decode(v):
        w = []
        for _ in range(n):
            v, r = divmod(v, q)
            w.append(r)
        return w[::-1]

    # Precompute ball masks for every word (Hamming ball of radius `radius`).
    ball = [0] * N
    for v in range(N):
        w = decode(v)
        m = 1 << v
        if radius >= 1:
            for i in range(n):
                old = w[i]
                for x in range(q):
                    if x == old:
                        continue
                    w[i] = x
                    m |= 1 << encode(w)
                w[i] = old
        if radius >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    oi, oj = w[i], w[j]
                    for xi in range(q):
                        if xi == oi:
                            continue
                        w[i] = xi
                        for xj in range(q):
                            if xj == oj:
                                continue
                            w[j] = xj
                            m |= 1 << encode(w)
                        w[j] = oj
                    w[i] = oi
        ball[v] = m

    full = (1 << N) - 1
    best = None
    best_size = N + 1
    rng = random.Random(1234567)

    def greedy(uncovered, chosen, chosen_set, deadline):
        # local greedy set cover
        rem = uncovered
        local = list(chosen)
        local_set = set(chosen_set)
        while rem and time.monotonic() < deadline:
            best_v = -1
            best_gain = 0
            best_mask = 0
            for v in range(N):
                if v in local_set:
                    continue
                gain = (ball[v] & rem).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
                    best_mask = ball[v] & rem
                    if best_gain == rem.bit_count():
                        break
            if best_v < 0 or best_gain == 0:
                break
            local.append(best_v)
            local_set.add(best_v)
            rem &= ~best_mask
        return local, local_set, rem

    def try_one(deadline):
        uncovered = full
        chosen = []
        chosen_set = set()
        # randomized greedy: randomize tie-breaking via random permutation each pass
        while uncovered and time.monotonic() < deadline:
            # build candidate gains
            cands = []
            for v in range(N):
                if v in chosen_set:
                    continue
                gain = (ball[v] & uncovered).bit_count()
                if gain > 0:
                    cands.append((gain, v))
            if not cands:
                break
            # pick best with random tie-break
            maxg = max(c[0] for c in cands)
            ties = [c[1] for c in cands if c[0] == maxg]
            v = rng.choice(ties)
            chosen.append(v)
            chosen_set.add(v)
            uncovered &= ~ball[v]
            if not uncovered:
                break

        if not uncovered:
            if len(chosen) < best_size:
                best_size = len(chosen)
                best = list(chosen)

        # local improvement: 1-exchange
        if uncovered == 0 and time.monotonic() < deadline:
            chosen_set = set(chosen)
            improved = True
            while improved and time.monotonic() < deadline:
                improved = False
                # current coverage union
                union = 0
                for v in chosen:
                    union |= ball[v]
                # for each chosen word, compute what it uniquely/over covers
                # try replacing each chosen with an unchosen
                for idx, cv in enumerate(chosen):
                    without = 0
                    for k, u in enumerate(chosen):
                        if k != idx:
                            without |= ball[u]
                    # words only covered by cv
                    only_cv = ball[cv] & ~without
                    if only_cv == 0:
                        # cv is redundant; drop it
                        chosen.pop(idx)
                        chosen_set.discard(cv)
                        improved = True
                        break
                    # find unchosen v that covers all of only_cv and more
                    best_v = -1
                    best_gain = 0
                    for v in range(N):
                        if v in chosen_set:
                            continue
                        if (only_cv & ~ball[v]):
                            continue
                        gain = (ball[v] & ~without).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                    if best_v >= 0:
                        chosen[idx] = best_v
                        chosen_set.discard(cv)
                        chosen_set.add(best_v)
                        improved = True
                        break
        return chosen

    # First deterministic run to establish a baseline
    try_one(deadline)

    # Randomized restarts within budget
    while time.monotonic() < deadline:
        try_one(deadline)
        if best_size <= 186:
            break

    if best is None:
        # fallback: return all words (always valid)
        best = list(range(N))

    return [decode(v) for v in best]
