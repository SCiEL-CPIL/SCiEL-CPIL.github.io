"""Fresh design: randomized greedy + local improvement for max-color no-rainbow-C4."""
import time
import random

N = 16
M = N * (N - 1) // 2  # 120 edges
MAXC = 120

# Precompute edge index and the three C4s per 4-subset (as edge indices)
EDGE = {}
EDGE_LIST = []
for i in range(N):
    for j in range(i + 1, N):
        EDGE[(i, j)] = len(EDGE_LIST)
        EDGE_LIST.append((i, j))

# For each 4-subset, the three ways to form a 4-cycle (edge index triples)
C4S = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # cycle 1: a-b-c-d-a
                C4S.append((EDGE[(a, b)], EDGE[(b, c)], EDGE[(c, d)], EDGE[(d, a)]))
                # cycle 2: a-b-d-c-a
                C4S.append((EDGE[(a, b)], EDGE[(b, d)], EDGE[(d, c)], EDGE[(c, a)]))
                # cycle 3: a-c-b-d-a
                C4S.append((EDGE[(a, c)], EDGE[(c, b)], EDGE[(b, d)], EDGE[(d, a)]))

# For each edge, which C4s contain it
EDGE_C4S = [[] for _ in range(M)]
for idx, c4 in enumerate(C4S):
    for e in c4:
        EDGE_C4S[e].append(idx)


def count_colors(col):
    return len(set(col))


def is_valid(col):
    for e1, e2, e3, e4 in C4S:
        if len((col[e1], col[e2], col[e3], col[e4])) == 4:
            return False
    return True


def greedy(order, rng, prefer_unused=True):
    col = [-1] * M
    used = [False] * MAXC
    # For fast checking, track per-C4 the colors seen
    for e in order:
        best = -1
        # candidate colors: try unused first
        if prefer_unused:
            for c in range(MAXC):
                if used[c]:
                    continue
                # check validity
                ok = True
                for ci in EDGE_C4S[e]:
                    e1, e2, e3, e4 = C4S[ci]
                    other = [col[x] for x in (e1, e2, e3, e4) if x != e]
                    if all(o != -1 for o in other):
                        if len(set(other + [c])) == 4:
                            ok = False
                            break
                if ok:
                    best = c
                    break
            if best == -1:
                for c in range(MAXC):
                    ok = True
                    for ci in EDGE_C4S[e]:
                        e1, e2, e3, e4 = C4S[ci]
                        other = [col[x] for x in (e1, e2, e3, e4) if x != e]
                        if all(o != -1 for o in other):
                            if len(set(other + [c])) == 4:
                                ok = False
                                break
                    if ok:
                        best = c
                        break
            if best == -1:
                best = 0
        else:
            best = 0
        col[e] = best
        used[best] = True
    return col


def local_improve(col, deadline):
    col = col[:]
    best_count = count_colors(col)
    order = list(range(M))
    while time.time() < deadline:
        progress = False
        rng_order = order[:]
        random.shuffle(rng_order)
        for e in rng_order:
            cur = col[e]
            # try to change e to a color that increases distinct count and stays valid
            for c in range(MAXC):
                if c == cur:
                    continue
                col[e] = c
                ok = True
                for ci in EDGE_C4S[e]:
                    e1, e2, e3, e4 = C4S[ci]
                    vals = [col[x] for x in (e1, e2, e3, e4)]
                    if len(set(vals)) == 4:
                        ok = False
                        break
                if ok:
                    cnt = count_colors(col)
                    if cnt > best_count:
                        best_count = cnt
                        progress = True
                    else:
                        col[e] = cur
                else:
                    col[e] = cur
            if time.time() >= deadline:
                break
        if not progress:
            # try one more pass; if still nothing, stop
            break
    return col, best_count


def construct():
    deadline = time.time() + 75.0  # safety margin under 90s
    best = None
    best_count = 0

    # A few greedy restarts with different orders
    for trial in range(200):
        if time.time() > deadline:
            break
        rng = random.Random(trial)
        order = list(range(M))
        rng.shuffle(order)
        col = greedy(order, rng, prefer_unused=True)
        if not is_valid(col):
            continue
        cnt = count_colors(col)
        if cnt > best_count:
            best_count = cnt
            best = col[:]
        # quick local improve on promising candidates
        if cnt >= best_count and time.time() < deadline - 3.0:
            col2, cnt2 = local_improve(col, deadline - 3.0)
            if cnt2 > best_count:
                best_count = cnt2
                best = col2[:]

    if best is None:
        # fallback: color = j (15 colors), always valid
        best = [0] * M
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                best[idx] = j
                idx += 1
        return best

    return best


# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
