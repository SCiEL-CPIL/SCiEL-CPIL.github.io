"""Fresh design: greedy baseline + simulated annealing (recolor + swap moves)."""
import time
import random
import numpy as np

N = 16
M = N * (N - 1) // 2  # 120
MAXC = 120

# Precompute edge endpoints and the three 4-cycles per 4-vertex subset.
_EI = []
_EJ = []
for i in range(N):
    for j in range(i + 1, N):
        _EI.append(i)
        _EJ.append(j)
_EI = np.array(_EI, dtype=np.int32)
_EJ = np.array(_EJ, dtype=np.int32)

# For each 4-subset {a<b<c<d}, the 6 edges (a,b),(a,c),(a,d),(b,c),(b,d),(c,d).
_C4 = []
for a in range(N - 3):
    for b in range(a + 1, N - 2):
        for c in range(b + 1, N - 1):
            for d in range(c + 1, N):
                _C4.append((a, b, a, c, a, d, b, c, b, d, c, d))
_C4 = np.array(_C4, dtype=np.int32)  # shape (K, 12)

_K = _C4.shape[0]
# edge index for each of the 12 slots
_SLOT_IDX = []
for u, v in _C4:
    pass
# Build slot -> edge index mapping
def _edge_index(u, v):
    if u > v:
        u, v = v, u
    return u * (2 * N - u - 1) // 2 - 1 + (v - u - 1)
# Precompute: for each of the 12 slots, the flat edge index
_SLOT = np.empty((_K, 12), dtype=np.int32)
for k in range(_K):
    for s in range(12):
        u = int(_C4[k, 2 * s])
        v = int(_C4[k, 2 * s + 1])
        _SLOT[k, s] = _edge_index(u, v)


def _is_valid(colors):
    """Return True iff no 4-cycle has 4 distinct colors."""
    # colors: length-M array
    vals = colors[_SLOT]  # (K,12)
    # number of distinct per row
    # Use: distinct < 4  <=>  not (4 distinct). Compute via set sizes using argsort.
    # Efficient: for each row, check if there exist 4 distinct values.
    # distinct count = 1 + number of positions where value differs from previous after sort.
    s = np.sort(vals, axis=1)  # (K,12)
    diff = (s[:, 1:] != s[:, :-1]).sum(axis=1) + 1  # distinct count per row
    return bool(np.all(diff < 4))


def _distinct(colors):
    return int(len(np.unique(colors)))


def _greedy_build(rng):
    """Greedy: assign edges in random order, pick a color that keeps validity."""
    colors = np.empty(M, dtype=np.int32)
    order = np.arange(M)
    rng.shuffle(order)
    # We maintain validity incrementally by full check only when needed; but greedy
    # with full check each step is O(M * cost). M=120, cost of _is_valid ~ K*12 sort.
    # K = C(16,4)=1820, sort of 12 -> cheap. 120 steps * 1820 -> fine.
    for e in order:
        # try colors: prefer a fresh color first, then existing, then 0
        placed = False
        # candidate colors: a new color (MAXC-1 down) then existing distinct then 0
        used = np.unique(colors[:e] if e else np.array([]))
        # Build candidate list
        cands = []
        # new colors (high to low) - try a few
        for c in range(MAXC - 1, MAXC - 40, -1):
            cands.append(c)
        for c in used:
            cands.append(int(c))
        cands.append(0)
        # de-dup preserving order
        seen = set()
        uniq = []
        for c in cands:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        for c in uniq:
            colors[e] = c
            if _is_valid(colors):
                placed = True
                break
        if not placed:
            colors[e] = 0
            # force valid: this shouldn't happen since 0-always-valid baseline exists
            # but if it does, we fallback below
    if not _is_valid(colors):
        return np.zeros(M, dtype=np.int32)
    return colors


def _random_valid_init(rng):
    # Start from all-zero (valid) then try to diversify via random recolors that keep validity.
    colors = np.zeros(M, dtype=np.int32)
    for _ in range(4000):
        e = rng.randrange(M)
        c = rng.randrange(MAXC)
        if c == colors[e]:
            continue
        old = colors[e]
        colors[e] = c
        if _is_valid(colors):
            pass
        else:
            colors[e] = old
    return colors


def _anneal(colors, rng, deadline):
    colors = colors.copy()
    best = colors.copy()
    best_score = _distinct(best)
    T = 30.0
    T_min = 0.05
    cool = 0.99995
    # moves: 0 = recolor, 1 = swap
    while time.time() < deadline:
        move = rng.random()
        if move < 0.6:
            # recolor
            e = rng.randrange(M)
            c = rng.randrange(MAXC)
            if c == colors[e]:
                continue
            old = colors[e]
            colors[e] = c
            if _is_valid(colors):
                s = _distinct(colors)
                if s >= best_score:
                    # accept (we always accept improvements / equals)
                    if s > best_score:
                        best = colors.copy()
                        best_score = s
                else:
                    # worse: accept with probability
                    if rng.random() < np.exp((s - best_score) / T):
                        pass
                    else:
                        colors[e] = old
            else:
                colors[e] = old
        else:
            # swap two edges
            e1, e2 = rng.sample(range(M), 2)
            c1, c2 = colors[e1], colors[e2]
            if c1 == c2:
                continue
            colors[e1], colors[e2] = c2, c1
            if _is_valid(colors):
                s = _distinct(colors)
                if s >= best_score:
                    if s > best_score:
                        best = colors.copy()
                        best_score = s
                else:
                    if rng.random() < np.exp((s - best_score) / T):
                        pass
                    else:
                        colors[e1], colors[e2] = c1, c2
            else:
                colors[e1], colors[e2] = c1, c2
        T = max(T_min, T * cool)
    return best, best_score


def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill
    rng = random.Random(12345)
    # Baseline: all zero is valid.
    best = np.zeros(M, dtype=np.int32)
    best_score = 1
    # Greedy to get a richer starting point.
    g = _greedy_build(rng)
    gs = _distinct(g)
    if gs > best_score:
        best = g.copy()
        best_score = gs
    # Random-valid init may be better.
    r = _random_valid_init(rng)
    rs = _distinct(r)
    if rs > best_score:
        best = r.copy()
        best_score = rs
    # Anneal from the best starting point, with restarts.
    while time.time() < deadline - 5.0:
        start = best.copy()
        # occasionally restart from a fresh random-valid init
        if rng.random() < 0.2:
            start = _random_valid_init(rng)
        cand, cs = _anneal(start, rng, deadline - 2.0)
        if cs > best_score:
            best = cand
            best_score = cs
        if time.time() >= deadline - 5.0:
            break
    # Final safety: ensure validity.
    if not _is_valid(best):
        best = np.zeros(M, dtype=np.int32)
    return [int(x) for x in best]
