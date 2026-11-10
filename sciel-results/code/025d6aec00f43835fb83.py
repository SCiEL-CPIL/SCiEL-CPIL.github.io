"""Algebraic (parametric) 4-cycle-safe colorings: maximize forced equalities."""
import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120


def _edge_idx():
    idx = [[-1] * N for _ in range(N)]
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    return idx


_EDGE = _edge_idx()


def _all_4cycles():
    cycles = []
    for a, b, c, d in itertools.combinations(range(N), 4):
        e1 = (_EDGE[a][b], _EDGE[b][c], _EDGE[c][d], _EDGE[d][a])
        e2 = (_EDGE[a][b], _EDGE[b][d], _EDGE[d][c], _EDGE[c][a])
        e3 = (_EDGE[a][c], _EDGE[c][b], _EDGE[b][d], _EDGE[d][a])
        cycles.append(e1)
        cycles.append(e2)
        cycles.append(e3)
    return cycles


_CYCLES = _all_4cycles()
_NUM_CYCLES = len(_CYCLES)  # 336


def _colors_from(P, Q, R, M):
    return [((a * P + b * Q + R) % M) for a, b in
            itertools.combinations(range(N), 2)]


def _is_valid(colors):
    for e in _CYCLES:
        c0, c1, c2, c3 = colors[e[0]], colors[e[1]], colors[e[2]], colors[e[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True


def _count_coincidences(colors):
    """Total number of (unordered) equal-color pairs across all 4-cycles."""
    total = 0
    for e in _CYCLES:
        c0, c1, c2, c3 = colors[e[0]], colors[e[1]], colors[e[2]], colors[e[3]]
        if c0 == c1:
            total += 1
        if c0 == c2:
            total += 1
        if c0 == c3:
            total += 1
        if c1 == c2:
            total += 1
        if c1 == c3:
            total += 1
        if c2 == c3:
            total += 1
    return total


def _force_pair(P, Q, R, M, a1, b1, a2, b2):
    """Set (P,Q) so edge(a1,b1) == edge(a2,b2) (mod M). R unchanged.
    Solves  (a1-a2)P + (b1-b2)Q = 0 (mod M)."""
    da = (a1 - a2) % M
    db = (b1 - b2) % M
    if da == 0 and db == 0:
        return P, Q
    if da == 0:
        # db * Q = 0  ->  Q = 0
        Q = 0
        P = (P + random.randrange(M)) % M  # free parameter: randomize
    elif db == 0:
        P = 0
        Q = (Q + random.randrange(M)) % M
    else:
        g = pow(da, -1, M)  # M is prime
        Q = (-db * g) % M   # P = -db*da^-1 * Q ; choose Q=1
        P = (-db * g) % M
    return P, Q


def _improve(P, Q, R, M, deadline):
    colors = _colors_from(P, Q, R, M)
    cur = _count_coincidences(colors)
    stall = 0
    while time.time() < deadline:
        # find the cycle with the fewest equalities
        worst = None
        worst_cnt = 99
        for e in _CYCLES:
            c0, c1, c2, c3 = colors[e[0]], colors[e[1]], colors[e[2]], colors[e[3]]
            cnt = 0
            if c0 == c1:
                cnt += 1
            if c0 == c2:
                cnt += 1
            if c0 == c3:
                cnt += 1
            if c1 == c2:
                cnt += 1
            if c1 == c3:
                cnt += 1
            if c2 == c3:
                cnt += 1
            if cnt < worst_cnt:
                worst_cnt = cnt
                worst = e
                if cnt == 0:
                    break
        if worst is None:
            break
        e = worst
        c0, c1, c2, c3 = colors[e[0]], colors[e[1]], colors[e[2]], colors[e[3]]
        # candidate pairs to force equal (prefer a pair that already nearly matches)
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        random.shuffle(pairs)
        improved = False
        for (i, j) in pairs:
            vals = [c0, c1, c2, c3]
            if vals[i] == vals[j]:
                continue
            (a1, b1) = itertools.combinations(range(N), 2).__next__()  # placeholder, replaced below
            break
        # proper vertex extraction
        # cycle e is a tuple of 4 edge indices in order; we need the 4 vertices.
        # Reconstruct from known structure by scanning: find the 4-vertex set.
        # Simpler: recompute the vertex sets alongside cycles is cleaner; do that here.
        improved = False
        for (i, j) in pairs:
            (a1, b1), (a2, b2) = _cycle_vertices(e)[i], _cycle_vertices(e)[j]
            np_, nq = _force_pair(P, Q, R, M, a1, b1, a2, b2)
            if np_ == P and nq == Q:
                continue
            ncolors = _colors_from(np_, nq, R, M)
            nc = _count_coincidences(ncolors)
            if nc > cur:
                P, Q, colors, cur = np_, nq, ncolors, nc
                stall = 0
                improved = True
                break
            elif nc == cur and random.random() < 0.3:
                P, Q, colors, cur = np_, nq, ncolors, nc
                stall = 0
                improved = True
                break
        if not improved:
            stall += 1
            if stall > 150:
                break
    return P, Q, colors, cur


def _cycle_vertices(e):
    """Return the 4 vertex-pairs (edges) for a cycle in order, as a tuple of
    (u,v) pairs matching the order in e."""
    # e is (e_ab, e_bc, e_cd, e_da) etc. We stored them as edge indices; the
    # vertex order is not stored. Recover by brute force over the 4-vertex set.
    # Find the unique 4-set S whose three cycle structures contain e, then
    # return the ordered edges of the matching structure.
    for a, b, c, d in itertools.combinations(range(N), 4):
        s1 = (_EDGE[a][b], _EDGE[b][c], _EDGE[c][d], _EDGE[d][a])
        s2 = (_EDGE[a][b], _EDGE[b][d], _EDGE[d][c], _EDGE[c][a])
        s3 = (_EDGE[a][c], _EDGE[c][b], _EDGE[b][d], _EDGE[d][a])
        if e == s1:
            return [(a, b), (b, c), (c, d), (d, a)]
        if e == s2:
            return [(a, b), (b, d), (d, c), (c, a)]
        if e == s3:
            return [(a, c), (c, b), (b, d), (d, a)]
    return [(0, 1), (1, 2), (2, 3), (3, 0)]


def construct():
    start = time.time()
    deadline = start + 80.0  # 10s safety margin before the 90s kill

    # Deterministic seed for reproducibility within a run.
    rng = random.Random(20240517)
    random.seed(20240517)

    best_colors = [0] * NUM_EDGES
    best_coinc = 0

    # A small prime modulus; colors live in 0..M-1 (subset of 0..119).
    # M=13 gives 13 usable colors, comfortably within range and prime.
    M = 13

    # Try a handful of structured seeds, keep the best by coincidence count.
    attempts = [
        (1, 2, 0), (1, 3, 0), (2, 1, 0), (1, 1, 0), (3, 2, 0),
        (2, 3, 1), (1, 4, 0), (4, 1, 0), (2, 2, 1), (3, 1, 2),
    ]
    for (P, Q, R) in attempts:
        if time.time() > deadline:
            break
        P %= M
        Q %= M
        R %= M
        colors = _colors_from(P, Q, R, M)
        if not _is_valid(colors):
            continue
        P, Q, colors, coinc = _improve(P, Q, R, M, deadline)
        if coinc > best_coinc and _is_valid(colors):
            best_coinc = coinc
            best_colors = colors

    # Fallback: all zeros is always valid.
    if not _is_valid(best_colors):
        best_colors = [0] * NUM_EDGES

    return best_colors
