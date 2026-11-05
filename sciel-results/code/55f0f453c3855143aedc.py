"""New design: precompute triple-isosceles conflicts, randomized greedy with deadline guard."""
import time
import random
import itertools

N = 16
DEADLINE = 300.0  # safety margin under 330s

# Precompute all points
POINTS = [(x, y) for x in range(N) for y in range(N)]
IDX = {p: i for i, p in enumerate(POINTS)}
M = len(POINTS)  # 256

def _dist_sq(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

# Precompute isosceles for all triples? 256 choose 3 ~ 2.7M, each check is 3 dists.
# Better: for each pair (a,b), precompute the set of points p such that (p,a,b) is isosceles.
# That's 256*255/2 ~ 32k pairs, each with up to 256 points. Store as a list of bitsets or sets.

# For each pair (i,j), store a bitmask (or set) of points k such that {i,j,k} is isosceles.
# Using Python int as bitmask for 256 bits is feasible.

# iso_mask[i][j] = bitmask of k where triangle (i,j,k) is isosceles (and k != i, k != j)
# Actually we need: when adding point p to a set S, for every pair (a,b) in S, p must not be in iso_mask[a][b].

# Precompute iso_mask as a 2D list of ints (bitmasks).
iso_mask = [[0] * M for _ in range(M)]
for i in range(M):
    for j in range(i + 1, M):
        a = POINTS[i]
        b = POINTS[j]
        d_ab = _dist_sq(a, b)
        mask = 0
        for k in range(M):
            if k == i or k == j:
                continue
            c = POINTS[k]
            d_ac = _dist_sq(a, c)
            d_bc = _dist_sq(b, c)
            if d_ac == d_bc or d_ac == d_ab or d_bc == d_ab:
                mask |= (1 << k)
        iso_mask[i][j] = mask
        iso_mask[j][i] = mask

def construct():
    start_time = time.time()
    best_set = None
    best_len = M + 1

    # Precompute for each point, for fast conflict checking
    # We'll do randomized greedy: shuffle the point order, greedily add if no conflict.

    # Also try a few deterministic orderings: row-major, col-major, diagonal, etc.
    orderings = []

    # Row-major (original)
    orderings.append(list(range(M)))
    # Column-major
    orderings.append([IDX[(y, x)] for x in range(N) for y in range(N)])
    # Diagonal
    orderings.append([IDX[(x, y)] for d in range(2 * N - 1) for x in range(N) for y in range(N) if x + y == d])

    rng = random.Random(12345)

    for ordering in orderings:
        if time.time() - start_time > DEADLINE - 30:
            break
        chosen = []
        chosen_set = set()
        for idx in ordering:
            p = POINTS[idx]
            ok = True
            # Check against all pairs in chosen
            for ii in range(len(chosen)):
                a = chosen[ii]
                for jj in range(ii + 1, len(chosen)):
                    b = chosen[jj]
                    if (iso_mask[idx][IDX[a]] >> IDX[b]) & 1:
                        ok = False
                        break
                    if (iso_mask[idx][IDX[b]] >> IDX[a]) & 1:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(p)
        if len(chosen) < best_len:
            best_len = len(chosen)
            best_set = list(chosen)
        if time.time() - start_time > DEADLINE - 30:
            break

    # Randomized restarts
    while time.time() - start_time < DEADLINE - 10:
        order = list(range(M))
        rng.shuffle(order)
        chosen = []
        for idx in order:
            p = POINTS[idx]
            ok = True
            for ii in range(len(chosen)):
                a = chosen[ii]
                for jj in range(ii + 1, len(chosen)):
                    b = chosen[jj]
                    if (iso_mask[idx][IDX[a]] >> IDX[b]) & 1:
                        ok = False
                        break
                    if (iso_mask[idx][IDX[b]] >> IDX[a]) & 1:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(p)
        if len(chosen) < best_len:
            best_len = len(chosen)
            best_set = list(chosen)

    return best_set
