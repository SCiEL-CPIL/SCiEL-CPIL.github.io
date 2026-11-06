# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

_DEADLINE = time.time() + 140.0  # safety margin before 150s wall-clock kill


def _build(n):
    """Precompute triple index map and triple -> quads / pents membership."""
    idx_of = {}
    i = 0
    for a in range(n - 2):
        for b in range(a + 1, n - 1):
            for c in range(b + 1, n):
                idx_of[(a, b, c)] = i
                i += 1
    m = i

    # For each quad, its 4 triples (indices)
    quad_tris = []
    for a, b, c, d in combinations(range(n), 4):
        quad_tris.append((idx_of[(a, b, c)], idx_of[(a, b, d)],
                          idx_of[(a, c, d)], idx_of[(b, c, d)]))

    # For each pent, its 10 triples (indices)
    pent_tris = []
    for a, b, c, d, e in combinations(range(n), 5):
        pent_tris.append((idx_of[(a, b, c)], idx_of[(a, b, d)], idx_of[(a, b, e)],
                          idx_of[(a, c, d)], idx_of[(a, c, e)], idx_of[(a, d, e)],
                          idx_of[(b, c, d)], idx_of[(b, c, e)], idx_of[(b, d, e)],
                          idx_of[(c, d, e)]))

    # For each triple, the quads and pents it belongs to
    nq = len(quad_tris)
    np_ = len(pent_tris)
    tri_quads = [[] for _ in range(m)]
    tri_pents = [[] for _ in range(m)]
    for qi, qt in enumerate(quad_tris):
        for t in qt:
            tri_quads[t].append(qi)
    for pi, pt in enumerate(pent_tris):
        for t in pt:
            tri_pents[t].append(pi)

    return idx_of, m, quad_tris, pent_tris, tri_quads, tri_pents


def _init_counts(colors, quad_tris, pent_tris):
    quad_red = np.zeros(len(quad_tris), dtype=np.int8)
    for qt in quad_tris:
        quad_red += np.fromiter((colors[t] == 0 for t in qt), dtype=np.int8, count=4)
    pent_blue = np.zeros(len(pent_tris), dtype=np.int8)
    for pt in pent_tris:
        pent_blue += np.fromiter((colors[t] == 1 for t in pt), dtype=np.int8, count=10)
    red_v = int((quad_red == 4).sum())
    blue_v = int((pent_blue == 10).sum())
    return quad_red, pent_blue, red_v, blue_v


def _local_search(colors, n, idx_of, quad_tris, pent_tris, tri_quads, tri_pents, deadline):
    colors = np.array(colors, dtype=np.int8)
    quad_red, pent_blue, red_v, blue_v = _init_counts(colors, quad_tris, pent_tris)

    while red_v > 0 or blue_v > 0:
        if time.time() > deadline:
            break
        # pick a random violating unit
        if red_v > 0:
            if blue_v == 0 or random.random() < red_v / (red_v + blue_v):
                qi = int(np.random.choice(np.flatnonzero(quad_red == 4)))
                t = random.choice(quad_tris[qi])
                if colors[t] != 0:
                    continue
                # flip red -> blue
                colors[t] = 1
                for q in tri_quads[t]:
                    old = quad_red[q]
                    quad_red[q] = old - 1
                    if old == 4:
                        red_v -= 1
                    if old - 1 == 4:
                        red_v += 1
                for p in tri_pents[t]:
                    old = pent_blue[p]
                    pent_blue[p] = old - 1
                    if old == 10:
                        blue_v -= 1
                    if old - 1 == 10:
                        blue_v += 1
            else:
                continue
        else:
            pi = int(np.random.choice(np.flatnonzero(pent_blue == 10)))
            t = random.choice(pent_tris[pi])
            if colors[t] != 1:
                continue
            # flip blue -> red
            colors[t] = 0
            for q in tri_quads[t]:
                old = quad_red[q]
                quad_red[q] = old + 1
                if old == 4:
                    red_v -= 1
                if old + 1 == 4:
                    red_v += 1
            for p in tri_pents[t]:
                old = pent_blue[p]
                pent_blue[p] = old + 1
                if old == 10:
                    blue_v -= 1
                if old + 1 == 10:
                    blue_v += 1

    return colors, red_v == 0 and blue_v == 0


def construct():
    start = time.time()
    deadline = start + 140.0
    best_n = 0
    best_colors = None

    def starts(n):
        s = []
        # structured: sum mod 3
        s.append([0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)])
        # random
        for _ in range(3):
            s.append([random.randint(0, 1) for _ in range(len(list(combinations(range(n), 3))))])
        return s

    for n in range(12, 35):
        if time.time() > deadline - 25:
            break
        idx_of, m, quad_tris, pent_tris, tri_quads, tri_pents = _build(n)
        found = False
        for st in starts(n):
            if time.time() > deadline - 25:
                break
            colors, ok = _local_search(st, n, idx_of, quad_tris, pent_tris, tri_quads, tri_pents, deadline - 25)
            if ok:
                best_n = n
                best_colors = colors.tolist()
                found = True
                break
        if not found and best_n < n:
            # couldn't reach n; stop trying larger
            break

    if best_colors is None:
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        best_n = n

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
