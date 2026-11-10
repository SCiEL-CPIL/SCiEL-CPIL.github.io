import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 140.0   # safety margin before 150s kill

# Precompute all 1820 K4s and, for each, the 6 edges (indices into the 120-edge list).
_EDGE_INDEX = {}
_EDGE_LIST = []
for _i in range(16):
    for _j in range(_i + 1, 16):
        _EDGE_INDEX[(_i, _j)] = len(_EDGE_LIST)
        _EDGE_LIST.append((_i, _j))
K4S = list(combinations(range(16), 4))
K4_EDGES = []
for _k4 in K4S:
    _idx = []
    for _a, _b in combinations(_k4, 2):
        _idx.append(_EDGE_INDEX[(_a, _b)])
    K4_EDGES.append(_idx)
K4_EDGES = np.array(K4_EDGES, dtype=np.int32)  # shape (1820, 6)
EDGE_VERTS = np.array(_EDGE_LIST, dtype=np.int32)  # shape (120, 2)


def _count_k4_colors(colors):
    """Return number of K4s that span >= 5 distinct colours (validity count)."""
    cols = np.asarray(colors, dtype=np.int32)
    # for each K4, count distinct colours among its 6 edges
    valid = 0
    for idx in K4_EDGES:
        c = cols[idx]
        if len(set(c.tolist())) >= 5:
            valid += 1
    return valid


def _violations(colors):
    """Return list of K4 indices that violate (span < 5 colours)."""
    cols = np.asarray(colors, dtype=np.int32)
    viol = []
    for k, idx in enumerate(K4_EDGES):
        c = cols[idx]
        if len(set(c.tolist())) < 5:
            viol.append(k)
    return viol


def _greedy(n_colors=13, rng=None):
    """Greedy assignment: for each edge pick colour with smallest matching number,
    tie-break by least load. Returns (colors, matching_number_matrix)."""
    if rng is None:
        rng = random.Random()
    colors = [-1] * 120
    # matching number per (K4, colour): count of edges of that colour in the K4
    # stored as dict keyed by K4 index -> list of counts per colour (only track small)
    # For speed, track per-K4 per-colour edge set to check disjointness.
    k4_color_edges = [dict() for _ in K4S]  # k4 -> {color: [edge_idx,...]}
    load = [0] * n_colors
    for e, (u, v) in enumerate(_EDGE_LIST):
        # K4s containing both u and v
        best_color = -1
        best_match = 10**9
        best_load = 10**9
        # determine candidate colours: all n_colors
        # For each colour, the matching number after adding e = max(current matching,
        #   (1 + max over edges sharing a vertex with e of 1)) ... simpler: recompute
        # matching number of colour class restricted to K4 after adding e.
        for c in range(n_colors):
            # find K4s containing e
            pass
        # Instead evaluate per-K4 feasibility is heavy; do global matching number of colour class.
        # Use a lighter proxy: number of edges of colour c in each K4 (must stay <=2) and
        # disjointness. We'll compute matching number of the colour subgraph globally is too big.
        # Use per-K4 counts: choose colour minimizing max over K4s containing e of (count_c[K4]+1)
        # and ensure disjointness (no two edges of colour c share a vertex within a K4).
        # Precompute for each edge the list of K4s containing it.
        best = None
        for c in range(n_colors):
            k4s_e = EDGE_K4S[e]
            maxcnt = 0
            ok = True
            for k in k4s_e:
                edges_c = k4_color_edges[k].get(c)
                if edges_c is None:
                    cnt = 1
                else:
                    # check disjointness with e
                    for ee in edges_c:
                        a, b = _EDGE_LIST[ee]
                        if a == u or a == v or b == u or b == v:
                            ok = False
                            break
                    if not ok:
                        break
                    cnt = len(edges_c) + 1
                if cnt > 2:
                    ok = False
                    break
                if cnt > maxcnt:
                    maxcnt = cnt
            if not ok:
                continue
            l = load[c]
            key = (maxcnt, l)
            if best is None or key < best[0]:
                best = (key, c)
        if best is None:
            # fallback: least loaded colour ignoring constraints
            c = min(range(n_colors), key=lambda x: load[x])
        else:
            c = best[1]
        colors[e] = c
        load[c] += 1
        for k in EDGE_K4S[e]:
            k4_color_edges[k].setdefault(c, []).append(e)
    return colors


# Precompute, for each edge, the list of K4 indices that contain it.
EDGE_K4S = [[] for _ in range(120)]
for _k, _k4 in enumerate(K4S):
    for _e in K4_EDGES[_k]:
        EDGE_K4S[_e].append(_k)


def _polish(colors, rng, time_budget=20.0):
    """Local search: repeatedly recolor a random violating K4's edge to fix it."""
    colors = list(colors)
    deadline = time.time() + time_budget
    while time.time() < deadline:
        viol = _violations(colors)
        if not viol:
            break
        k = viol[rng.randrange(len(viol))]
        idx = K4_EDGES[k]
        c = np.array(colors, dtype=np.int32)[idx]
        distinct = set(c.tolist())
        # try to recolor one of the edges to a new colour not in distinct
        order = list(range(6))
        rng.shuffle(order)
        improved = False
        for pos in order:
            e = idx[pos]
            for newc in range(13):
                if newc in distinct:
                    continue
                # tentatively recolor e -> newc, check all K4s containing e stay valid
                oldc = colors[e]
                colors[e] = newc
                ok = True
                for k2 in EDGE_K4S[e]:
                    cc = np.array(colors, dtype=np.int32)[K4_EDGES[k2]]
                    if len(set(cc.tolist())) < 5:
                        ok = False
                        break
                if ok:
                    improved = True
                    break
                else:
                    colors[e] = oldc
            if improved:
                break
        if not improved:
            # try a random recolor of any edge in the K4 to reduce distinct collapse
            e = idx[rng.randrange(6)]
            colors[e] = rng.randrange(13)
    return colors


def construct():
    global _START, _DEADLINE
    rng = random.Random(20240517)

    # Build a greedy 13-colouring.
    colors = _greedy(13, rng)
    colors = list(colors)

    # Polish to fix any violations, bounded by time.
    remaining = _DEADLINE - time.time()
    if remaining > 5:
        colors = _polish(colors, rng, time_budget=min(25.0, max(0.0, remaining - 5.0)))

    # Final safety: if still invalid, fall back to the guaranteed-valid 17-colour scheme.
    if _violations(colors):
        colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

    # Ensure colours are within 0..119 and exactly 120 values.
    colors = [int(c) % 120 for c in colors]
    assert len(colors) == 120
    return colors
