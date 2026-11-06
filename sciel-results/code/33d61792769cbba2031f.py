"""Dense rainbow-4-cycle edge-coloured graph on 20 vertices, 20 colours.
Combined strategy:
  1) Try to find a valid edge-colouring of a dense bipartite K_{10,10} subgraph
     (100 edges) using structured linear forms + fast numpy 4-cycle checks +
     a bounded random repair search. This is the structural idea from the
     inspiration and gives a large jump if it succeeds.
  2) Fall back to the current program's general-graph randomized local search
     (greedy add + recolor) on all 20 vertices, which reliably reaches ~90+.
  3) Keep the best valid result across both, with a deadline guard.
"""
import time
import random
import numpy as np

N = 20
C = 20
DEADLINE = 140.0  # safety margin under 150s kill

# ---------------------------------------------------------------------------
# General-graph machinery (from the current program)
# ---------------------------------------------------------------------------

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)


def _build_base_cycle():
    """Cycle 0-1-2-...-19-0 with colour (u+v)%20. Every 4-cycle is rainbow."""
    edges = {}
    for i in range(N):
        j = (i + 1) % N
        u, v = min(i, j), max(i, j)
        edges[(u, v)] = (i + j) % C
    return edges


def _all_4cycles():
    """Each 4-cycle once as frozenset of 4 edge keys."""
    from itertools import combinations
    verts = list(range(N))
    cycles = []
    for a, b, c, d in combinations(verts, 4):
        for (e1, e2, e3, e4) in [
            ((a, b), (b, c), (c, d), (d, a)),
            ((a, b), (b, d), (d, c), (c, a)),
            ((a, c), (c, b), (b, d), (d, a)),
        ]:
            keys = frozenset(_edge_key(*e) for e in (e1, e2, e3, e4))
            if len(keys) == 4:
                cycles.append(keys)
    return cycles


def _colors_of_cycle(edges, cycle):
    cols = []
    for k in cycle:
        c = edges.get(k, -1)
        if c < 0:
            return None
        cols.append(c)
    return cols


def _is_valid(edges, all_cycles):
    for cyc in all_cycles:
        cols = _colors_of_cycle(edges, cyc)
        if cols is None:
            continue
        if len(set(cols)) != 4:
            return False
    return True


def _try_add_edge(edges, all_cycles, u, v, rng):
    k = _edge_key(u, v)
    if k in edges:
        return False
    used = set()
    for (a, b), c in edges.items():
        if a in (u, v) or b in (u, v):
            used.add(c)
    cand = [c for c in range(C) if c not in used]
    if not cand:
        cand = list(range(C))
    rng.shuffle(cand)
    for c in cand:
        edges[k] = c
        if _is_valid(edges, all_cycles):
            return True
        else:
            del edges[k]
    return False


def _try_recolor(edges, all_cycles, k, rng):
    if k not in edges:
        return False
    old = edges[k]
    bad = set()
    for cyc in all_cycles:
        if k in cyc:
            cols = _colors_of_cycle(edges, cyc)
            if cols is None:
                continue
            if len(set(cols)) != 4:
                seen = set()
                for cc in cols:
                    if cc in seen:
                        bad.add(cc)
                    seen.add(cc)
    cand = [c for c in range(C) if c not in bad and c != old]
    if not cand:
        cand = [c for c in range(C) if c != old]
    rng.shuffle(cand)
    for c in cand:
        edges[k] = c
        if _is_valid(edges, all_cycles):
            return True
    edges[k] = old
    return False


def _general_search(t0, rng, all_cycles, budget):
    """Current-program local search. Returns (best_edges_dict, best_count)."""
    edges = _build_base_cycle()
    best = dict(edges)
    best_count = len(edges)

    def remaining():
        return DEADLINE - (time.time() - t0)

    missing = [(_edge_key(u, v)) for u in range(N) for v in range(u + 1, N)
               if (u, v) not in edges]
    rng.shuffle(missing)
    for k in missing:
        if remaining() < 5.0:
            break
        if _try_add_edge(edges, all_cycles, k[0], k[1], rng):
            if len(edges) > best_count:
                best = dict(edges)
                best_count = len(edges)

    for restart in range(200):
        if remaining() < 8.0:
            break
        missing = [(_edge_key(u, v)) for u in range(N) for v in range(u + 1, N)
                   if (u, v) not in edges]
        rng.shuffle(missing)
        for k in missing:
            if remaining() < 8.0:
                break
            if _try_add_edge(edges, all_cycles, k[0], k[1], rng):
                if len(edges) > best_count:
                    best = dict(edges)
                    best_count = len(edges)
        if remaining() < 8.0:
            break
        if edges:
            ks = list(edges.keys())
            rng.shuffle(ks)
            for k in ks[:10]:
                if remaining() < 8.0:
                    break
                old = edges[k]
                del edges[k]
                if _try_recolor(edges, all_cycles, k, rng):
                    pass
                else:
                    edges[k] = old
                missing2 = [(_edge_key(u, v)) for u in range(N) for v in range(u + 1, N)
                            if (u, v) not in edges]
                if missing2:
                    rng.shuffle(missing2)
                    for k2 in missing2[:50]:
                        if remaining() < 8.0:
                            break
                        if _try_add_edge(edges, all_cycles, k2[0], k2[1], rng):
                            if len(edges) > best_count:
                                best = dict(edges)
                                best_count = len(edges)
                            break
    return best, best_count


# ---------------------------------------------------------------------------
# Bipartite K_{10,10} search (structural idea from the inspiration)
# ---------------------------------------------------------------------------

def _bipartite_search(t0, rng, budget):
    """Try to find a valid rainbow-4-cycle colouring of K_{10,10}.
    Returns a list of [u,v,c] edges (on vertices 0..19) or None."""
    m = 10
    left = list(range(m))
    right = list(range(m, 2 * m))

    # Precompute all 4-cycles of K_{10,10} as index tuples (l1,l2,r1,r2).
    cycles = []
    for l1 in range(m):
        for l2 in range(l1 + 1, m):
            for r1 in range(m):
                for r2 in range(r1 + 1, m):
                    cycles.append((l1, l2, r1, r2))
    cycles = np.array(cycles, dtype=np.int64)
    l1 = cycles[:, 0]; l2 = cycles[:, 1]; r1 = cycles[:, 2]; r2 = cycles[:, 3]

    def check_mat(M):
        """M is (m,m) int array. True iff every 4-cycle is rainbow."""
        c1 = M[l1, r1]; c2 = M[l2, r1]; c3 = M[l2, r2]; c4 = M[l1, r2]
        # rainbow iff all four pairwise distinct
        bad = ((c1 == c2) | (c1 == c3) | (c1 == c4) |
               (c2 == c3) | (c2 == c4) | (c3 == c4))
        return not bool(bad.any())

    def mat_to_edges(M):
        return [[left[l], right[r], int(M[l, r])]
                for l in range(m) for r in range(m)]

    # --- structured linear forms: C[l][r] = (a*l + b*r) % C ---
    for a in range(1, C):
        for b in range(1, C):
            if time.time() - t0 > budget:
                return None
            M = ((a * np.arange(m)[:, None] + b * np.arange(m)[None, :]) % C).astype(np.int64)
            if check_mat(M):
                return mat_to_edges(M)

    # --- C[l][r] = (l*r + c) % C ---
    for c in range(C):
        if time.time() - t0 > budget:
            return None
        M = ((np.arange(m)[:, None] * np.arange(m)[None, :] + c) % C).astype(np.int64)
        if check_mat(M):
            return mat_to_edges(M)

    # --- bounded random repair on K_{10,10} ---
    for attempt in range(3000):
        if time.time() - t0 > budget:
            return None
        M = (rng.randrange(C) * np.ones((m, m))).astype(np.int64)
        M = np.array([[rng.randrange(C) for _ in range(m)] for _ in range(m)], dtype=np.int64)
        for _ in range(120):
            if time.time() - t0 > budget:
                break
            c1 = M[l1, r1]; c2 = M[l2, r1]; c3 = M[l2, r2]; c4 = M[l1, r2]
            bad = ((c1 == c2) | (c1 == c3) | (c1 == c4) |
                   (c2 == c3) | (c2 == c4) | (c3 == c4))
            if not bool(bad.any()):
                return mat_to_edges(M)
            idx = int(np.flatnonzero(bad)[rng.randrange(int(bad.sum()))])
            li, lj, ri, rj = int(l1[idx]), int(l2[idx]), int(r1[idx]), int(r2[idx])
            pos = rng.randrange(4)
            if pos == 0:
                M[li, ri] = rng.randrange(C)
            elif pos == 1:
                M[lj, ri] = rng.randrange(C)
            elif pos == 2:
                M[lj, rj] = rng.randrange(C)
            else:
                M[li, rj] = rng.randrange(C)
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def construct():
    t0 = time.time()
    rng = random.Random(20240517)

    all_cycles = _all_4cycles()

    # Phase A: try the dense bipartite K_{10,10} (100 edges) first.
    bip_edges = _bipartite_search(t0, rng, budget=30.0)
    best_edges = None
    best_count = 0
    if bip_edges is not None:
        # verify it's a valid subgraph of the general problem (it is, by
        # construction every 4-cycle within the bipartite part is rainbow;
        # there are no 4-cycles using vertices outside the bipartite part
        # because those vertices have no edges).
        best_edges = bip_edges
        best_count = len(bip_edges)

    # Phase B: general-graph local search to fill remaining time.
    gen_best, gen_count = _general_search(t0, rng, all_cycles, budget=DEADLINE - 3.0)
    if gen_count > best_count:
        best_edges = [[u, v, gen_best[(u, v)]] for (u, v) in gen_best]
        best_count = gen_count

    if best_edges is None:
        # absolute fallback: the base cycle is always valid.
        base = _build_base_cycle()
        best_edges = [[u, v, base[(u, v)]] for (u, v) in base]

    return best_edges


if __name__ == "__main__":
    res = construct()
    print(len(res))
