"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

# Precompute graph structures once at import time.
_N = P['n']
_EDGES = [(min(a, b), max(a, b)) for a, b in P['edges']]
_EDGE_SET = frozenset(_EDGES)
_ADJ = [set() for _ in range(_N)]
for _a, _b in _EDGES:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)
_DEG = [len(_ADJ[i]) for i in range(_N)]
# Edge conflict graph: two edges conflict if they share a vertex (they cannot be
# in the same biclique). Used for a lower bound via maximum clique.
_EDGE_IDX = {e: i for i, e in enumerate(_EDGES)}
_m_edges = len(_EDGES)
# For each edge, the set of edge-indices that conflict with it.
_EDGE_CONFLICT = [set() for _ in range(_m_edges)]
for i in range(_m_edges):
    a, b = _EDGES[i]
    for v in (a, b):
        for nb in _ADJ[v]:
            for e2 in ((v, nb), (nb, v)):
                pass
# Build conflict more directly.
_EDGE_CONFLICT = [set() for _ in range(_m_edges)]
for i in range(_m_edges):
    a, b = _EDGES[i]
    for j in range(_m_edges):
        if i == j:
            continue
        c, d = _EDGES[j]
        if a in (c, d) or b in (c, d):
            _EDGE_CONFLICT[i].add(j)


def _max_clique_max_card(adj_sets, nodes):
    """Bron–Kerbosch with pivot; returns max clique size among `nodes`."""
    best = 0
    U = list(nodes)
    Pset = set(U)
    Xset = set()
    R = set()

    def bk(Pset, Xset, R):
        nonlocal best
        if not Pset and not Xset:
            if len(R) > best:
                best = len(R)
            return
        if len(R) + len(Pset) <= best:
            return
        # choose pivot
        if Xset:
            x = next(iter(Xset))
            cands = Pset - adj_sets[x]
        else:
            cands = Pset
        if not cands:
            cands = Pset
        for v in list(cands):
            R.add(v)
            bk(Pset & adj_sets[v], Xset & adj_sets[v], R)
            R.discard(v)
            Pset.discard(v)
            Xset.add(v)
            if len(R) + len(Pset) <= best:
                break

    bk(Pset, Xset, R)
    return best


def _lb_clique_conflict(remaining):
    """Lower bound: max clique in the edge-conflict graph over `remaining` edge-indices."""
    if not remaining:
        return 0
    # Only consider conflict edges restricted to remaining.
    rem_set = set(remaining)
    sub_adj = {i: (adj & rem_set) for i, adj in _EDGE_CONFLICT.items() if i in rem_set}
    if not sub_adj:
        return 0
    return _max_clique_max_card(sub_adj, list(rem_set))


def _greedy_bipartite_from_edge(a, b):
    """Given a seed edge (a,b), grow a maximal biclique via alternating expansion."""
    L = {a}
    R = {b}
    changed = True
    while changed:
        changed = False
        # All vertices in L must be adjacent to all in R; add to R any vertex
        # adjacent to every vertex in L (and not in L).
        for v in range(_N):
            if v not in L and v not in R:
                if all(v in _ADJ[u] for u in L):
                    R.add(v)
                    changed = True
        for v in range(_N):
            if v not in L and v not in R:
                if all(v in _ADJ[u] for u in R):
                    L.add(v)
                    changed = True
    # Ensure the seed edge is fully covered (it is, by construction).
    return L, R


def _greedy_partition():
    """Greedy: repeatedly pick a maximal biclique covering the most uncovered edges."""
    remaining = set(_EDGE_IDX.values())
    bicliques = []
    while remaining:
        # Candidate: for each remaining edge, grow a biclique and count coverage.
        best_cov = -1
        best_L = None
        best_R = None
        for ei in list(remaining):
            a, b = _EDGES[ei]
            L, R = _greedy_bipartite_from_edge(a, b)
            Lf = frozenset(L)
            Rf = frozenset(R)
            # Count how many remaining edges lie inside L x R.
            cov = 0
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in _EDGE_SET and e in remaining:
                        cov += 1
            if cov > best_cov:
                best_cov = cov
                best_L = L
                best_R = R
                if best_cov == len(remaining):
                    break
        if best_L is None:
            # Fallback: single edge.
            ei = next(iter(remaining))
            a, b = _EDGES[ei]
            best_L, best_R = {a}, {b}
        bicliques.append((best_L, best_R))
        # Remove covered edges.
        for u in best_L:
            for v in best_R:
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)
    return bicliques


def _validate(bicliques):
    """Return True if bicliques form a valid exact partition."""
    seen = set()
    for L, R in bicliques:
        L = list(L)
        R = list(R)
        if not L or not R:
            return False
        if set(L) & set(R):
            return False
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e not in _EDGE_SET:
                    return False
                if e in seen:
                    return False
                seen.add(e)
    return seen == _EDGE_SET


def _branch_and_bound(remaining, current, best, deadline):
    """Branch-and-bound over edge-conflict clique choices.
    remaining: set of edge-indices still uncovered.
    current: list of (L,R) bicliques chosen so far.
    best: mutable [count] for best solution; best_sol mutable [list].
    Returns nothing; updates best.
    """
    if time.time() > deadline:
        return
    if not remaining:
        if len(current) < best[0]:
            best[0] = len(current)
            best[1] = [ (set(L), set(R)) for L, R in current ]
        return
    # Lower bound via conflict clique.
    lb = _lb_clique_conflict(remaining)
    if len(current) + lb >= best[0]:
        return

    # Choose a pivot edge (the one with most conflicts among remaining).
    rem_list = list(remaining)
    pivot = max(rem_list, key=lambda i: len(_EDGE_CONFLICT[i] & remaining))
    pa, pb = _EDGES[pivot]

    # Candidate bicliques: all maximal bicliques containing the pivot edge.
    # Enumerate via subsets of common neighbors (bounded), dedupe by frozenset.
    common = _ADJ[pa] & _ADJ[pb]
    cand_set = set()
    # Always include the trivial one.
    cand_set.add((frozenset({pa}), frozenset({pb})))
    cand_set.add((frozenset({pb}), frozenset({pa})))
    # Try growing: include various subsets of common neighbors.
    common_list = list(common)
    # Limit enumeration to keep it fast.
    max_sub = 6
    for size in range(0, min(max_sub, len(common_list)) + 1):
        for sub in combinations(common_list, size):
            S = set(sub)
            # L = {pa} + S, R = {pb}  (ensure no overlap and all cross edges exist)
            L = {pa} | S
            R = {pb}
            if L & R:
                continue
            ok = True
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e not in _EDGE_SET:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                cand_set.add((frozenset(L), frozenset(R)))
            # Also the mirror: L={pb}+S, R={pa}
            L2 = {pb} | S
            R2 = {pa}
            if L2 & R2:
                continue
            ok = True
            for u in L2:
                for v in R2:
                    e = (min(u, v), max(u, v))
                    if e not in _EDGE_SET:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                cand_set.add((frozenset(L2), frozenset(R2)))

    # Score candidates by coverage of remaining edges; try best first.
    scored = []
    for L, R in cand_set:
        cov = 0
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                idx = _EDGE_IDX.get(e)
                if idx is not None and idx in remaining:
                    cov += 1
        scored.append((cov, L, R))
    scored.sort(reverse=True)

    for cov, L, R in scored:
        if time.time() > deadline:
            return
        new_remaining = set(remaining)
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                idx = _EDGE_IDX.get(e)
                if idx is not None:
                    new_remaining.discard(idx)
        current.append((L, R))
        _branch_and_bound(new_remaining, current, best, deadline)
        current.pop()


def construct():
    deadline = time.time() + 50.0  # safety margin under the 90s kill.
    rng = random.Random(12345)

    # 1) Greedy initializer.
    best_count = 10 ** 9
    best_sol = None
    for attempt in range(200):
        if time.time() > deadline - 5.0:
            break
        bicliques = _greedy_partition()
        if _validate(bicliques):
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_sol = bicliques
                if best_count <= 8:
                    break

    # 2) Branch-and-bound improvement from the greedy solution's remaining structure.
    # We run B&B directly from scratch with the greedy count as initial upper bound.
    if best_sol is not None:
        best = [len(best_sol), [ (set(L), set(R)) for L, R in best_sol ]]
        _branch_and_bound(set(range(_m_edges)), [], best, deadline)
        if best[0] < best_count:
            best_count = best[0]
            best_sol = best[1]

    # 3) Local search: merge/replace to reduce count.
    if best_sol is not None:
        for _ in range(300):
            if time.time() > deadline - 3.0:
                break
            cur = [ (set(L), set(R)) for L, R in best_sol ]
            # Try dropping one biclique and re-covering its edges with others.
            improved = False
            for i in range(len(cur)):
                L, R = cur[i]
                # Edges covered by this biclique.
                cov_edges = set()
                for u in L:
                    for v in R:
                        cov_edges.add((min(u, v), max(u, v)))
                # Can other bicliques already cover these? No (partition). Instead,
                # try to find a single biclique that covers cov_edges plus possibly more,
                # and see if we can merge with an existing one.
                # Simple merge: find j whose biclique can be enlarged to include
                # all edges of i without introducing non-edges.
                for j in range(len(cur)):
                    if i == j:
                        continue
                    Lj, Rj = cur[j]
                    # Try union variants.
                    for (Lc, Rc) in [
                        (Lj | L, Rj | R),
                        (Lj | L, Rj),
                        (Lj, Rj | R),
                        (Lj | L, Rj & R),
                        (Lj & L, Rj | R),
                    ]:
                        if not Lc or not Rc or (Lc & Rc):
                            continue
                        ok = True
                        for u in Lc:
                            for v in Rc:
                                e = (min(u, v), max(u, v))
                                if e not in _EDGE_SET:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        # Does the merged biclique cover all of cov_edges?
                        merged_edges = set()
                        for u in Lc:
                            for v in Rc:
                                merged_edges.add((min(u, v), max(u, v)))
                        if cov_edges <= merged_edges:
                            # Replace j with merged, remove i.
                            new_cur = [cur[k] for k in range(len(cur)) if k != i and k != j]
                            new_cur.append((Lc, Rc))
                            if _validate(new_cur):
                                best_sol = new_cur
                                best_count = len(new_cur)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

    # Final validation and type conversion.
    if best_sol is None:
        # Fallback to seed-style (always valid).
        best_sol = [ ( {u}, {v for a, v in _EDGES if a == u} ) for u in range(_N) if _DEG[u] > 0 ]

    if not _validate(best_sol):
        # Emergency fallback.
        best_sol = [ ( {u}, {v for a, v in _EDGES if a == u} ) for u in range(_N) if _DEG[u] > 0 ]

    out = []
    for L, R in best_sol:
        out.append([list(int(x) for x in L), list(int(x) for x in R)])
    return out
# EVOLVE-BLOCK-END
