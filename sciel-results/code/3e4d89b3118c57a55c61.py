"""Biclique partition via greedy construction + local search (merge/split)."""
import time

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

# Precompute edge set and adjacency for fast lookup
_EDGE_SET = frozenset(frozenset(e) for e in P['edges'])
_ADJ = [set() for _ in range(P['n'])]
for _a, _b in P['edges']:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)

def _cross_ok(L, R):
    """Check every (u in L, v in R) is an edge."""
    for u in L:
        for v in R:
            if v not in _ADJ[u]:
                return False
    return True

def _biclique_edges(L, R):
    """Set of frozenset edges covered by this biclique."""
    s = set()
    for u in L:
        for v in R:
            s.add(frozenset((u, v)))
    return s

def _valid_partition(bics):
    """Return (is_valid, covered_set, count) for a list of bicliques."""
    covered = set()
    for L, R in bics:
        if not L or not R:
            return False, covered, len(bics)
        if any(v in L for v in R):
            return False, covered, len(bics)
        for u in L:
            for v in R:
                e = frozenset((u, v))
                if e not in _EDGE_SET:
                    return False, covered, len(bics)
                if e in covered:
                    return False, covered, len(bics)
                covered.add(e)
    return covered == set(_EDGE_SET), covered, len(bics)

def _seed_star_partition():
    """Star partition: one biclique per vertex with its neighbors (seed-like)."""
    res = []
    for u in range(P['n']):
        nbrs = [v for v in _ADJ[u] if v > u]  # avoid duplicate edges
        if nbrs:
            res.append(([u], nbrs))
    return res

def _greedy_construct():
    """Greedy: repeatedly pick the biclique covering the most remaining edges."""
    remaining = set(_EDGE_SET)
    bics = []
    while remaining:
        # Pick a seed edge, grow a biclique
        seed_edge = next(iter(remaining))
        a, b = tuple(seed_edge)
        # Candidate left = {a}, right = neighbors of a in remaining
        # Try to grow: left can be neighbors of b that connect to all of right
        # Simple: start L={a}, R = {v in adj[a] : (a,v) in remaining}
        L = [a]
        R = [v for v in _ADJ[a] if frozenset((a, v)) in remaining]
        # Try to add to L: vertices u such that u connects to all of R and (u,v) in remaining
        changed = True
        while changed:
            changed = False
            for u in _ADJ[b]:
                if u == a or u in L:
                    continue
                if all(frozenset((u, v)) in remaining for v in R):
                    L.append(u)
                    changed = True
        if not L or not R:
            L, R = [a], [b]
        # Remove covered edges
        for u in L:
            for v in R:
                remaining.discard(frozenset((u, v)))
        bics.append((L, R))
    return bics

def _try_merge(bics, i, j):
    """Try to merge bicliques i and j into one. Return (new_bics, ok)."""
    L1, R1 = bics[i]
    L2, R2 = bics[j]
    # Merged biclique must have all cross pairs as edges.
    # The union L1+L2 and R1+R2 requires cross edges between L1-R2 and L2-R1.
    # That's too strong. Instead, we can only merge if the union is itself a valid biclique.
    L = list(set(L1) | set(L2))
    R = list(set(R1) | set(R2))
    if not L or not R:
        return None, False
    if set(L) & set(R):
        return None, False
    if not _cross_ok(L, R):
        return None, False
    # Ensure the union's edge set equals the union of the two bicliques' edge sets
    # (no extra edges beyond what was covered, since partition must be exact)
    e1 = _biclique_edges(L1, R1)
    e2 = _biclique_edges(L2, R2)
    eu = _biclique_edges(L, R)
    if eu != (e1 | e2):
        return None, False
    new_bics = [b for k, b in enumerate(bics) if k not in (i, j)]
    new_bics.append((L, R))
    return new_bics, True

def _local_search(bics, deadline):
    """Repeatedly try to merge any pair of bicliques until no improvement or deadline."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bics)
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                new_bics, ok = _try_merge(bics, i, j)
                if ok:
                    bics = new_bics
                    improved = True
                    break
            if improved:
                break
    return bics

def _final_partition(bics):
    """Convert to required list-of-lists-of-int form."""
    return [[list(map(int, L)), list(map(int, R))] for L, R in bics]

def construct():
    deadline = time.time() + 80  # safety margin under 90s kill
    # Start from greedy (usually fewer than star)
    bics = _greedy_construct()
    # Local search: merge pairs
    bics = _local_search(bics, deadline)
    # Validate; if invalid, fall back to star
    ok, _, _ = _valid_partition(bics)
    if not ok:
        bics = _seed_star_partition()
    return _final_partition(bics)
# EVOLVE-BLOCK-END
