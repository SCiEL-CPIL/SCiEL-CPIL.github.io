"""Biclique partition: grow-based greedy set-cover + enhanced local search."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_N = P['n']
_ADJ = [set() for _ in range(_N)]
_EDGE_SET = set()
for _a, _b in P['edges']:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)
    _EDGE_SET.add((min(_a, _b), max(_a, _b)))


def _star_partition():
    """Trivial valid partition: one biclique per vertex covering edges to higher-index neighbors."""
    res = []
    for u in range(_N):
        nbrs = [v for v in _ADJ[u] if v > u]
        if nbrs:
            res.append(([u], list(nbrs)))
    return res


def _valid_partition(bics):
    """Return True iff bics is an exact, non-overlapping, host-edge biclique partition."""
    covered = set()
    for L, R in bics:
        if not L or not R:
            return False
        if set(L) & set(R):
            return False
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e not in _EDGE_SET:
                    return False
                if e in covered:
                    return False
                covered.add(e)
    return covered == _EDGE_SET


def _biclique_edges(L, R):
    """Return set of edges in biclique (L,R)."""
    es = set()
    for u in L:
        for v in R:
            es.add((min(u, v), max(u, v)))
    return es


def _greedy_construct(deadline, seed=None):
    """Grow-based greedy set cover with optional randomization."""
    rng = random.Random(seed)
    remaining = set(_EDGE_SET)
    bics = []
    while remaining and time.time() < deadline:
        best_L = None
        best_R = None
        best_count = 0

        # Single-vertex left sides
        for u in range(_N):
            R = [v for v in _ADJ[u] if (min(u, v), max(u, v)) in remaining]
            if len(R) > best_count:
                best_count = len(R)
                best_L = [u]
                best_R = R

        # Seed-edge bicliques with growing L
        edges_list = list(remaining)
        if seed is not None:
            rng.shuffle(edges_list)
        for e in edges_list:
            if time.time() >= deadline:
                break
            a, b = e
            R = [v for v in _ADJ[a] if (min(a, v), max(a, v)) in remaining]
            if not R:
                continue
            Rset = set(R)
            L = [a]
            Lset = set(L)
            changed = True
            while changed:
                changed = False
                for u in _ADJ[b]:
                    if u in Lset:
                        continue
                    if all((min(u, v), max(u, v)) in remaining for v in R):
                        L.append(u)
                        Lset.add(u)
                        changed = True
            count = len(L) * len(R)
            if count > best_count:
                best_count = count
                best_L = L
                best_R = R

        if best_L is None or best_count == 0:
            e = next(iter(remaining))
            best_L = [e[0]]
            best_R = [e[1]]
            best_count = 1

        bics.append((best_L, best_R))
        for u in best_L:
            for v in best_R:
                remaining.discard((min(u, v), max(u, v)))

    for e in list(remaining):
        bics.append(([e[0]], [e[1]]))
    return bics


def _try_merge(bics, i, j):
    """Try to merge bicliques i and j. Return new list or None."""
    L1, R1 = bics[i]
    L2, R2 = bics[j]
    L = list(set(L1) | set(L2))
    R = list(set(R1) | set(R2))
    if not L or not R:
        return None
    if set(L) & set(R):
        return None
    # Check all cross pairs are host edges
    for u in L:
        for v in R:
            if (min(u, v), max(u, v)) not in _EDGE_SET:
                return None
    # Merged edge set must equal union (no new edges introduced)
    e1 = _biclique_edges(L1, R1)
    e2 = _biclique_edges(L2, R2)
    eu = _biclique_edges(L, R)
    if eu != (e1 | e2):
        return None
    new_bics = [b for k, b in enumerate(bics) if k not in (i, j)]
    new_bics.append((L, R))
    return new_bics


def _local_search(bics, deadline):
    """Repeatedly merge pairs until no improvement or deadline."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bics)
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                new_bics = _try_merge(bics, i, j)
                if new_bics is not None:
                    bics = new_bics
                    improved = True
                    break
            if improved:
                break
    return bics


def _split_repack(bics, idxs, deadline):
    """
    Take bicliques at indices idxs, collect their edges, and re-partition
    them using a fresh greedy pass on just those edges.
    """
    if len(idxs) < 2:
        return bics
    edges_to_repack = set()
    for i in idxs:
        L, R = bics[i]
        for u in L:
            for v in R:
                edges_to_repack.add((min(u, v), max(u, v)))

    # Greedy re-partition of just these edges
    remaining = set(edges_to_repack)
    new_bics = []
    while remaining and time.time() < deadline:
        best_L = None
        best_R = None
        best_count = 0
        for u in range(_N):
            R = [v for v in _ADJ[u] if (min(u, v), max(u, v)) in remaining]
            if len(R) > best_count:
                best_count = len(R)
                best_L = [u]
                best_R = R
        # Also try growing from each remaining edge
        for e in list(remaining):
            a, b = e
            R = [v for v in _ADJ[a] if (min(a, v), max(a, v)) in remaining]
            if not R:
                continue
            L = [a]
            Lset = set(L)
            changed = True
            while changed:
                changed = False
                for u in _ADJ[b]:
                    if u in Lset:
                        continue
                    if all((min(u, v), max(u, v)) in remaining for v in R):
                        L.append(u)
                        Lset.add(u)
                        changed = True
            count = len(L) * len(R)
            if count > best_count:
                best_count = count
                best_L = L
                best_R = R
        if best_L is None or best_count == 0:
            e = next(iter(remaining))
            best_L = [e[0]]
            best_R = [e[1]]
        new_bics.append((best_L, best_R))
        for u in best_L:
            for v in best_R:
                remaining.discard((min(u, v), max(u, v)))
    for e in list(remaining):
        new_bics.append(([e[0]], [e[1]]))

    # Only accept if it reduces the count
    if len(new_bics) < len(idxs):
        other_bics = [b for k, b in enumerate(bics) if k not in idxs]
        other_bics.extend(new_bics)
        return other_bics
    return bics


def _local_search_enhanced(bics, deadline):
    """Merge pass + split-repack on pairs of bicliques."""
    # First: standard merge pass
    bics = _local_search(bics, deadline)

    # Second: try split-repack on pairs
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bics)
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                new_bics = _split_repack(bics, [i, j], deadline)
                if new_bics is not bics and len(new_bics) < n:
                    bics = new_bics
                    improved = True
                    break
            if improved:
                break
    return bics


def construct():
    deadline = time.time() + 80.0

    best_bics = None
    best_count = float('inf')

    # Multiple greedy restarts with different seeds
    for seed in [None, 42, 123, 7, 999, 31415]:
        if time.time() >= deadline - 5.0:
            break
        bics = _greedy_construct(deadline - 2.0, seed=seed)
        bics = _local_search_enhanced(bics, deadline - 1.0)

        if not _valid_partition(bics):
            continue

        if len(bics) < best_count:
            best_count = len(bics)
            best_bics = bics

    if best_bics is None:
        best_bics = _star_partition()

    result = [[list(map(int, L)), list(map(int, R))] for L, R in best_bics]
    return result
# EVOLVE-BLOCK-END
