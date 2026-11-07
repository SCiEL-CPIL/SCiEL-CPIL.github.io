"""Solver for few-color graph coloring with no bicolored cycles.

This program keeps the same structural design family (Cayley-based construction
plus a merge-based reduction) and adds a fast, exact local-search routine that
tries to reach 4 colors using a lightweight cycle-check.  It also re-organises
the code for clarity and speed while preserving the public interface.
"""
import time
from math import gcd
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# Graph data (fixed by the task)
# ---------------------------------------------------------------------------
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Precompute adjacency lists once
_ADJ = [[] for _ in range(N)]
for _u, _v in EDGES:
    _ADJ[_u].append(_v)
    _ADJ[_v].append(_u)


# ---------------------------------------------------------------------------
# Fast exact validity check
# ---------------------------------------------------------------------------
def _valid(coloring):
    """Return True iff coloring is a proper coloring with no bicolored cycles."""
    # Adjacency check
    for u, v in EDGES:
        if coloring[u] == coloring[v]:
            return False
    # Group edges by color pair
    pair_edges = defaultdict(list)
    for u, v in EDGES:
        cu, cv = coloring[u], coloring[v]
        key = (cu, cv) if cu < cv else (cv, cu)
        pair_edges[key].append((u, v))
    # Check each color-pair subgraph is a forest via union-find
    for _, edges in pair_edges.items():
        parent = list(range(N))
        rank = [0] * N
        for u, v in edges:
            # find with path compression
            ru = u
            while parent[ru] != ru:
                parent[ru] = parent[parent[ru]]
                ru = parent[ru]
            rv = v
            while parent[rv] != rv:
                parent[rv] = parent[parent[rv]]
                rv = parent[rv]
            if ru == rv:
                return False
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
    return True


# ---------------------------------------------------------------------------
# Cayley-structure construction: c(v) = (a * v) % k
# ---------------------------------------------------------------------------
def _cayley_search(deadline):
    """Find the best Cayley-type coloring (fewest colors)."""
    best = None
    best_count = N + 1
    for k in range(3, 13):
        if time.time() > deadline:
            break
        for a in range(1, k):
            if gcd(k, a) != 1 or gcd(k, 7) != 1:
                continue
            coloring = [(a * v) % k for v in range(N)]
            if _valid(coloring):
                used = len(set(coloring))
                if used < best_count:
                    best = coloring
                    best_count = used
                break  # first valid a for this k suffices
    return best, best_count


# ---------------------------------------------------------------------------
# Merge-based reduction (same family as before)
# ---------------------------------------------------------------------------
def _reduce(coloring, deadline):
    """Iteratively merge color classes if the result remains valid."""
    if coloring is None:
        return None
    cur = list(coloring)
    n = len(cur)
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        distinct = sorted(set(cur))
        if len(distinct) <= 2:
            break
        counts = Counter(cur)
        # Try merging pairs, most frequent first
        ordered = sorted(distinct, key=lambda c: (-counts[c], c))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                ca, cb = ordered[i], ordered[j]
                # Quick filter: if no edge currently has endpoints colored ca and cb,
                # merging cb -> ca cannot create a new bicolored edge (only changes labels)
                # But it CAN change which color pairs exist, so we must verify.
                new = [ca if c == cb else c for c in cur]
                if _valid(new):
                    cur = new
                    changed = True
                    break
            if changed:
                break
    return cur


# ---------------------------------------------------------------------------
# Local search: try to reach 4 colors via targeted recoloring
# ---------------------------------------------------------------------------
def _local_search_4(coloring, deadline):
    """
    Given a valid coloring, try to reduce to 4 colors by recoloring individual
    vertices.  Uses a fast approximate validity check during the search and
    only runs the exact _valid at the end of each successful move.
    """
    if coloring is None:
        return None
    cur = list(coloring)
    target = 4

    def _fast_ok(coloring):
        """Fast necessary-condition check (not sufficient for bicolored cycles)."""
        for u, v in EDGES:
            if coloring[u] == coloring[v]:
                return False
        # Count edges per color pair; if any pair has > (n_vertices_in_pair) edges, cycle likely
        # This is a weak heuristic; we rely on exact _valid for acceptance.
        return True

    # We'll do a simple hill-climbing / simulated-annealing style search.
    # At each step, pick a random vertex, try changing its color to one of 0..3,
    # and accept if the result is valid (exact check).
    # To speed up, we first do a quick adjacency check.
    import random
    rng = random.Random(42)
    colors = list(range(target))
    
    # Precompute for speed: for each vertex, the set of forbidden colors from neighbors
    # This changes as we recolor, so we recompute on the fly (cheap for degree 4).
    
    best = cur[:]
    best_count = len(set(cur))
    if best_count <= target:
        return best
    
    steps = 0
    max_steps = 20000
    while time.time() < deadline - 2 and steps < max_steps:
        steps += 1
        v = rng.randrange(N)
        old_c = cur[v]
        # Choose a new color different from neighbors
        forbidden = set()
        for nb in _ADJ[v]:
            forbidden.add(cur[nb])
        candidates = [c for c in colors if c not in forbidden]
        if not candidates:
            continue
        new_c = rng.choice(candidates)
        if new_c == old_c:
            continue
        cur[v] = new_c
        # Quick adjacency check for the changed vertex (already done above)
        # Exact validity check
        if _valid(cur):
            used = len(set(cur))
            if used < best_count:
                best = cur[:]
                best_count = used
                if best_count <= target:
                    return best
        else:
            # Revert
            cur[v] = old_c
    return best


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    # Safety margin: return 5 seconds before the 90 s kill
    deadline = start + 85

    # 1. Cayley construction (fast, gives a valid 6-coloring typically)
    best, best_count = _cayley_search(deadline)

    # 2. Fallback: trivial 50-coloring if Cayley failed (should not happen)
    if best is None:
        best = list(range(N))
        best_count = N

    # 3. Merge-based reduction
    reduced = _reduce(best, deadline)
    if reduced is not None:
        ok = _valid(reduced)
        if ok:
            cnt = len(set(reduced))
            if cnt < best_count:
                best = reduced
                best_count = cnt

    # 4. Local search to try to reach 4 colors
    if time.time() < deadline - 5 and best_count > 4:
        improved = _local_search_4(best, deadline)
        if improved is not None:
            if _valid(improved):
                cnt = len(set(improved))
                if cnt < best_count:
                    best = improved
                    best_count = cnt

    # 5. Final verification
    if not _valid(best):
        # Absolute fallback
        return [int(x) for x in range(N)]

    return [int(x) for x in best]
