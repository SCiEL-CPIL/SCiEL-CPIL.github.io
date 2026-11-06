import time

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

FULL = (1 << N) - 1

# Edge position lookup (1-based)
_edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    a, b = (u, v) if u < v else (v, u)
    _edge_pos[(a, b)] = i

# Precompute bitmask of incident edges per vertex, sorted by arrival index
_vertex_edges = [[] for _ in range(N)]
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    _vertex_edges[u].append((v, i))
    _vertex_edges[v].append((u, i))
for v in range(N):
    _vertex_edges[v].sort(key=lambda x: x[1])

# Lower bound: max over vertices of the arrival index of the 2nd incident edge
def _second_edge_index(v):
    c = 0
    for nb, idx in _vertex_edges[v]:
        c += 1
        if c == 2:
            return idx
    return 0
LB2 = max(_second_edge_index(v) for v in range(N))

# Per-vertex bitmask of neighbors (full graph) for connectivity checks
_full_adj_mask = [0] * N
for v in range(N):
    m = 0
    for nb, _ in _vertex_edges[v]:
        m |= (1 << nb)
    _full_adj_mask[v] = m


def _build_adj(K):
    """Adjacency as bitmask + sorted neighbor list (by arrival index) for edges <= K."""
    mask = [0] * N
    nbrs = [[] for _ in range(N)]
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        if i > K:
            break
        mask[u] |= (1 << v)
        mask[v] |= (1 << u)
        nbrs[u].append((v, i))
        nbrs[v].append((u, i))
    for v in range(N):
        nbrs[v].sort(key=lambda x: x[1])
    return mask, nbrs


def _connected(mask, start):
    """BFS connectivity over bitmask adjacency."""
    seen = 1 << start
    frontier = 1 << start
    while frontier:
        # expand all neighbors of frontier
        nb = 0
        f = frontier
        while f:
            lsb = f & (-f)
            v = lsb.bit_length() - 1
            f -= lsb
            nb |= mask[v]
        nb &= ~seen
        seen |= nb
        frontier = nb
    return (seen == FULL)


def _find_cycle(K, deadline):
    """Branch-and-bound (bitmask) search for a Hamilton cycle using edges <= K.
    Returns a list of vertices (length N, cycle) or None."""
    mask, nbrs = _build_adj(K)

    # Degree feasibility: every vertex needs >= 2 incident edges
    for v in range(N):
        if mask[v].bit_count() < 2:
            return None

    # Connectivity
    if not _connected(mask, 0):
        return None

    # Warnsdorff ordering: for each vertex, sort unvisited neighbors by their degree
    # (bit_count of mask). Recomputed dynamically is expensive; use static degree as
    # a good proxy, tie-break by arrival index.
    order = []
    for v in range(N):
        lst = list(nbrs[v])  # already sorted by arrival index
        lst.sort(key=lambda x: (mask[x[0]].bit_count(), x[1]))
        order.append(lst)

    path = [0]
    visited = 1
    # iterative DFS with explicit stack to avoid recursion limits
    # state: (vertex, neighbor_index_to_try_next)
    stack = [(0, 0)]
    # We'll implement recursive-style via manual stack.

    # Use recursion with sys.setrecursionlimit high; N=24 depth is fine.
    import sys
    sys.setrecursionlimit(10000)

    def dfs(v, vis, path):
        if time.time() > deadline:
            return None
        if len(path) == N:
            # closing edge v -> 0 must exist
            if mask[v] & 1:  # neighbor 0
                return path[:]
            return None
        # try neighbors in Warnsdorff order
        for nb, idx in order[v]:
            bit = 1 << nb
            if vis & bit:
                continue
            # prune: if nb != 0 and nb has no unvisited neighbor (and nb != 0), dead end
            # i.e., after visiting nb, nb's remaining unvisited neighbors (excluding path) must be >=1 unless nb is the last
            if len(path) < N - 1:
                rem = mask[nb] & ~vis & ~bit  # unvisited neighbors of nb
                if rem == 0:
                    continue
            # connectivity prune: visited set + {nb} must still be able to reach all unvisited
            new_vis = vis | bit
            path.append(nb)
            res = dfs(nb, new_vis, path)
            if res is not None:
                return res
            path.pop()
        return None

    res = dfs(0, 1, path)
    return res


def _cycle_score(cycle):
    m = 0
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        a, b = (u, v) if u < v else (v, u)
        idx = _edge_pos[(a, b)]
        if idx > m:
            m = idx
    return m


def _two_opt_improve(cycle, max_iter=5000):
    n = len(cycle)

    def score(cyc):
        m = 0
        for i in range(n):
            u = cyc[i]
            v = cyc[(i + 1) % n]
            a, b = (u, v) if u < v else (v, u)
            idx = _edge_pos[(a, b)]
            if idx > m:
                m = idx
        return m

    best = score(cycle)
    for _ in range(max_iter):
        improved = False
        # 2-opt
        for i in range(n):
            if improved:
                break
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                newc = cycle[:i + 1] + cycle[i + 1:j + 1][::-1] + cycle[j + 1:]
                s = score(newc)
                if s < best:
                    cycle = newc
                    best = s
                    improved = True
                    break
        if improved:
            continue
        # or-opt: move single vertex
        for i in range(1, n):
            if improved:
                break
            v = cycle[i]
            for j in range(n):
                if j == i or j == (i - 1) % n or j == (i + 1) % n:
                    continue
                rest = cycle[:i] + cycle[i + 1:]
                jj = j if j < i else j - 1
                cand = rest[:jj] + [v] + rest[jj:]
                s = score(cand)
                if s < best:
                    cycle = cand
                    best = s
                    improved = True
                    break
        if not improved:
            break
    return cycle, best


def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    lo, hi = LB2, len(EDGE_ORDER)
    best_cycle = None
    best_K = hi

    # First secure a full-graph cycle
    c = _find_cycle(hi, deadline)
    if c is not None:
        best_cycle = c
        best_K = hi

    # Binary search downward
    while lo < hi and (time.time() - start_time) < deadline - 10:
        mid = (lo + hi) // 2
        c = _find_cycle(mid, deadline)
        if c is not None:
            best_cycle = c
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1

    # Polish
    if best_cycle is not None:
        remaining = deadline - (time.time() - start_time)
        if remaining > 2:
            max_iter = max(500, int(remaining * 400))
            improved, s = _two_opt_improve(best_cycle, max_iter=max_iter)
            if s < _cycle_score(best_cycle):
                best_cycle = improved

    if best_cycle is None:
        return list(range(N))
    return best_cycle
