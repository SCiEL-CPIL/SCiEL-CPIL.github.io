"""Combined: binary search on prefix + Warnsdorff DFS + greedy fallback."""
import time

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
TOTAL_EDGES = len(EDGE_ORDER)

# Precompute 1-based arrival index for each edge
_ARRIVAL = {}
for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
    _ARRIVAL[(min(_u, _v), max(_u, _v))] = _i


def _cycle_score(cycle):
    """Return the max 1-based arrival index over the cycle's edges."""
    m = 0
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        a = _ARRIVAL[(min(u, v), max(u, v))]
        if a > m:
            m = a
    return m


def _build_adj(prefix_len):
    """Build adjacency list (list of sets) using first prefix_len edges."""
    adj = [set() for _ in range(N)]
    for i in range(prefix_len):
        u, v = EDGE_ORDER[i]
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _deg2_check(adj):
    """All vertices must have degree >= 2 for a Hamilton cycle."""
    for v in range(N):
        if len(adj[v]) < 2:
            return False
    return True


def _dfs_hamilton(adj, start, deadline, best_score):
    """
    Backtracking DFS with Warnsdorff-style heuristic.
    Returns a cycle (list of N vertices) or None.
    """
    visited = [False] * N
    visited[start] = True
    path = [start]

    # Warnsdorff: number of unvisited neighbors for each vertex
    def warnsdorff(v):
        cnt = 0
        for w in adj[v]:
            if not visited[w]:
                cnt += 1
        return cnt

    def dfs():
        if time.time() > deadline:
            return None
        if len(path) == N:
            last = path[-1]
            if start in adj[last]:
                # Verify score improvement
                cyc = path[:]
                sc = _cycle_score(cyc)
                if sc < best_score:
                    return cyc
            return None

        cur = path[-1]
        # Order neighbors by Warnsdorff (fewest onward options first)
        cands = [v for v in adj[cur] if not visited[v]]
        # If only one unvisited left, it must connect back to start
        if len(cands) == 1:
            nb = cands[0]
            if start not in adj[nb]:
                return None
            cands = [nb]
        else:
            cands.sort(key=lambda v: warnsdorff(v))

        for nb in cands:
            visited[nb] = True
            path.append(nb)
            res = dfs()
            if res is not None:
                return res
            path.pop()
            visited[nb] = False
        return None

    return dfs()


def _find_at_prefix(prefix_len, deadline, best_score):
    """Try to find a Hamilton cycle within the prefix_len edge subgraph."""
    if prefix_len < N:
        return None
    adj = _build_adj(prefix_len)
    if not _deg2_check(adj):
        return None
    # Try multiple start vertices to improve chances
    for start in range(N):
        if time.time() > deadline:
            break
        res = _dfs_hamilton(adj, start, deadline, best_score)
        if res is not None:
            return res
    return None


def _greedy_2regular(deadline):
    """
    Greedily add edges in arrival order, keeping each vertex degree <= 2,
    avoiding premature cycles. If we end with a single cycle of length N, return it.
    """
    degree = [0] * N
    adj = [set() for _ in range(N)]
    edges = []
    for i in range(TOTAL_EDGES):
        if time.time() > deadline:
            break
        u, v = EDGE_ORDER[i]
        if degree[u] >= 2 or degree[v] >= 2:
            continue
        # Check if adding this edge forms a cycle shorter than N
        # (i.e., u and v are already connected in the current graph)
        if u != v:
            # BFS/DFS to see if u and v are in the same component
            seen = set()
            stack = [u]
            seen.add(u)
            same_comp = False
            while stack:
                x = stack.pop()
                if x == v:
                    same_comp = True
                    break
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            if same_comp:
                # Adding this edge closes a cycle; only allow if it's the full N-cycle
                if len(edges) != N - 1:
                    continue
        # Add edge
        adj[u].add(v)
        adj[v].add(u)
        degree[u] += 1
        degree[v] += 1
        edges.append((u, v))
        if len(edges) == N:
            break

    if len(edges) != N:
        return None
    # Verify it's a single cycle of length N
    if any(d != 2 for d in degree):
        return None
    # Extract cycle
    start = 0
    cyc = [start]
    prev = -1
    cur = start
    for _ in range(N):
        nxt = None
        for w in adj[cur]:
            if w != prev:
                nxt = w
                break
        if nxt is None:
            return None
        if nxt == start and len(cyc) < N:
            return None
        cyc.append(nxt)
        prev, cur = cur, nxt
    # cyc should have N+1 entries with last == start
    if len(cyc) != N + 1 or cyc[-1] != start:
        return None
    return cyc[:N]


def construct():
    t0 = time.time()
    deadline = t0 + 80  # safety margin for 90s limit

    # Fallback: identity permutation
    best_cycle = list(range(N))
    best_score = _cycle_score(best_cycle)

    # Greedy 2-regular construction as a quick fallback
    g = _greedy_2regular(deadline - 5)
    if g is not None:
        gs = _cycle_score(g)
        if gs < best_score:
            best_score = gs
            best_cycle = g

    # Binary search on prefix length for existence of a Hamilton cycle
    # Monotonic: if a cycle exists in prefix L, it exists in any larger prefix.
    lo = N
    hi = TOTAL_EDGES
    found_len = None
    while lo <= hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        # Quick feasibility: degree check
        adj = _build_adj(mid)
        feasible = _deg2_check(adj)
        if not feasible:
            lo = mid + 1
            continue
        # Try to actually find a cycle with a short time budget
        sub_deadline = min(deadline - 5, time.time() + 2.0)
        res = _find_at_prefix(mid, sub_deadline, best_score)
        if res is not None:
            found_len = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if found_len is not None:
        # We have a candidate prefix length; do a final focused search at that length
        # (the binary search may have accepted a cycle; refine with more time)
        final_deadline = deadline - 2
        res = _find_at_prefix(found_len, final_deadline, best_score)
        if res is not None:
            sc = _cycle_score(res)
            if sc < best_score:
                best_score = sc
                best_cycle = res

    # If binary search didn't beat the fallback, try a linear scan near the found region
    if time.time() < deadline - 3:
        scan_start = max(N, (found_len or best_score) - 5)
        scan_end = min(TOTAL_EDGES, (found_len or best_score) + 5)
        for L in range(scan_start, scan_end + 1):
            if time.time() > deadline - 2:
                break
            res = _find_at_prefix(L, deadline - 2, best_score)
            if res is not None:
                sc = _cycle_score(res)
                if sc < best_score:
                    best_score = sc
                    best_cycle = res

    return best_cycle
# EVOLVE-BLOCK-END
