"""Exact minimum-edge search for triangle-free diameter-4 augmentation."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _bfs(adj, n):
    """BFS from all sources, return dist matrix or None if disconnected."""
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[s][u] + 1
            for v in adj[u]:
                if dist[s][v] == -1:
                    dist[s][v] = du
                    q.append(v)
        for j in range(n):
            if dist[s][j] == -1:
                return None
    return dist


def _check_diameter(adj, n):
    """Return diameter if <= 4 and connected, else -1."""
    dist = _bfs(adj, n)
    if dist is None:
        return -1
    mx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] > mx:
                mx = dist[i][j]
    return mx if mx <= 4 else -1


def _build_base():
    """Build base adjacency and set of base edges."""
    n = P['n']
    adj = [[] for _ in range(n)]
    base_set = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_set.add((u, v))
        base_set.add((v, u))
    return adj, base_set


def _get_candidates(adj, base_set, n):
    """Get all non-base edges that are triangle-free when added alone."""
    cands = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_set:
                continue
            # Check triangle-free: no common neighbor
            common = set(adj[u]) & set(adj[v])
            if not common:
                cands.append((u, v))
    return cands


def _compatible(cand, adj, n):
    """Check if adding cand to adj keeps triangle-free."""
    u, v = cand
    common = set(adj[u]) & set(adj[v])
    return len(common) == 0


def _search_k(cands, adj_base, base_set, n, k, deadline, visited):
    """
    DFS: find k candidates from cands that are mutually compatible
    and with base graph give diameter <= 4.
    Returns list of k edge tuples or None.
    """
    if time.time() > deadline:
        return None
    if k == 0:
        # Check if base graph alone has diameter <= 4 (shouldn't, but safe)
        d = _check_diameter(adj_base, n)
        return [] if d >= 0 else None
    if len(cands) < k:
        return None

    # Pruning: if we've already tried this combination
    # (we use a set of frozensets for small k, but for larger k we skip)

    for i in range(len(cands) - k + 1):
        if time.time() > deadline:
            return None
        cand = cands[i]
        # Check compatibility with base
        if not _compatible(cand, adj_base, n):
            continue

        # Add to adjacency
        u, v = cand
        adj_base[u].append(v)
        adj_base[v].append(u)

        # Recurse
        result = _search_k(cands[i + 1:], adj_base, base_set, n, k - 1, deadline, visited)
        if result is not None:
            return [cand] + result

        # Undo
        adj_base[u].remove(v)
        adj_base[v].remove(u)

    return None


def _search_k_with_tracking(cands, base_adj, n, k, deadline):
    """
    Search for k edges. Returns (edge_list, adj_after) or (None, None).
    Uses a copy of base_adj for each branch to avoid mutation issues.
    """
    if time.time() > deadline:
        return None, None
    if k == 0:
        d = _check_diameter(base_adj, n)
        if d >= 0:
            return [], base_adj
        return None, None
    if len(cands) < k:
        return None, None

    for i in range(len(cands) - k + 1):
        if time.time() > deadline:
            return None, None
        cand = cands[i]
        u, v = cand
        # Check triangle-free with current adj
        if set(base_adj[u]) & set(base_adj[v]):
            continue

        # Add edge
        base_adj[u].append(v)
        base_adj[v].append(u)

        sub_result, sub_adj = _search_k_with_tracking(cands[i + 1:], base_adj, n, k - 1, deadline)
        if sub_result is not None:
            return [cand] + sub_result, sub_adj

        # Undo
        base_adj[u].remove(v)
        base_adj[v].remove(u)

    return None, None


def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin (90s limit)
    n = P['n']

    # Build base
    base_adj, base_set = _build_base()

    # Get candidate edges (triangle-free with base alone)
    cands = _get_candidates(base_adj, base_set, n)

    # Try k = 1, 2, 3, ... until we find a valid solution
    for k in range(1, 15):
        if time.time() > deadline:
            break

        # Use a fresh copy of base_adj for each k-level search
        adj_copy = [list(lst) for lst in base_adj]

        edges, final_adj = _search_k_with_tracking(cands, adj_copy, n, k, deadline)

        if edges is not None:
            # Convert to proper format
            result = [[int(u), int(v)] for u, v in edges]
            return result

    # Fallback: if exact search didn't find within time, do greedy
    # (This should rarely be reached for k <= 5)
    adj = [list(lst) for lst in base_adj]
    added = []
    for cand in cands:
        if time.time() > deadline:
            break
        u, v = cand
        if set(adj[u]) & set(adj[v]):
            continue
        adj[u].append(v)
        adj[v].append(u)
        added.append([u, v])
        d = _check_diameter(adj, n)
        if d >= 0:
            return [[int(u), int(v)] for u, v in added]

    # Last resort: return empty (invalid, but better than timeout)
    return []

# EVOLVE-BLOCK-END
