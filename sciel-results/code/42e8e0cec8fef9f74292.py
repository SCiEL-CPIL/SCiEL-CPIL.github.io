"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _bfs_all(adj, n):
    """BFS from all vertices, return max distance and whether connected."""
    max_dist = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return -1, False
        md = max(dist)
        if md > max_dist:
            max_dist = md
    return max_dist, True


def _is_valid(added_edges, base_edges, n):
    """Check all constraints: no dup, triangle-free (bipartite), connected, diameter<=4."""
    # Check no duplicate/existing edges
    existing = set()
    for u, v in base_edges:
        existing.add(frozenset((u, v)))
    for u, v in added_edges:
        fs = frozenset((u, v))
        if fs in existing:
            return False
        existing.add(fs)
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check bipartite (triangle-free)
    color = [-1] * n
    for start in range(n):
        if color[start] == -1:
            color[start] = 0
            stack = [start]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        stack.append(v)
                    elif color[v] == color[u]:
                        return False
    
    # Check diameter and connectivity
    max_d, connected = _bfs_all(adj, n)
    if not connected or max_d > 4:
        return False
    
    return True


def _build_adj(added, base_edges, n):
    adj = [[] for _ in range(n)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _max_dist_fast(adj, n):
    max_d = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return 999
        md = max(dist)
        if md > max_d:
            max_d = md
    return max_d


def construct():
    deadline = time.time() + 80
    n = P['n']
    base_edges = P['edges']
    base_set = set(frozenset((u, v)) for u, v in base_edges)
    
    # Generate candidate edges (opposite parity, not in base)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if u % 2 != v % 2 and frozenset((u, v)) not in base_set:
                candidates.append((u, v))
    
    # Structured initial solution: create a "ladder" pattern
    # For diameter 4, we need shortcuts. Use a pattern where each vertex
    # connects to vertices ~4-8 steps away in opposite parity.
    added = []
    added_set = set()
    
    # Strategy: connect vertices in a way that creates multiple paths
    # Each vertex i should be within 4 hops of every other vertex.
    # Use edges that span roughly n/4 = 6 positions
    
    for i in range(n):
        # Add edge to vertex at distance ~6 (opposite parity)
        for dist in [5, 6, 7]:
            j = i + dist
            if j < n and i % 2 != j % 2:
                fs = frozenset((i, j))
                if fs not in base_set and fs not in added_set:
                    added.append([i, j])
                    added_set.add(fs)
    
    # Verify and optimize
    if not _is_valid(added, base_edges, n):
        # Fallback: denser connection
        added = []
        added_set = set()
        for i in range(n):
            for dist in [4, 5, 6, 7, 8]:
                j = i + dist
                if j < n and i % 2 != j % 2:
                    fs = frozenset((i, j))
                    if fs not in base_set and fs not in added_set:
                        added.append([i, j])
                        added_set.add(fs)
    
    if not _is_valid(added, base_edges, n):
        # Last resort
        added = [list(e) for e in candidates]
    
    # Local search: remove edges while valid
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(len(added)):
            if time.time() > deadline - 10:
                break
            candidate = added[:i] + added[i+1:]
            if _is_valid(candidate, base_edges, n):
                added = candidate
                improved = True
                break
    
    # Try 1-for-1 swaps to reduce count
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(added)):
                if time.time() > deadline - 5:
                    break
                for edge in candidates:
                    if edge in added_set:
                        continue
                    candidate = added[:i] + added[i+1:] + [list(edge)]
                    if _is_valid(candidate, base_edges, n):
                        # Try to remove another edge
                        for j in range(len(candidate)):
                            test = candidate[:j] + candidate[j+1:]
                            if _is_valid(test, base_edges, n):
                                added = test
                                # Update added_set
                                added_set = set(frozenset(e) for e in added)
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
    
    # Final removal pass
    improved2 = True
    while improved2 and time.time() < deadline - 3:
        improved2 = False
        for i in range(len(added)):
            candidate = added[:i] + added[i+1:]
            if _is_valid(candidate, base_edges, n):
                added = candidate
                improved2 = True
                break
    
    # Ensure proper types
    result = []
    for e in added:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
