"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_all(adj, n):
    """Compute all-pairs shortest paths via BFS from each vertex."""
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[s][v] == -1:
                    dist[s][v] = dist[s][u] + 1
                    queue.append(v)
    return dist

def _check_solution(added_edges, n):
    """Check triangle-free, connected, diameter <= 4."""
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check triangle-free
    for u in range(n):
        for i, v in enumerate(adj[u]):
            for w in adj[u][i+1:]:
                if w in adj[v]:
                    return False
    
    # Check connected and diameter
    dist = _bfs_all(adj, n)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1 or dist[i][j] > 4:
                return False
    return True

def _build_adj(added_edges, n):
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _diameter_ok(added_edges, n):
    """Check if diameter <= 4 for the given added edges."""
    adj = _build_adj(added_edges, n)
    dist = _bfs_all(adj, n)
    for i in range(n):
        for j in range(i+1, n):
            if dist[i][j] > 4:
                return False
    return True

def _is_triangle_free(added_edges, n):
    """Check if the graph with path + added_edges is triangle-free."""
    adj = set()
    for u, v in P['edges']:
        adj.add((min(u,v), max(u,v)))
    for u, v in added_edges:
        adj.add((min(u,v), max(u,v)))
    
    # Check all triples
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in adj:
                continue
            for w in range(v+1, n):
                if (u, w) in adj and (v, w) in adj:
                    return False
    return True

def _try_reduce(edges, n, time_limit):
    """Try to reduce the number of edges by removing one at a time."""
    edges = list(edges)
    improved = True
    start = time.time()
    while improved and (time.time() - start) < time_limit:
        improved = False
        for i in range(len(edges)):
            candidate = edges[:i] + edges[i+1:]
            if _check_solution(candidate, n):
                edges = candidate
                improved = True
                break
    return edges

def construct():
    start_time = time.time()
    n = P['n']
    deadline = 80  # safety margin before 90s kill
    
    # Path edges
    path_edges = set()
    for u, v in P['edges']:
        path_edges.add((min(u, v), max(u, v)))
    
    # All candidate edges: even-odd pairs not in path
    candidates = []
    for e in range(0, n, 2):
        for o in range(1, n, 2):
            pair = (min(e, o), max(e, o))
            if pair not in path_edges:
                candidates.append(pair)
    
    # Known good solution: 9 edges
    # Hubs: 10, 14
    # 10: 1, 3, 5, 7
    # 14: 17, 19, 21, 23
    # Bridge: (10, 13)
    known_solution = [[10, 1], [10, 3], [10, 5], [10, 7], 
                       [14, 17], [14, 19], [14, 21], [14, 23],
                       [10, 13]]
    
    # Verify known solution
    if _check_solution(known_solution, n):
        # Try to reduce
        reduced = _try_reduce(known_solution, n, deadline - (time.time() - start_time))
        if _check_solution(reduced, n):
            return [[int(u), int(v)] for u, v in reduced]
        else:
            return [[int(u), int(v)] for u, v in known_solution]
    
    # If known solution doesn't work, fall back to search
    best_solution = None
    best_count = 999
    
    # Try two hubs
    even_vertices = list(range(0, n, 2))
    for h1_idx in range(len(even_vertices)):
        for h2_idx in range(h1_idx + 1, len(even_vertices)):
            h1 = even_vertices[h1_idx]
            h2 = even_vertices[h2_idx]
            edges = []
            for o in range(1, n, 2):
                d1 = abs(o - h1)
                d2 = abs(o - h2)
                if d1 <= d2:
                    if (min(h1, o), max(h1, o)) not in path_edges:
                        edges.append([h1, o])
                else:
                    if (min(h2, o), max(h2, o)) not in path_edges:
                        edges.append([h2, o])
            if len(edges) < best_count:
                if _check_solution(edges, n):
                    best_count = len(edges)
                    best_solution = edges
                    if best_count <= 5:
                        return [[int(u), int(v)] for u, v in best_solution]
            if time.time() - start_time > deadline:
                break
        if time.time() - start_time > deadline:
            break
    
    if best_solution is not None:
        reduced = _try_reduce(best_solution, n, deadline - (time.time() - start_time))
        if _check_solution(reduced, n):
            return [[int(u), int(v)] for u, v in reduced]
        else:
            return [[int(u), int(v)] for u, v in best_solution]
    
    # Fallback: greedy set cover
    adj = _build_adj([], n)
    dist = _bfs_all(adj, n)
    bad = set()
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] > 4:
                bad.add((i, j))
    
    selected = []
    remaining_bad = set(bad)
    
    while remaining_bad and (time.time() - start_time) < deadline:
        best_edge = None
        best_fixed_count = 0
        for e in candidates:
            if e in selected:
                continue
            u, v = e
            fixed_count = 0
            for (a, b) in remaining_bad:
                d1 = dist[a][u] + 1 + dist[v][b] if dist[a][u] >= 0 and dist[v][b] >= 0 else 999
                d2 = dist[a][v] + 1 + dist[u][b] if dist[a][v] >= 0 and dist[u][b] >= 0 else 999
                if min(d1, d2) <= 4:
                    fixed_count += 1
            if fixed_count > best_fixed_count:
                best_fixed_count = fixed_count
                best_edge = e
        if best_edge is None or best_fixed_count == 0:
            break
        selected.append(best_edge)
        adj = _build_adj(selected, n)
        dist = _bfs_all(adj, n)
        remaining_bad = set()
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i][j] > 4:
                    remaining_bad.add((i, j))
    
    if _check_solution(selected, n):
        reduced = _try_reduce(selected, n, deadline - (time.time() - start_time))
        if _check_solution(reduced, n):
            return [[int(u), int(v)] for u, v in reduced]
        else:
            return [[int(u), int(v)] for u, v in selected]
    
    # Last resort: hub at 12
    hub = 12
    fallback = []
    for v in range(1, 24, 2):
        if v != 11 and v != 13:
            fallback.append([hub, v])
    return [[int(u), int(v)] for u, v in fallback]
# EVOLVE-BLOCK-END
