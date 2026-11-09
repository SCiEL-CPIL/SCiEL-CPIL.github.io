"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj, n):
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
    dist = _bfs_distances(adj, n)
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
    dist = _bfs_distances(adj, n)
    for i in range(n):
        for j in range(i+1, n):
            if dist[i][j] > 4:
                return False
    return True

def construct():
    start_time = time.time()
    n = P['n']
    deadline = 80
    
    path_edges = set()
    for u, v in P['edges']:
        path_edges.add((min(u, v), max(u, v)))
    
    # Known good solution: hub 12 connected to odd vertices (11 edges)
    # Try to find something with 10 or fewer edges
    
    # Strategy: Try all single-hub solutions first (fast)
    best_solution = None
    best_count = 999
    
    even_vertices = list(range(0, n, 2))
    odd_vertices = list(range(1, n, 2))
    
    # Try each even vertex as a single hub
    for h in even_vertices:
        edges = []
        for o in odd_vertices:
            if (min(h, o), max(h, o)) not in path_edges:
                edges.append([h, o])
        if len(edges) < best_count:
            if _check_solution(edges, n):
                best_count = len(edges)
                best_solution = edges
                if best_count <= 8:
                    return [[int(u), int(v)] for u, v in best_solution]
    
    # Try two-hub solutions with smarter assignment
    for h1_idx in range(len(even_vertices)):
        if time.time() - start_time > deadline:
            break
        for h2_idx in range(h1_idx + 1, len(even_vertices)):
            if time.time() - start_time > deadline:
                break
            h1 = even_vertices[h1_idx]
            h2 = even_vertices[h2_idx]
            edges = []
            for o in odd_vertices:
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
                    if best_count <= 8:
                        return [[int(u), int(v)] for u, v in best_solution]
    
    # Try three-hub solutions
    for h1_idx in range(len(even_vertices)):
        if time.time() - start_time > deadline:
            break
        for h2_idx in range(h1_idx + 1, len(even_vertices)):
            if time.time() - start_time > deadline:
                break
            for h3_idx in range(h2_idx + 1, len(even_vertices)):
                if time.time() - start_time > deadline:
                    break
                h1 = even_vertices[h1_idx]
                h2 = even_vertices[h2_idx]
                h3 = even_vertices[h3_idx]
                edges = []
                for o in odd_vertices:
                    dists = [(abs(o - h1), h1), (abs(o - h2), h2), (abs(o - h3), h3)]
                    dists.sort()
                    h = dists[0][1]
                    if (min(h, o), max(h, o)) not in path_edges:
                        edges.append([h, o])
                if len(edges) < best_count:
                    if _check_solution(edges, n):
                        best_count = len(edges)
                        best_solution = edges
                        if best_count <= 8:
                            return [[int(u), int(v)] for u, v in best_solution]
    
    # Greedy approach: start with no edges, add most impactful edges
    candidates = []
    for e in even_vertices:
        for o in odd_vertices:
            pair = (min(e, o), max(e, o))
            if pair not in path_edges:
                candidates.append(pair)
    
    adj_base = _build_adj([], n)
    dist_base = _bfs_distances(adj_base, n)
    
    selected = []
    current_dist = [row[:] for row in dist_base]
    
    while (time.time() - start_time) < deadline:
        # Find pairs still > 4
        bad_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if current_dist[i][j] > 4:
                    bad_pairs.append((i, j))
        
        if not bad_pairs:
            break
        
        # Find candidate that fixes most bad pairs
        best_edge = None
        best_fixed = 0
        for c in candidates:
            if c in selected:
                continue
            u, v = c
            fixed = 0
            for (a, b) in bad_pairs:
                d1 = current_dist[a][u] + 1 + current_dist[v][b] if current_dist[a][u] >= 0 and current_dist[v][b] >= 0 else 999
                d2 = current_dist[a][v] + 1 + current_dist[u][b] if current_dist[a][v] >= 0 and current_dist[u][b] >= 0 else 999
                new_d = min(d1, d2, current_dist[a][b])
                if new_d <= 4:
                    fixed += 1
            if fixed > best_fixed:
                best_fixed = fixed
                best_edge = c
        
        if best_edge is None or best_fixed == 0:
            break
        
        selected.append(best_edge)
        adj = _build_adj(selected, n)
        current_dist = _bfs_distances(adj, n)
    
    if len(selected) < best_count and _check_solution(selected, n):
        best_count = len(selected)
        best_solution = selected
    
    if best_solution is not None:
        return [[int(u), int(v)] for u, v in best_solution]
    
    # Fallback: hub at 12
    hub = 12
    fallback = []
    for v in odd_vertices:
        if (min(hub, v), max(hub, v)) not in path_edges:
            fallback.append([hub, v])
    return [[int(u), int(v)] for u, v in fallback]
# EVOLVE-BLOCK-END
