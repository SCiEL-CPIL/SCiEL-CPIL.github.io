"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_distances(adj, start):
    n = P['n']
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _max_diameter(adj):
    n = P['n']
    max_d = 0
    for i in range(n):
        dists = _bfs_distances(adj, i)
        if -1 in dists:
            return float('inf')
        max_d = max(max_d, max(dists))
    return max_d

def _is_triangle_free(adj):
    n = P['n']
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return False
    return True

def _check_solution(added_edges):
    all_edges = list(P['edges']) + list(added_edges)
    adj = _build_adj(all_edges)
    if not _is_triangle_free(adj):
        return False
    diam = _max_diameter(adj)
    return diam <= 4

def construct():
    deadline = time.time() + 85
    n = P['n']
    
    # Build base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # Generate candidate edges: not in base, and no common neighbor in base
    base_edge_set = set((min(u,v), max(u,v)) for u, v in P['edges'])
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edge_set:
                continue
            if base_adj[u] & base_adj[v]:
                continue
            candidates.append((u, v))
    
    # Sort by span (larger span first) as heuristic
    candidates.sort(key=lambda e: e[1] - e[0], reverse=True)
    
    # Try to find minimal solution using iterative deepening
    # Start with small number of edges and increase
    
    # First, try a constructive approach
    # The path is 0-1-2-...-23. We need diameter <= 4.
    # Key pairs to connect: (0,23), (0,22), etc.
    
    # Let's try adding edges that create a "grid" or "ladder" structure
    # A known approach: add edges (i, i+6) for appropriate i to create shortcuts
    
    # Try specific patterns
    test_patterns = [
        # Pattern 1: connect every 6th vertex
        [[0, 6], [6, 12], [12, 18], [18, 23]],
        # Pattern 2: more connections
        [[0, 6], [6, 12], [12, 18], [18, 23], [0, 12], [6, 18]],
        # Pattern 3: even more
        [[0, 6], [6, 12], [12, 18], [18, 23], [0, 12], [6, 18], [0, 18], [6, 23]],
    ]
    
    best_solution = None
    best_size = float('inf')
    
    for pattern in test_patterns:
        if time.time() > deadline:
            break
        if _check_solution(pattern):
            if len(pattern) < best_size:
                best_size = len(pattern)
                best_solution = [list(e) for e in pattern]
    
    # If no pattern works, use greedy
    if best_solution is None:
        added = []
        edges_so_far = list(P['edges'])
        
        for step in range(15):
            if time.time() > deadline:
                break
            current_adj = _build_adj(edges_so_far)
            diam = _max_diameter(current_adj)
            if diam <= 4:
                break
            
            # Find pair with max distance
            max_dist = 0
            max_pair = None
            for i in range(n):
                dists = _bfs_distances(current_adj, i)
                if -1 in dists:
                    continue
                for j in range(i + 1, n):
                    if dists[j] > max_dist:
                        max_dist = dists[j]
                        max_pair = (i, j)
            
            if max_pair is None:
                break
            
            u, v = max_pair
            # Find best candidate edge to add
            best_candidate = None
            best_span = -1
            
            for cu, cv in candidates:
                if (cu, cv) in [(a, b) for a, b in added]:
                    continue
                # Check triangle-free
                new_edges = edges_so_far + [[cu, cv]]
                new_adj = _build_adj(new_edges)
                if not _is_triangle_free(new_adj):
                    continue
                span = cv - cu
                if span > best_span:
                    best_span = span
                    best_candidate = (cu, cv)
            
            if best_candidate is None:
                break
            
            added.append(list(best_candidate))
            edges_so_far.append(list(best_candidate))
        
        if _check_solution(added):
            if len(added) < best_size:
                best_size = len(added)
                best_solution = [list(e) for e in added]
    
    # Try to minimize by removing edges
    if best_solution is not None:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_solution)):
                trial = best_solution[:i] + best_solution[i+1:]
                if _check_solution(trial):
                    best_solution = trial
                    improved = True
                    break
    
    if best_solution is None:
        return []
    
    return [[int(u), int(v)] for u, v in best_solution]

# EVOLVE-BLOCK-END
