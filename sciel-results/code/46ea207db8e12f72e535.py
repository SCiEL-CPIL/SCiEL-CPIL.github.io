"""Structural change: model as minimum cardinality independent dominating set on P_24,
then add edges between S-vertices at path-distance 2. Verify and optimize."""
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_diameter(n, adj):
    max_d = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return -1
        md = max(dist)
        if md > max_d:
            max_d = md
    return max_d

def _is_triangle_free(n, edges):
    adj = _build_adj(n, edges)
    colors = [-1] * n
    for i in range(n):
        if colors[i] == -1:
            colors[i] = 0
            q = [i]
            while q:
                u = q.pop(0)
                for v in adj[u]:
                    if colors[v] == -1:
                        colors[v] = 1 - colors[u]
                        q.append(v)
                    elif colors[v] == colors[u]:
                        return False
    return True

def _valid_solution(n, base_edges, added_edges):
    all_edges = base_edges + added_edges
    edge_set = set()
    for u, v in all_edges:
        a, b = min(u, v), max(u, v)
        if (a, b) in edge_set:
            return False
        edge_set.add((a, b))
    if not _is_triangle_free(n, all_edges):
        return False
    adj = _build_adj(n, all_edges)
    diam = _bfs_diameter(n, adj)
    return diam != -1 and diam <= 4

def construct():
    n = P['n']
    base_edges = P['edges']
    base_edge_set = set((min(u, v), max(u, v)) for u, v in base_edges)
    
    deadline = time.time() + 80
    
    # Key insight: For triangle-free graph, added edges must connect opposite parity vertices
    # (since path edges connect same parity... wait, no: path 0-1 connects 0(even) to 1(odd))
    # Actually, path edges connect consecutive vertices which have opposite parity.
    # For triangle-free: if we add edge (u,v), we cannot have a common neighbor.
    # In a bipartite graph (which path is), adding edges between opposite partitions keeps it bipartite
    # and thus triangle-free. But adding edges within same partition creates odd cycles.
    # Path P_24 is bipartite: even vertices {0,2,...,22} and odd vertices {1,3,...,23}
    # Path edges go between even and odd.
    # So added edges must also go between even and odd to keep bipartite (and thus triangle-free).
    
    # Strategy: Try to find minimum number of edges between even and odd vertices
    # such that diameter <= 4.
    
    # Let's enumerate candidate edges (even, odd) pairs and search for small solutions.
    
    even_verts = list(range(0, n, 2))  # 0, 2, 4, ..., 22
    odd_verts = list(range(1, n, 2))   # 1, 3, 5, ..., 23
    
    # Generate all valid candidate edges (between even and odd, not in base)
    candidates = []
    for u in even_verts:
        for v in odd_verts:
            if (min(u,v), max(u,v)) not in base_edge_set:
                candidates.append((u, v))
    
    # Try 3-edge solutions first
    # We need diameter <= 4, so the added edges must create good shortcuts.
    
    # Heuristic: good edges are those that jump far.
    # Let's score candidates by how much they help reduce diameter.
    
    # For a 3-edge solution to work on P_24, we need the three edges to partition
    # the path into segments of length at most 4 (in terms of path distance between hubs).
    
    # Let's try systematic search for 3-edge solutions.
    # To make it tractable, restrict to "long" edges.
    
    # An edge (u,v) with |u-v| = d creates a shortcut of length 1 instead of d.
    # For diameter 4, we need every pair within distance 4.
    
    # Let's try all combinations of 3 from a curated set of promising edges.
    
    # Promising edges: those with large span
    curated = []
    for u in even_verts:
        for v in odd_verts:
            dist = abs(u - v)
            if dist >= 5:  # at least 5 apart
                if (min(u,v), max(u,v)) not in base_edge_set:
                    curated.append((u, v))
    
    # Sort by span descending
    curated.sort(key=lambda e: abs(e[0]-e[1]), reverse=True)
    
    # Limit to top candidates for combinatorial search
    top_candidates = curated[:50]
    
    best_sol = None
    
    # Try 3-edge combinations
    if time.time() < deadline - 5:
        for combo in combinations(top_candidates, 3):
            if time.time() > deadline:
                break
            sol = list(combo)
            sol_set = set((min(u, v), max(u, v)) for u, v in sol)
            if len(sol_set) != 3:
                continue
            if _valid_solution(n, base_edges, sol):
                best_sol = sol
                break
    
    if best_sol is not None:
        return [[u, v] for u, v in best_sol]
    
    # Try 4-edge combinations
    if time.time() < deadline - 10:
        for combo in combinations(top_candidates, 4):
            if time.time() > deadline:
                break
            sol = list(combo)
            sol_set = set((min(u, v), max(u, v)) for u, v in sol)
            if len(sol_set) != 4:
                continue
            if _valid_solution(n, base_edges, sol):
                best_sol = sol
                break
    
    if best_sol is not None:
        return [[u, v] for u, v in best_sol]
    
    # Fallback: greedy construction
    added = []
    current_edges = list(base_edges)
    added_set = set()
    
    while time.time() < deadline - 5:
        adj = _build_adj(n, current_edges)
        diam = _bfs_diameter(n, adj)
        
        if diam <= 4:
            break
        if diam == -1:
            break
        
        # Find worst pair
        worst_u, worst_v = -1, -1
        max_dist_found = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                        if dist[v] > max_dist_found:
                            max_dist_found = dist[v]
                            worst_u = s
                            worst_v = v
        
        if worst_u == -1:
            break
        
        # Find best candidate edge
        best_candidate = None
        best_diam = max_dist_found + 1
        
        for a, b in candidates:
            if (min(a, b), max(a, b)) in added_set:
                continue
            adj[a].add(b)
            adj[b].add(a)
            new_diam = _bfs_diameter(n, adj)
            adj[a].remove(b)
            adj[b].remove(a)
            
            if new_diam != -1 and new_diam < best_diam:
                best_diam = new_diam
                best_candidate = (a, b)
        
        if best_candidate is None:
            break
        
        added.append(best_candidate)
        added_set.add((min(best_candidate[0], best_candidate[1]), max(best_candidate[0], best_candidate[1])))
        current_edges.append(best_candidate)
    
    if _valid_solution(n, base_edges, added):
        if best_sol is None or len(added) < len(best_sol):
            best_sol = added
    
    # Try to reduce
    if best_sol is not None:
        for i in range(len(best_sol) - 1, -1, -1):
            if time.time() > deadline:
                break
            reduced = best_sol[:i] + best_sol[i+1:]
            if _valid_solution(n, base_edges, reduced):
                best_sol = reduced
    
    if best_sol is not None:
        return [[u, v] for u, v in best_sol]
    
    return []
