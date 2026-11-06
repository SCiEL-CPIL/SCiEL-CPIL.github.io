import time
import random
import numpy as np
from itertools import combinations

def _build_adj(edges, n):
    """Build adjacency matrix from edge list."""
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _is_k4_free(adj, n):
    """Check if graph (given by adjacency matrix) is K4-free."""
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                continue
            for k in range(j + 1, n):
                if not adj[i][k] or not adj[j][k]:
                    continue
                for l in range(k + 1, n):
                    if adj[i][l] and adj[j][l] and adj[k][l]:
                        return False
    return True

def _max_triangle_free_induced(adj, n):
    """
    Find the size of the largest triangle-free induced subgraph.
    Uses exact branch and bound on minimum vertex cover of triangles.
    """
    # Find all triangles
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                continue
            for k in range(j + 1, n):
                if adj[i][k] and adj[j][k]:
                    triangles.append((1 << i) | (1 << j) | (1 << k))
    
    if not triangles:
        return n
    
    # Convert to set for faster lookup
    tri_set = set(triangles)
    
    # Precompute for each vertex which triangles it's in
    vert_tris = [[] for _ in range(n)]
    for t in triangles:
        for v in range(n):
            if (t >> v) & 1:
                vert_tris[v].append(t)
    
    # Greedy vertex cover for initial upper bound
    def greedy_vertex_cover(triangles):
        if not triangles:
            return 0
        remaining = list(triangles)
        cover = 0
        while remaining:
            # Count occurrences of each vertex
            counts = [0] * n
            for t in remaining:
                for v in range(n):
                    if (t >> v) & 1:
                        counts[v] += 1
            # Pick vertex with max count
            max_v = max(range(n), key=lambda x: counts[x])
            if counts[max_v] == 0:
                break
            cover += 1
            remaining = [t for t in remaining if not ((t >> max_v) & 1)]
        return cover
    
    ub = greedy_vertex_cover(triangles)
    
    # Branch and bound for minimum vertex cover
    # State: (frozenset of remaining triangles, frozenset of candidate vertices, depth)
    # We want to find minimum number of vertices to remove to hit all triangles
    
    best_cover = [ub]
    deadline_global = time.time() + 120.0
    
    def min_vertex_cover(remaining_tris, candidates, depth):
        if not remaining_tris:
            return 0
        if time.time() > deadline_global:
            return float('inf')
        
        # Lower bound: at least 1 if there are triangles
        # Better: use a simple matching-based bound
        # For now, use 1 as lower bound
        
        # Pick triangle with fewest candidate vertices
        min_tri = None
        min_count = float('inf')
        for t in remaining_tris:
            count = 0
            for v in candidates:
                if (t >> v) & 1:
                    count += 1
            if count < min_count:
                min_count = count
                min_tri = t
            if count == 0:
                return float('inf')
        
        if min_count == 0:
            return float('inf')
        
        # Try removing each vertex in the chosen triangle
        best = float('inf')
        t = min_tri
        for v in candidates:
            if (t >> v) & 1:
                new_candidates = [x for x in candidates if x != v]
                new_tris = []
                for tri in remaining_tris:
                    if (tri >> v) & 1:
                        continue
                    new_tris.append(tri)
                
                cost = 1 + min_vertex_cover(new_tris, new_candidates, depth + 1)
                if cost < best:
                    best = cost
                if best <= 1:  # Can't do better than 1
                    break
        
        return best
    
    # Initial candidates are all vertices
    candidates = list(range(n))
    result = min_vertex_cover(triangles, candidates, 0)
    
    if result == float('inf'):
        # Fallback: use greedy
        result = ub
    
    return n - result

def _get_paley_edges(q):
    """Get edges for Paley graph of order q."""
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
    
    edges = []
    for u in range(q):
        for v in range(u + 1, q):
            diff = (v - u) % q
            if diff in residues:
                edges.append([u, v])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 18
    
    # Candidate 1: Paley graph P(17) + isolated vertex
    edges_17 = _get_paley_edges(17)
    adj_17 = _build_adj(edges_17, n)
    if _is_k4_free(adj_17, n):
        score_17 = _max_triangle_free_induced(adj_17, n)
        best_edges = edges_17
        best_score = score_17
    else:
        best_edges = None
        best_score = float('inf')
        
    # Candidate 2: Two disjoint Paley graphs P(9)
    edges_9_1 = _get_paley_edges(9)
    edges_9_2 = [[u + 9, v + 9] for u, v in edges_9_1]
    edges_18 = edges_9_1 + edges_9_2
    adj_18 = _build_adj(edges_18, n)
    if _is_k4_free(adj_18, n):
        score_18 = _max_triangle_free_induced(adj_18, n)
        if score_18 < best_score:
            best_edges = edges_18
            best_score = score_18
            
    # Candidate 3: K6,6,6 (Baseline)
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    edges_k666 = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    adj_k666 = _build_adj(edges_k666, n)
    if _is_k4_free(adj_k666, n):
        score_k666 = _max_triangle_free_induced(adj_k666, n)
        if score_k666 < best_score:
            best_edges = edges_k666
            best_score = score_k666
    
    # Candidate 4: Randomized Local Search
    if best_edges is not None:
        current_edges = best_edges[:]
        current_adj = _build_adj(current_edges, n)
        current_score = best_score
        
        # Build non-edges list
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not current_adj[u][v]:
                    non_edges.append((u, v))
                    
        random.shuffle(non_edges)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                # Check if adding edge (u,v) creates a K4
                # A K4 would be formed if there exist two vertices w1, w2 that are both
                # adjacent to u and v, and w1-w2 is an edge
                common = []
                for w in range(n):
                    if w != u and w != v and current_adj[u][w] and current_adj[v][w]:
                        common.append(w)
                        
                creates_k4 = False
                for i in range(len(common)):
                    for j in range(i + 1, len(common)):
                        if current_adj[common[i]][common[j]]:
                            creates_k4 = True
                            break
                    if creates_k4:
                        break
                            
                if not creates_k4:
                    # Try adding this edge
                    current_adj[u][v] = True
                    current_adj[v][u] = True
                    current_edges.append([u, v])
                    
                    new_score = _max_triangle_free_induced(current_adj, n)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        # Rebuild non_edges
                        non_edges = []
                        for uu in range(n):
                            for vv in range(uu + 1, n):
                                if not current_adj[uu][vv]:
                                    non_edges.append((uu, vv))
                        random.shuffle(non_edges)
                        break
                    else:
                        # Revert
                        current_adj[u][v] = False
                        current_adj[v][u] = False
                        current_edges.pop()
                        
        if current_score < best_score:
            best_edges = current_edges
            best_score = current_score
            
    # Candidate 5: Try removing edges from best solution to see if we can improve
    if best_edges is not None:
        current_edges = best_edges[:]
        current_adj = _build_adj(current_edges, n)
        current_score = best_score
        
        # Try removing edges one at a time
        for idx in range(len(current_edges) - 1, -1, -1):
            if time.time() > deadline:
                break
            u, v = current_edges[idx]
            # Temporarily remove
            current_adj[u][v] = False
            current_adj[v][u] = False
            current_edges.pop()
            
            new_score = _max_triangle_free_induced(current_adj, n)
            if new_score < current_score:
                current_score = new_score
            else:
                # Revert
                current_edges.append([u, v])
                current_adj[u][v] = True
                current_adj[v][u] = True
        
        if current_score < best_score:
            best_edges = current_edges
            best_score = current_score
    
    if best_edges is None:
        return edges_k666
        
    return best_edges
