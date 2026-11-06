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
    Uses branch and bound / exact search.
    """
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
    
    # Exact branch and bound:
    # We want to find the maximum size of a subset S such that no triangle is fully contained in S.
    # This is the Maximum Independent Set in the hypergraph of triangles.
    # For n=18, we can use a recursive search with memoization or simple pruning.
    
    # Let's use a different approach: 
    # The complement of a triangle-free set is a vertex cover for the triangles.
    # So we want to find the minimum vertex cover in the hypergraph of triangles.
    # Then the answer is n - (min vertex cover size).
    
    # Minimum vertex cover in a 3-uniform hypergraph on 18 vertices.
    # We can use a branch and bound on the vertex cover.
    
    # State: (remaining_triangles, candidate_vertices)
    # We want to remove the minimum number of vertices to hit all triangles.
    
    # Let's implement a recursive search for the minimum vertex cover.
    
    def min_vertex_cover(triangles, candidates, depth=0):
        if not triangles:
            return 0
        if time.time() > deadline_global:
            return float('inf')
        
        # Lower bound: at least 1 vertex needed if there are triangles
        # Better lower bound: matching number (not easy to compute)
        # Simple bound: 1
        
        # Pick a triangle with the fewest candidates
        min_tri = None
        min_count = float('inf')
        for t in triangles:
            count = 0
            for v in candidates:
                if (t >> v) & 1:
                    count += 1
            if count < min_count:
                min_count = count
                min_tri = t
            if count == 0:
                # This triangle has no candidates, impossible
                return float('inf')
        
        if min_count == 0:
            return float('inf')
        
        # Try removing each vertex in the chosen triangle
        best = float('inf')
        t = min_tri
        for v in candidates:
            if (t >> v) & 1:
                # Remove v from candidates and remove all triangles containing v
                new_candidates = [x for x in candidates if x != v]
                new_triangles = []
                for tri in triangles:
                    if (tri >> v) & 1:
                        continue
                    new_triangles.append(tri)
                
                cost = 1 + min_vertex_cover(new_triangles, new_candidates, depth + 1)
                if cost < best:
                    best = cost
                if best == min_count:  # Can't do better than min_count
                    break
        
        return best
    
    # To speed up, we can use a heuristic initial upper bound.
    # Greedy vertex cover: repeatedly pick the vertex in the most triangles.
    def greedy_vertex_cover(triangles):
        if not triangles:
            return 0
        remaining = list(triangles)
        cover = []
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
            cover.append(max_v)
            remaining = [t for t in remaining if not ((t >> max_v) & 1)]
        return len(cover)
    
    # Get initial upper bound
    ub = greedy_vertex_cover(triangles)
    
    # Now do branch and bound with this upper bound
    # To make it faster, we can use a simpler approach for n=18.
    # Let's just iterate over all subsets in increasing size and check if they form a vertex cover.
    # A set C is a vertex cover if for every triangle t, (C & t) != 0.
    
    # Since we want the minimum vertex cover, we iterate C by size.
    # But that's still 2^18.
    
    # Let's stick with the recursive B&B but optimize it.
    # Actually, for n=18, the recursive B&B might be slow if not well-pruned.
    
    # Alternative: Use the fact that we want the max triangle-free set.
    # Let's try a different B&B:
    # Start with all vertices. If there's a triangle, we must exclude at least one vertex from it.
    # Branch on which vertex to exclude.
    
    def max_triangle_free(vertices, triangles):
        if not triangles:
            return len(vertices)
        
        # Find a triangle
        t = triangles[0]
        # Branch: for each vertex in the triangle, exclude it and recurse
        best = 0
        for v in vertices:
            if (t >> v) & 1:
                new_vertices = [x for x in vertices if x != v]
                new_triangles = [tri for tri in triangles if not ((tri >> v) & 1)]
                val = max_triangle_free(new_vertices, new_triangles)
                if val > best:
                    best = val
        return best
    
    # This might be too slow. Let's use the vertex cover approach with a better B&B.
    
    # Actually, let's just use the simple iteration over masks, but optimized with numpy or bit operations.
    # Since n=18, 2^18 = 262144.
    # We can precompute for each vertex the triangles it belongs to.
    
    # Let's try the mask iteration again, but with early termination and better pruning.
    
    # For each mask, we check if it's a valid triangle-free set.
    # A mask is valid if for all triangles t, (mask & t) != t.
    
    # We can precompute a list of triangles as integers.
    # Then for each mask, we check all triangles.
    
    # To speed up, we can precompute for each vertex v, the set of triangles containing v.
    # But that doesn't directly help.
    
    # Let's just do the mask iteration. It should be fast enough.
    
    max_size = 0
    for mask in range((1 << n) - 1, -1, -1):
        valid = True
        for t in triangles:
            if (mask & t) == t:
                valid = False
                break
        if valid:
            max_size = bin(mask).count('1')
            return max_size
    
    return 0

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
    global deadline_global
    start_time = time.time()
    deadline_global = start_time + 140.0  # Safety margin
    
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
        
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not current_adj[u][v]:
                    non_edges.append((u, v))
                    
        random.shuffle(non_edges)
        
        improved = True
        while improved and time.time() < deadline_global:
            improved = False
            for u, v in non_edges:
                if time.time() > deadline_global:
                    break
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
                    current_adj[u][v] = True
                    current_adj[v][u] = True
                    current_edges.append([u, v])
                    
                    new_score = _max_triangle_free_induced(current_adj, n)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        non_edges = []
                        for uu in range(n):
                            for vv in range(uu + 1, n):
                                if not current_adj[uu][vv]:
                                    non_edges.append((uu, vv))
                        random.shuffle(non_edges)
                        break
                    else:
                        current_adj[u][v] = False
                        current_adj[v][u] = False
                        current_edges.pop()
                        
        if current_score < best_score:
            best_edges = current_edges
            best_score = current_score
            
    if best_edges is None:
        return edges_k666
        
    return best_edges
