"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    """
    Constructs a graph on 12 vertices (0..11) with minimum degree 3,
    triangle-free (girth >= 4), and minimizes distinct simple cycle lengths.
    
    Strategy: Enumerate all 3-regular bipartite graphs on 12 vertices (6+6 split)
    and find one with the minimum number of distinct cycle lengths.
    Prefer girth 8 (only 8-cycles) if it exists, otherwise girth 4 (K_{3,3} copies).
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    n = 12
    min_deg = 3
    min_girth = 4
    
    best_score = float('inf')
    best_edges = None
    
    def _build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def _has_triangle(adj, n):
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                for k in adj[j]:
                    if k > j and k in adj[i]:
                        return True
        return False
    
    def _min_degree_ok(adj, n, min_deg):
        return all(len(adj[i]) >= min_deg for i in range(n))
    
    def _girth_ok(adj, n, min_girth):
        """Check if all cycles have length >= min_girth using BFS from each vertex."""
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for node in queue:
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    elif neighbor != parent[node] and parent[neighbor] != node:
                        cycle_len = dist[node] + dist[neighbor] + 1
                        if cycle_len < min_girth:
                            return False
        return True
    
    def _has_cycle(adj, n):
        """Check if graph has at least one cycle."""
        visited = [False] * n
        for start in range(n):
            if not visited[start]:
                stack = [(start, -1)]
                while stack:
                    node, par = stack.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    for nb in adj[node]:
                        if nb == par:
                            continue
                        if visited[nb]:
                            return True
                        stack.append((nb, node))
        return False
    
    def _count_distinct_cycle_lengths(adj, n):
        """Count distinct simple cycle lengths using DFS from smallest vertex in cycle."""
        cycle_lengths = set()
        for start in range(n):
            visited = [False] * n
            visited[start] = True
            path = [start]
            
            def dfs(current):
                # Try to close cycle
                if len(path) >= 3 and start in adj[current]:
                    cycle_lengths.add(len(path))
                for next_node in adj[current]:
                    if next_node == start:
                        continue
                    if next_node > start and not visited[next_node]:
                        visited[next_node] = True
                        path.append(next_node)
                        dfs(next_node)
                        path.pop()
                        visited[next_node] = False
            
            dfs(start)
        return len(cycle_lengths) if cycle_lengths else 0
    
    def _evaluate(edges, n=12, min_deg=3, min_girth=4):
        """Returns (valid, num_distinct_cycle_lengths)."""
        edge_set = set()
        for u, v in edges:
            if u == v or u >= n or v >= n:
                return (False, 0)
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return (False, 0)
            edge_set.add(key)
        
        adj = _build_adj([(u, v) for u, v in edge_set], n)
        
        if not _min_degree_ok(adj, n, min_deg):
            return (False, 0)
        if _has_triangle(adj, n):
            return (False, 0)
        if not _girth_ok(adj, n, min_girth):
            return (False, 0)
        if not _has_cycle(adj, n):
            return (False, 0)
        
        ncl = _count_distinct_cycle_lengths(adj, n)
        return (True, ncl)
    
    def _make_edges(edge_set):
        return [list(e) for e in sorted(edge_set)]
    
    # Strategy 1: Try known structured constructions
    constructions = []
    
    # Prism graph (current seed)
    es = set()
    for i in range(6):
        for u, v in [(i, (i+1)%6), (i+6, (i+1)%6+6), (i, i+6)]:
            es.add(tuple(sorted((u, v))))
    constructions.append(es)
    
    # Two copies of K_{3,3} on 12 vertices
    es2 = set()
    for i in range(3):
        for j in range(3, 6):
            es2.add((i, j))
    for i in range(6, 9):
        for j in range(9, 12):
            es2.add((i, j))
    constructions.append(es2)
    
    # Try twisted prism
    es3 = set()
    for i in range(6):
        es3.add((i, (i+1)%6))
        es3.add((i+6, (i+1)%6+6))
    cross = [(0,6),(1,8),(2,10),(3,11),(4,7),(5,9)]
    for u,v in cross:
        es3.add((u,v))
    constructions.append(es3)
    
    # Evaluate all constructions
    for es in constructions:
        edges = _make_edges(es)
        valid, ncl = _evaluate(edges)
        if valid and ncl < best_score:
            best_score = ncl
            best_edges = edges[:]
    
    # Strategy 2: Enumerate 3-regular bipartite graphs on 12 vertices (6+6)
    # Left: 0..5, Right: 6..11
    # Each left vertex connects to exactly 3 right vertices
    # Each right vertex connects to exactly 3 left vertices
    
    col_sums = [0] * 6
    right_vertices = [6, 7, 8, 9, 10, 11]
    
    def _enumerate_bipartite(row_idx, current_matrix):
        nonlocal best_score, best_edges
        
        if time.time() - start_time > DEADLINE:
            return
        
        if row_idx == 6:
            # Check if all col sums are 3
            if all(cs == 3 for cs in col_sums):
                # Build edge set
                es = set()
                for r in range(6):
                    for c in range(6):
                        if current_matrix[r][c] == 1:
                            es.add((r, right_vertices[c]))
                edges = _make_edges(es)
                valid, ncl = _evaluate(edges)
                if valid and ncl < best_score:
                    best_score = ncl
                    best_edges = edges[:]
            return
        
        # Try all combinations of 3 right vertices for this row
        for combo in combinations(range(6), 3):
            # Check if adding this row would exceed col sum 3
            valid_row = True
            for c in combo:
                if col_sums[c] >= 3:
                    valid_row = False
                    break
            if not valid_row:
                continue
            
            # Add this row
            row = [0] * 6
            for c in combo:
                row[c] = 1
                col_sums[c] += 1
            
            current_matrix.append(row)
            _enumerate_bipartite(row_idx + 1, current_matrix)
            current_matrix.pop()
            
            # Undo
            for c in combo:
                col_sums[c] -= 1
    
    # Start enumeration
    _enumerate_bipartite(0, [])
    
    # If we found a better solution, return it
    if best_edges is not None:
        return best_edges
    
    # Fallback: return the best from constructions
    if best_edges is None:
        # Return two K_{3,3} copies as default
        es2 = set()
        for i in range(3):
            for j in range(3, 6):
                es2.add((i, j))
        for i in range(6, 9):
            for j in range(9, 12):
                es2.add((i, j))
        return _make_edges(es2)
    
    return best_edges

# EVOLVE-BLOCK-END
