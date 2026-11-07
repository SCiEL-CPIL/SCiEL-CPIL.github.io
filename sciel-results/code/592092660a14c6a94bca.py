"""Packing edge-disjoint cycles on a 21-vertex host graph."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_small_cycles(adj, used_edges, max_cycles, deadline):
    """
    Find edge-disjoint cycles greedily.
    Strategy: prefer smaller cycles (3, 4, 5, ...) to maximize count.
    For each length, enumerate candidate cycles and pick one that doesn't conflict.
    """
    cycles = []
    n = max(adj.keys()) + 1 if adj else 0
    all_vertices = list(range(n))
    
    # Precompute all edges as frozensets for quick lookup
    edge_set = set()
    for u in adj:
        for v in adj[u]:
            e = (min(u, v), max(u, v))
            edge_set.add(e)
    
    # We'll try to find cycles of increasing length
    # For each length L from 3 up to n, find all simple cycles of that length
    # that use only unused edges, and pick greedily
    
    # To be efficient, we'll use a greedy approach:
    # 1. For L=3 (triangles): find all triangles, pick one with unused edges
    # 2. For L=4: find all 4-cycles, pick one with unused edges
    # 3. Continue for larger L if needed
    # But we want MAX count, so we should prefer SMALL cycles first
    
    # Actually, to maximize count, we should greedily pick the smallest available cycles.
    # Let's do: repeatedly find the smallest cycle (by length) that can be formed with unused edges.
    
    # For efficiency, let's precompute all cycles up to some length.
    # But 21 vertices is manageable.
    
    # Alternative: iterative greedy
    # At each step, find the shortest cycle (by vertex count) that uses only unused edges.
    # If no cycle exists, stop.
    
    # To find the shortest cycle, we can do BFS/DFS from each vertex.
    # But we need to check edge availability.
    
    # Let's do: for L from 3 to n:
    #   try to find a cycle of length L using unused edges
    #   if found, add it and continue
    #   if not found, move to L+1
    # This ensures we always take the smallest possible cycle at each step.
    
    # However, this greedy might not be optimal. But it's a good heuristic.
    
    # Let's implement a function to find a cycle of given length L using only unused edges.
    
    def find_cycle_of_length(L):
        """Find a simple cycle of exactly L vertices using only unused edges."""
        if L < 3:
            return None
        
        # We need to find L distinct vertices v0, v1, ..., v_{L-1} such that
        # (v0,v1), (v1,v2), ..., (v_{L-2},v_{L-1}), (v_{L-1},v0) are all unused edges.
        
        # Use DFS with pruning.
        # To reduce search space, fix v0 = smallest vertex that has enough unused degree.
        
        # First, build a subgraph of unused edges.
        # For each vertex, its unused neighbors.
        
        unused_adj = defaultdict(set)
        for u in all_vertices:
            for v in adj[u]:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    unused_adj[u].add(v)
        
        # Try each starting vertex
        for start in all_vertices:
            if len(unused_adj[start]) < 2:
                continue
            
            # DFS to find a path of length L-1 starting from start, ending at a neighbor of start
            # Path: start -> v1 -> v2 -> ... -> v_{L-1} where v_{L-1} is adjacent to start (unused edge)
            
            # Use iterative DFS with backtracking
            # State: (current, visited_set, path)
            
            # To speed up, use heuristic: at each step, prefer vertices with fewer unused neighbors
            # that are still in the remaining candidate set.
            
            def dfs(current, visited, path):
                if len(path) == L:
                    # Check if we can close: last vertex must be adjacent to start via unused edge
                    last = path[-1]
                    if start in unused_adj[last]:
                        return path
                    return None
                
                # Get candidates: unused neighbors of current not in visited
                candidates = []
                for nb in unused_adj[current]:
                    if nb not in visited:
                        candidates.append(nb)
                
                if not candidates:
                    return None
                
                # Sort by unused degree (Warnsdorff-like)
                candidates.sort(key=lambda x: len(unused_adj[x]))
                
                for nb in candidates:
                    visited.add(nb)
                    path.append(nb)
                    
                    result = dfs(nb, visited, path)
                    if result is None:
                        path.pop()
                        visited.discard(nb)
                    else:
                        return result
                return None
            
            visited = {start}
            path = [start]
            result = dfs(start, visited, path)
            if result is not None:
                return result
        
        return None
    
    # Greedy: repeatedly find the smallest cycle
    for _ in range(max_cycles):
        if time.time() > deadline:
            break
        
        found = False
        for L in range(3, n + 1):
            if time.time() > deadline:
                break
            
            cycle = find_cycle_of_length(L)
            if cycle is not None:
                # Mark its edges as used
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    used_edges.add(e)
                cycles.append(cycle)
                found = True
                break
        
        if not found:
            break
    
    return cycles

def construct():
    deadline = time.time() + 85
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    
    used_edges = set()
    cycles = _find_small_cycles(adj, used_edges, 10, deadline)
    
    # Convert to proper types
    result = [list(c) for c in cycles]
    return result
# EVOLVE-BLOCK-END
