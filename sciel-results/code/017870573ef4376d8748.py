"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 85.0
    
    # Precompute base adjacency
    adj_base = [set() for _ in range(n)]
    for u, v in base_edges:
        adj_base[u].add(v)
        adj_base[v].add(u)
        
    # Precompute all potential edges that are not in base and not self-loops
    # Also, we can pre-filter edges that would immediately create a triangle with base edges?
    # If (u,v) is added, and (u,w) and (w,v) are in base, it's a triangle.
    # But we are adding multiple edges, so we must check dynamically.
    
    all_pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj_base[u]:
                all_pairs.append((u, v))
                
    # To speed up search, we can order pairs.
    # Pairs that reduce diameter more effectively might be better.
    # But for a general search, random or index order is fine.
    # Let's use a deterministic order but maybe prioritize "long" jumps?
    # Actually, for finding ANY solution of size k, the order affects which solution is found first.
    # Since we want to minimize k, we iterate k from 1 upwards.
    
    # Helper: Check if current set of added edges forms a triangle-free graph with base
    # And if diameter is <= 4.
    
    def is_valid(added_edges):
        # Check Triangle-free
        # Build adj sets including added edges
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        for u, v in added_edges:
            # Check if adding (u,v) creates a triangle
            # Common neighbors of u and v in current adj (before adding u-v)
            # If they share a neighbor w, then u-w-v is a triangle.
            common = adj[u] & adj[v]
            if common:
                return False
            adj[u].add(v)
            adj[v].add(u)
            
        # Check Diameter <= 4
        # BFS from every node is O(N*(N+E)). N=24, E ~ 24+k. Very fast.
        # We can optimize by only checking if diameter > 4.
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= 4:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            # If any node is unreachable or dist > 4, invalid
            for d in dist:
                if d == -1 or d > 4:
                    return False
        return True

    # Backtracking search for minimum k
    # We try k=1, k=2, ...
    # For each k, we choose k edges from all_pairs.
    # To avoid permutations, we choose combinations.
    
    # Optimization: Pre-filter all_pairs?
    # If an edge (u,v) is such that dist_base(u,v) <= 2, adding it doesn't help reduce diameter much?
    # Actually, it might help if it creates shortcuts for other paths? No, triangle-free constraint is strict.
    # If dist_base(u,v) == 2, adding (u,v) creates a triangle if the intermediate node is connected to both.
    # In a path, dist(u,v)=2 implies u-w-v. Adding u-v creates triangle u-w-v.
    # So, any edge (u,v) where dist_base(u,v) == 2 is FORBIDDEN immediately because the path contains the triangle.
    # Let's verify: Path 0-1-2. Add 0-2. Triangle 0-1-2. Yes.
    # So we can remove all pairs with base distance 2 from candidates.
    
    # Compute base distances
    base_dist = [[-1]*n for _ in range(n)]
    for start in range(n):
        dist = [-1]*n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj_base[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        base_dist[start] = dist
        
    valid_candidates = []
    for u, v in all_pairs:
        if base_dist[u][v] == 2:
            continue # Would form triangle with base path
        valid_candidates.append((u, v))
        
    # Sort candidates? 
    # Maybe sort by "usefulness"? 
    # For now, keep as is.
    
    # Iterative Deepening
    max_k = 10 # We know 10 works. We start from 1.
    
    for k in range(1, max_k + 1):
        # Check time
        if time.time() > deadline:
            break
            
        # Recursively choose k edges from valid_candidates
        # We use indices to avoid object copying overhead in recursion?
        # Or just pass the list of selected edges.
        
        def backtrack(start_idx, selected):
            if time.time() > deadline:
                return None
                
            if len(selected) == k:
                if is_valid(selected):
                    return selected
                return None
                
            # Pruning: If we can't reach k edges, return
            # Number of remaining candidates
            remaining = len(valid_candidates) - start_idx
            if remaining < k - len(selected):
                return None
                
            for i in range(start_idx, len(valid_candidates)):
                edge = valid_candidates[i]
                # Optional: Check if adding this edge immediately violates triangle-free with already selected edges?
                # is_valid does full check. 
                # We can do a quick check:
                # Does edge form a triangle with any edge in 'selected'?
                # If (u,v) in selected, and (u,w) or (v,w) in base? 
                # Actually, is_valid checks the whole graph.
                # A faster local check: 
                # Let's just add and recurse. The depth is small.
                
                res = backtrack(i + 1, selected + [edge])
                if res is not None:
                    return res
            return None
        
        result = backtrack(0, [])
        if result is not None:
            return result
            
    # Fallback to known valid solution (10 edges) if search fails or times out
    # This ensures we always return a valid result
    return [
        [12, 0],
        [12, 2],
        [12, 4],
        [12, 6],
        [12, 8],
        [12, 15],
        [12, 17],
        [12, 19],
        [12, 21],
        [12, 23]
    ]

# EVOLVE-BLOCK-END
