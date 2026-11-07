import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 80
    
    # Precompute base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    def is_valid_star(c, S):
        """Check if adding star edges (c, v) for v in S gives triangle-free, connected, diameter <= 4."""
        # S must be independent set (no two vertices in S adjacent)
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if S[j] in base_adj[S[i]]:
                    return False
        
        # Build adjacency
        adj = [set(s) for s in base_adj]
        for v in S:
            adj[c].add(v)
            adj[v].add(c)
        
        # Check triangle-free: for each added edge (c, v), no common neighbor
        for v in S:
            if len(adj[c] & adj[v]) > 0:
                return False
        
        # Check diameter <= 4 using BFS from c (since it's a star, max dist from c is 2)
        # Actually need to verify all pairs, but with star structure:
        # dist(u, v) <= dist(u, c) + dist(v, c)
        # So we just need max dist from c to be <= 2
        dist = [-1] * n
        dist[c] = 0
        queue = deque([c])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > 2:
                        return False
                    queue.append(v)
        
        # Verify all vertices reached
        for d in dist:
            if d == -1:
                return False
        
        return True
    
    best_solution = None
    
    # Search for 7-edge star solution
    for c in range(n):
        if time.time() > deadline:
            break
        
        # Which vertices are already within distance 2 of c via path?
        covered = set([c])
        # Distance 1: neighbors of c
        for v in base_adj[c]:
            covered.add(v)
        # Distance 2: neighbors of neighbors
        for v in base_adj[c]:
            for w in base_adj[v]:
                covered.add(w)
        
        # Vertices NOT covered need to be in S or adjacent to S
        # A vertex v in S covers {v-1, v, v+1} (intersected with [0, n-1])
        # We need to cover all vertices not in 'covered'
        
        uncovered = set(range(n)) - covered
        
        # Candidates: vertices not adjacent to c (can't add edge to neighbor)
        # and v != c
        candidates = [v for v in range(n) if v != c and v not in base_adj[c]]
        
        # For each candidate, compute which uncovered vertices it covers
        def covers(v):
            s = set()
            for x in [v-1, v, v+1]:
                if 0 <= x < n:
                    s.add(x)
            return s & uncovered
        
        # We need to pick 7 vertices from candidates such that:
        # 1. They form an independent set
        # 2. Their coverage union covers all uncovered vertices
        
        # Filter candidates that cover at least one uncovered vertex
        useful_candidates = [(v, covers(v)) for v in candidates]
        useful_candidates = [(v, cov) for v, cov in useful_candidates if cov]
        
        # Sort by coverage size (greedy hint)
        useful_candidates.sort(key=lambda x: -len(x[1]))
        
        # If even using all useful candidates doesn't cover everything, skip
        total_cover = set()
        for _, cov in useful_candidates:
            total_cover |= cov
        if total_cover != uncovered:
            continue
        
        # Now search for independent set of size 7 that covers all uncovered
        # Use backtracking with pruning
        def backtrack(idx, chosen, covered_so_far, remaining_needed):
            nonlocal best_solution
            if time.time() > deadline:
                return False
            if len(chosen) == 7:
                if covered_so_far == uncovered:
                    best_solution = [(c, v) for v in chosen]
                    return True
                return False
            
            # Pruning: if remaining needed can't be covered by remaining candidates
            if idx >= len(useful_candidates):
                return False
            
            # Check if we can still cover all remaining
            remaining_covers = set()
            for i in range(idx, len(useful_candidates)):
                remaining_covers |= useful_candidates[i][1]
            if remaining_needed - remaining_covers:
                return False
            
            # Option 1: skip this candidate
            if backtrack(idx + 1, chosen, covered_so_far, remaining_needed):
                return True
            
            # Option 2: take this candidate
            v, cov = useful_candidates[idx]
            # Check independence with all chosen
            independent = True
            for w in chosen:
                if v in base_adj[w]:
                    independent = False
                    break
            if independent:
                new_covered = covered_so_far | cov
                if backtrack(idx + 1, chosen + [v], new_covered, remaining_needed - new_covered):
                    return True
            
            return False
        
        if backtrack(0, [], set(), uncovered):
            break
    
    if best_solution is None:
        return []
    
    return [[int(u), int(v)] for u, v in best_solution]
