# EVOLVE-BLOCK-START
import time
import sys
from collections import deque
import itertools
import random
import math

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin (90s limit, keep 10s buffer)
    
    n = P['n']
    
    # Build base adjacency
    base_adj = [set() for _ in range(n)]
    base_edge_set = set()
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)
        base_edge_set.add(frozenset((u, v)))
    
    # Generate all candidate edges (not in base, and |u-v| != 2 to avoid immediate triangle with path)
    # Note: For a path, adding edge (u, v) creates a triangle if and only if there is a w such that
    # (u,w) and (w,v) are in the graph. Since the base is a path, the only neighbors of u are u-1, u+1.
    # So (u,v) creates a triangle if v is adjacent to u-1 or u+1.
    # i.e., v == u-2 or v == u+2.
    # So we can exclude all pairs with distance 2.
    # Are there other triangles?
    # If we add (u,v) and (x,y), do they form a triangle?
    # A triangle is (a,b,c) where all 3 pairs are edges.
    # The base graph is a path (triangle-free).
    # Any triangle in the augmented graph must use at least 2 added edges, or 1 added edge and 2 base edges.
    # Case 1: 1 added edge (u,v) and 2 base edges.
    # This implies u and v are distance 2 in the base graph (via a common neighbor).
    # So excluding dist 2 edges prevents this.
    # Case 2: 2 added edges (u,v) and (x,y) and 1 base edge (z,w).
    # This requires a triangle like (u, v, w) where (u,v) is added, (v,w) is base, (w,u) is added? No, (w,u) is added means we added (u,w).
    # So we need 3 vertices a,b,c such that (a,b), (b,c), (c,a) are all edges.
    # If (a,b) is added, (b,c) is base, (c,a) is added.
    # This means a and c are both connected to b via added edges? No, (b,c) is base.
    # So a is connected to b (added) and c (added). And b is connected to c (base).
    # This forms a triangle.
    # Can this happen?
    # Example: Base: 0-1-2. Add (0,2) -> Triangle 0-1-2. (Dist 2, excluded).
    # Example: Base: 0-1-2-3. Add (0,2) [excluded]. Add (1,3) [excluded].
    # Example: Base: 0-1-2-3-4. Add (0,3). Dist 3.
    # Triangle? 0-3 is added. 3-4 is base. 4-0 is added? If we add (0,4).
    # Then 0-3-4-0 is a triangle?
    # Edges: (0,3), (3,4), (4,0).
    # Is (0,3) valid? Dist 3.
    # Is (4,0) valid? Dist 4.
    # Do they conflict?
    # If we have added (0,3) and (0,4), do we have a triangle?
    # Neighbors of 0: 1, 3, 4.
    # Neighbors of 3: 2, 4, 0.
    # Neighbors of 4: 3, 5, 0.
    # Triangle 0-3-4?
    # (0,3) is edge. (3,4) is base edge. (4,0) is edge.
    # Yes! So adding (0,3) and (0,4) creates a triangle.
    # So we cannot just check pairwise independence of added edges against base.
    # We must check if the set of added edges creates any triangle.
    
    # The strategy:
    # 1. Identify "Critical Pairs": Pairs (u,v) with dist(u,v) > 4 in the base graph.
    #    Actually, we need the final diameter to be <= 4.
    #    So any pair (u,v) with dist_base(u,v) > 4 must be "covered" by added edges such that dist_final(u,v) <= 4.
    #    Wait, dist_final is the shortest path in the augmented graph.
    #    If dist_base(u,v) <= 4, it's already satisfied (unless we mess it up? No, adding edges only reduces distances).
    #    So we only need to worry about pairs with dist_base(u,v) > 4.
    #    Let's call these "Far Pairs".
    
    # 2. For each candidate edge e, determine which Far Pairs it "helps" cover.
    #    An edge e "covers" a Far Pair (u,v) if adding e reduces the distance between u and v to <= 4.
    #    This is tricky to precompute exactly without building the graph.
    #    However, we can use a heuristic or a simpler condition.
    #    Actually, a stronger condition: An edge e=(a,b) covers (u,v) if there is a path u->...->a-b->...->v of length <= 4.
    #    This depends on other added edges too.
    
    # Alternative Strategy: Branch and Bound on the set of added edges.
    # State: Current set of added edges S.
    # Constraint: S must be triangle-free.
    # Objective: Minimize |S| such that diameter(G+S) <= 4.
    
    # Precomputation:
    # 1. List of all valid candidate edges (not in base, dist != 2).
    # 2. For each candidate edge, precompute its "conflicts" with other candidate edges?
    #    Two edges e1=(a,b) and e2=(c,d) conflict if they form a triangle with a base edge or each other.
    #    Triangle with base edge:
    #    - e1 and e2 share a vertex, say a=c.
    #    - Then we have edges (a,b) and (a,d).
    #    - If (b,d) is a base edge, we have triangle a-b-d.
    #    - So if dist_base(b,d) == 1, then e1 and e2 conflict.
    #    Triangle with 2 base edges:
    #    - e1=(a,b). If dist_base(a,b)==2, conflict with base. (Already excluded).
    #    Triangle with 3 added edges?
    #    - e1=(a,b), e2=(b,c), e3=(c,a).
    #    - This requires 3 added edges forming a cycle of length 3.
    #    - Since we are adding edges to a path, can we form a triangle with 3 added edges?
    #    - Example: 0-1-2-3-4. Add (0,2) [excluded], (2,4) [excluded], (4,0).
    #    - If we add (0,2) and (2,4) and (4,0), we have triangle 0-2-4.
    #    - But (0,2) and (2,4) are excluded because dist=2.
    #    - What about (0,3), (3,1), (1,0)? (1,0) is base.
    #    - It seems hard to form a triangle with only added edges and no base edges involved in the cycle,
    #      unless the added edges themselves form a triangle.
    #      Example: Add (0,3), (3,5), (5,0).
    #      0-3 (added), 3-5 (added), 5-0 (added).
    #      Is this possible?
    #      Base: 0-1-2-3-4-5.
    #      Add (0,3): dist 3. OK.
    #      Add (3,5): dist 2. EXCLUDED.
    #      So any edge with dist 2 is excluded.
    #      Can we have a triangle of added edges where no edge has dist 2?
    #      Triangle vertices a,b,c.
    #      Edges (a,b), (b,c), (c,a).
    #      Distances in path: d(a,b), d(b,c), d(c,a).
    #      One of these must be the sum of the other two? No, that's for paths.
    #      In the path, d(a,c) = d(a,b) + d(b,c) if b is between a and c.
    #      If b is between a and c, then d(a,c) > d(a,b) and d(a,c) > d(b,c).
    #      If d(a,b) >= 3 and d(b,c) >= 3, then d(a,c) >= 6.
    #      So the edge (a,c) has dist >= 6.
    #      So we could have added edges (a,b), (b,c), (c,a) where dists are 3, 3, 6.
    #      Example: 0-1-2-3-4-5-6-7-8.
      #      Add (0,3), (3,6), (6,0).
      #      (0,3) dist 3.
      #      (3,6) dist 3.
      #      (6,0) dist 6.
      #      Do they form a triangle? Yes, 0-3-6-0.
      #      So we MUST check for triangles formed by 3 added edges.
      
    # This makes the problem complex for simple bitset pruning.
    # However, n=24 is small.
    # Number of candidates:
    # Total pairs: 24*23/2 = 276.
    # Base edges: 23.
    # Dist 2 edges: 22.
    # Candidates: 276 - 23 - 22 = 231.
    
    # Branch and Bound:
    # We want to find the minimum set of edges.
    # We can iterate k = 1, 2, 3... and check if a solution of size k exists.
    # For k=1: Check all 231 edges.
    # For k=2: Check all pairs (231*230/2 ~ 26k).
    # For k=3: Check all triples (231^3/6 ~ 2M). This might be tight in 90s if checking is slow.
    # But we can prune.
    
    # Let's implement a recursive search for a valid set of size k.
    # Pruning:
    # 1. Triangle-free check.
    # 2. Diameter check.
    
    # To speed up diameter check, we can use BFS.
    # To speed up triangle check, we can maintain adjacency sets.
    
    # Let's try to find a solution with k edges.
    
    def bfs_diameter(adj):
        max_dist = 0
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_dist:
                            max_dist = dist[v]
                        q.append(v)
        return max_dist

    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    if adj[u].intersection(adj[v]):
                        return False
        return True

    # Precompute candidates
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if frozenset((u, v)) in base_edge_set:
                continue
            if abs(u - v) == 2:
                continue
            candidates.append((u, v))
            
    # Sort candidates by some heuristic? Longer edges first?
    # Longer edges are more likely to reduce diameter significantly.
    candidates.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)
    
    # Map edge to index for bitset?
    # Maybe just use lists for k=1,2,3.
    
    # Check k=1
    for u, v in candidates:
        if time.time() > deadline: return []
        adj = [set(s) for s in base_adj]
        adj[u].add(v)
        adj[v].add(u)
        if is_triangle_free(adj) and bfs_diameter(adj) <= 4:
            return [[int(u), int(v)]]
            
    # Check k=2
    # Optimize: Only consider pairs of candidates that don't immediately form a triangle.
    # Two edges (a,b) and (c,d) form a triangle if:
    # 1. They share a vertex, say a=c. Then b and d must be adjacent in base.
    # 2. They don't share a vertex. Then they can't form a triangle with 1 base edge?
    #    Wait, triangle needs 3 edges.
    #    If (a,b) and (c,d) are disjoint, they can't form a triangle with 1 base edge (which connects 2 vertices).
    #    They could form a triangle with 2 base edges?
    #    Edges: (a,b) added, (c,d) added.
    #    Base edges: (b,c) and (d,a)?
    #    Then triangle a-b-c-d-a? No, that's a cycle of 4.
    #    Triangle a-b-c? Needs (a,c) base.
    #    So if (a,b) and (c,d) are disjoint, they can only form a triangle if there are base edges closing the loop.
    #    Specifically, if (a,c) is base and (b,d) is base? No.
    #    If (a,c) is base, we have path a-c-b? No.
    #    Let's just check triangle-free in the code for k=2, it's fast enough.
    #    26k checks * BFS(24) is fine.
    
    for i in range(len(candidates)):
        if time.time() > deadline: return []
        u1, v1 = candidates[i]
        for j in range(i + 1, len(candidates)):
            u2, v2 = candidates[j]
            adj = [set(s) for s in base_adj]
            adj[u1].add(v1)
            adj[v1].add(u1)
            adj[u2].add(v2)
            adj[v2].add(u2)
            
            # Quick triangle check
            if not is_triangle_free(adj):
                continue
                
            if bfs_diameter(adj) <= 4:
                return [[int(u1), int(v1)], [int(u2), int(v2)]]
                
    # Check k=3
    # 2M checks. Might be slow.
    # We can prune by checking if the 3 edges form a triangle among themselves or with base.
    # Let's optimize the triangle check.
    # Precompute for each pair of candidates if they are "compatible" (don't form a triangle with base).
    # Actually, just running is_triangle_free on the small graph is O(n^3) or O(n*E).
    # n=24, E~26. Very fast.
    
    # To speed up k=3, we can iterate smarter.
    # But 2M * (BFS + Triangle) might take a few seconds in Python.
    # Let's hope it's fast enough.
    
    # We can also add a heuristic: 
    # If we pick edge 1, we can restrict edge 2 to those that don't conflict.
    # Then edge 3 from those that don't conflict with 1 and 2.
    
    for i in range(len(candidates)):
        if time.time() > deadline: return []
        u1, v1 = candidates[i]
        adj1 = [set(s) for s in base_adj]
        adj1[u1].add(v1)
        adj1[v1].add(u1)
        
        # Check if adj1 is triangle free (it is, by construction of candidates)
        
        for j in range(i + 1, len(candidates)):
            u2, v2 = candidates[j]
            
            # Check if adding (u2,v2) to adj1 creates a triangle
            # (u2,v2) conflicts with adj1 if:
            # - u2 or v2 has a neighbor in adj1 that is also neighbor of the other.
            # - Or if u2==u1 or v2==u1 etc, check base neighbors.
            
            # Simple check:
            adj2 = [set(s) for s in adj1]
            adj2[u2].add(v2)
            adj2[v2].add(u2)
            
            if not is_triangle_free(adj2):
                continue
                
            for k in range(j + 1, len(candidates)):
                u3, v3 = candidates[k]
                
                adj3 = [set(s) for s in adj2]
                adj3[u3].add(v3)
                adj3[v3].add(u3)
                
                if not is_triangle_free(adj3):
                    continue
                    
                if bfs_diameter(adj3) <= 4:
                    return [[int(u1), int(v1)], [int(u2), int(v2)], [int(u3), int(v3)]]
                    
    # Check k=4
    # 231^4 / 24 ~ 100M. Too slow for Python.
    # Need better pruning or different approach.
    # If k=3 fails, k=4 is likely.
    # We can use a greedy heuristic to find a good candidate for k=4, then verify.
    # Or use the previous greedy method as a fallback.
    
    # Let's try a greedy approach for k=4 and up, then verify.
    # Greedy:
    # Start with empty set.
    # Repeatedly add the edge that covers the most "far pairs" while maintaining triangle-free.
    
    # But we already know k=1,2,3 don't work (if we got here).
    # So we need k>=4.
    
    # Let's use the previous program's greedy logic but with the exact verification.
    
    def get_uncovered_pairs(adj):
        """Get pairs of vertices with distance > 4."""
        uncovered = set()
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for j in range(i + 1, n):
                if dist[j] > 4:
                    uncovered.add((i, j))
        return uncovered

    current_adj = [set(s) for s in base_adj]
    chosen_edges = []
    chosen_set = set()
    
    def is_valid_addition(u, v, adj_list):
        """Check if adding (u,v) keeps triangle-free property."""
        if adj_list[u].intersection(adj_list[v]):
            return False
        return True

    uncovered = get_uncovered_pairs(current_adj)
    
    while uncovered and time.time() < deadline:
        best_edge = None
        best_new_uncovered = None
        max_covered = -1
        
        valid_candidates = []
        for u, v in candidates:
            if frozenset((u, v)) in chosen_set:
                continue
            if is_valid_addition(u, v, current_adj):
                valid_candidates.append((u, v))
                
        if not valid_candidates:
            break
        
        for u, v in valid_candidates:
            current_adj[u].add(v)
            current_adj[v].add(u)
            
            new_uncovered = get_uncovered_pairs(current_adj)
            covered_count = len(uncovered) - len(new_uncovered)
            
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            
            if covered_count > max_covered:
                max_covered = covered_count
                best_edge = (u, v)
                best_new_uncovered = new_uncovered
                
        if max_covered <= 0:
            break
        
        u, v = best_edge
        current_adj[u].add(v)
        current_adj[v].add(u)
        chosen_edges.append((u, v))
        chosen_set.add(frozenset((u, v)))
        uncovered = best_new_uncovered
    
    if is_triangle_free(current_adj) and bfs_diameter(current_adj) <= 4:
        # Try to remove edges
        sol = list(chosen_edges)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(sol)):
                candidate = sol[:i] + sol[i+1:]
                # Build adj for candidate
                adj_c = [set(s) for s in base_adj]
                for u,v in candidate:
                    adj_c[u].add(v)
                    adj_c[v].add(u)
                if is_triangle_free(adj_c) and bfs_diameter(adj_c) <= 4:
                    sol = candidate
                    improved = True
                    break
        return [[int(u), int(v)] for u, v in sol]
        
    return []

# EVOLVE-BLOCK-END
