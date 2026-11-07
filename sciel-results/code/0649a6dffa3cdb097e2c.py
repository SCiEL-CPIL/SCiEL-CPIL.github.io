import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
        
    # Precompute all non-base candidate edges
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set:
                all_candidates.append((u, v))
    
    # Conflict matrix: conflict[i][j] = True if edges i and j form a triangle together
    # Edges i=(a,b), j=(c,d). They conflict if there's a vertex x such that:
    # x is adjacent to both a and b in the current graph (which depends on the set of edges chosen)
    # WAIT. Triangle-free constraint: The UNION of base edges and added edges must be triangle-free.
    # A triangle is formed by 3 edges. Since base is a path (no triangles), any triangle must involve at least 2 added edges?
    # No. A triangle can be:
    # 1. 3 added edges.
    # 2. 2 added edges + 1 base edge.
    # 3. 1 added edge + 2 base edges. (Impossible, base is a path, so no vertex has 2 base neighbors in a way that forms a triangle with 1 shortcut? 
    #    Let's check: Base path 0-1-2. Add edge (0,2). Triangle 0-1-2-0. Yes, this is possible.)
    
    # So, an added edge (u,v) conflicts with the current graph if u and v have a common neighbor in the current graph.
    # Also, two added edges (u,v) and (x,y) conflict if they form a triangle with each other or with base edges.
    
    # This makes the "independent set" formulation dynamic. 
    # However, we can pre-compute a static conflict graph for *pairs* of added edges.
    # Two added edges e1=(u,v) and e2=(x,y) are incompatible if:
    # 1. They share a vertex? No, sharing a vertex is fine unless it creates a triangle.
    #    If they share a vertex, say u=x, then edges (u,v) and (u,y).
    #    Do they form a triangle? Only if (v,y) is an edge (base or added).
    #    If (v,y) is a base edge, then (u,v) and (u,y) form a triangle with base edge (v,y).
    #    If (v,y) is an added edge, we'd have a triangle of 3 added edges.
    # 2. They don't share a vertex.
    #    Do they form a triangle? No, a triangle needs 3 vertices.
    #    But they might form a triangle with a base edge?
    #    No, a triangle involving 2 added edges and 1 base edge requires the base edge to connect two vertices that are "close" to the added edges.
    #    Example: Added (0,2) and (0,3). Base edge (2,3) doesn't exist.
    #    Example: Added (0,2) and (1,3). 
    #    Triangle 0-2-1-0? Edge (2,1) is base. Edge (1,0) is base. Edge (0,2) added.
    #    Wait, (0,2) and (1,3).
    #    Triangle: 0-2-1-0? 
    #    0-2 (added), 2-1 (base), 1-0 (base). This forms a triangle!
    #    So (0,2) and (1,3) are incompatible because they create a triangle with base edges (0,1) and (1,2).
    
    # General condition for incompatibility of two added edges e1=(u,v) and e2=(x,y):
    # The set of 3 edges {e1, e2, base_edge} forms a triangle.
    # This happens if:
    # - e1 and e2 share a vertex, say u=x. And (v,y) is a base edge.
    # - e1 and e2 do not share a vertex. And there exists a base edge (a,b) such that {u,v,x,y,a,b} form a triangle?
    #   Actually, if e1=(u,v) and e2=(x,y), and they are disjoint.
    #   Can they form a triangle? No, a triangle has 3 vertices. Two disjoint edges have 4 distinct vertices.
    #   So they can only form a triangle if they are NOT disjoint?
    #   Wait, the example (0,2) and (1,3) are disjoint. 
    #   Vertices: 0,2 and 1,3.
    #   Triangle 0-1-2-0 uses edges (0,1) base, (1,2) base, (0,2) added. 
    #   This triangle only involves ONE added edge (0,2).
    #   So (0,2) is incompatible with the *base graph* alone?
    #   Let's re-read: "The union with the pinned path must remain triangle-free".
    #   If I add (0,2), the graph has 0-1, 1-2, 0-2. This is a triangle.
    #   So (0,2) is INVALID on its own?
    #   Let's check the seed program. It says "Triangle-free".
    #   In a path graph, adding a shortcut (u,v) where dist(u,v) >= 2.
    #   If dist(u,v) == 2, i.e., u and v have a common neighbor w in the path, then (u,v) creates a triangle u-w-v-u.
    #   So ANY shortcut of length 2 is forbidden?
    #   Let's verify.
    #   Path 0-1-2. Add (0,2). Triangle 0-1-2. Yes.
    #   Path 0-1-2-3. Add (0,2). Triangle 0-1-2. Yes.
    #   So, shortcuts of length 2 are always forbidden?
    #   What about length 3?
    #   Path 0-1-2-3. Add (0,3).
    #   Edges: 0-1, 1-2, 2-3, 0-3.
    #   Triangles? 0-1-2 (no 0-2). 1-2-3 (no 1-3). 0-1-3 (no 1-3). 
    #   So length 3 is OK.
    #   What about length 4?
    #   Path 0-1-2-3-4. Add (0,4).
    #   Triangles? No.
    #   
    #   So, any added edge (u,v) must satisfy dist_path(u,v) >= 3.
    #   If dist_path(u,v) == 2, it creates a triangle with the middle node.
    #   If dist_path(u,v) == 1, it's a base edge (forbidden).
    #   If dist_path(u,v) >= 3, it doesn't create a triangle with base edges alone.
    #   
    #   Now, what about two added edges?
    #   If e1 and e2 both have dist >= 3, can they create a triangle together?
    #   Triangle requires 3 vertices.
    #   Case 1: They share a vertex. e1=(u,v), e2=(u,w).
    #   Triangle u-v-w-u requires edge (v,w).
    #   If (v,w) is base, then dist_path(v,w) must be small.
    #   But dist_path(u,v)>=3 and dist_path(u,w)>=3.
    #   Can v and w be close?
    #   Example: u=0. v=3 (dist 3). w=6 (dist 6).
    #   dist_path(3,6) = 3. Edge (3,6) is not base.
    #   So no triangle.
    #   What if v and w are such that (v,w) is a base edge?
    #   Then dist_path(v,w) = 1.
    #   Then dist_path(u,v) + dist_path(u,w) ...
    #   In a path, if u is 0, v is 1, w is 2. Dist(0,1)=1 (base, not allowed).
    #   So v and w must be at least 3 away from u.
    #   Can two nodes be 3+ away from u and adjacent to each other?
    #   In a path, neighbors of u are u-1, u+1.
    #   If v and w are neighbors, they are u-1 and u+1.
    #   Dist(u, u-1) = 1. Not allowed.
    #   So, if both e1 and e2 are valid (dist >= 3), they CANNOT form a triangle with a base edge?
    #   Let's check: u=0, v=3, w=1? No, w must be dist>=3 from u.
    #   So v and w are far from u.
    #   Can v and w be adjacent?
    #   If v and w are adjacent, dist(v,w)=1.
    #   Triangle u-v-w-u.
    #   Edges: (u,v) added, (v,w) base, (w,u) added.
    #   Is this possible?
    #   dist(u,v) >= 3, dist(u,w) >= 3.
    #   In a path, if dist(u,v)>=3 and dist(u,w)>=3, can v and w be adjacent?
    #   Let u=0. v >= 3 or v <= -3 (no). So v in {3..23}.
    #   w in {3..23}.
    #   If v and w are adjacent, say v=k, w=k+1.
    #   Then k >= 3.
    #   Triangle 0-k-(k+1)-0.
    #   Edges: (0,k), (k,k+1), (k+1,0).
    #   This is a valid triangle!
    #   So, (0,k) and (0,k+1) are INCOMPATIBLE if k>=3.
    #   Because they form a triangle with base edge (k,k+1).
    #   
    #   Conclusion: 
    #   1. Any edge (u,v) with path-dist < 3 is forbidden.
    #   2. Two edges sharing a vertex u, (u,v) and (u,w), are incompatible if (v,w) is a base edge.
    #      This happens if v and w are adjacent in the path.
    #   3. Two disjoint edges (u,v) and (x,y) are always compatible?
    #      They can't form a triangle (need 3 vertices, have 4).
    #      They can't form a triangle with base edges unless... 
    #      Wait, what if the triangle uses 1 added edge and 2 base edges?
    #      That would mean the added edge itself creates a triangle with base edges.
    #      We already established that if dist(u,v)>=3, it doesn't.
    #      So, disjoint edges are always compatible with each other and with the base graph?
    #      Let's double check.
    #      Triangle formed by e1=(u,v) added, and base edges (a,b), (b,c).
    #      Vertices u,v,a,b,c.
    #      If e1 and e2 are disjoint, and e2=(x,y) added.
    #      Can e1, e2 and base edges form a triangle?
    #      No, a triangle has 3 vertices. Two disjoint edges use 4 vertices.
    #      So they cannot be part of the same triangle.
    #      
    #   So the conflict graph is:
    #   - Nodes: Valid candidate edges (dist >= 3).
    #   - Edge between i and j if they share a vertex and the other two endpoints are adjacent in the path.
    
    # Let's build this conflict graph.
    
    valid_candidates = []
    for u, v in all_candidates:
        if abs(u - v) >= 3:
            valid_candidates.append((u, v))
            
    num_valid = len(valid_candidates)
    # Conflict matrix (sparse)
    # conflict[i] = set of j such that edge i and edge j are incompatible
    conflict = [set() for _ in range(num_valid)]
    
    for i in range(num_valid):
        u1, v1 = valid_candidates[i]
        for j in range(i + 1, num_valid):
            u2, v2 = valid_candidates[j]
            # Check if they share a vertex
            if u1 == u2:
                # Check if (v1, v2) is a base edge
                if (min(v1, v2), max(v1, v2)) in base_set:
                    conflict[i].add(j)
                    conflict[j].add(i)
            elif u1 == v2:
                # (u1, v1) and (u2, v2) -> share v1==u2? No, u1==v2.
                # Vertices: u1, v1, u2, v2.
                # Shared: v1 (from e1) and u2 (from e2)? No, u1 is shared.
                # e1 = (u1, v1), e2 = (u2, v2).
                # If u1 == v2.
                # Shared vertex is u1 (which is v2).
                # Other endpoints: v1 and u2.
                if (min(v1, u2), max(v1, u2)) in base_set:
                    conflict[i].add(j)
                    conflict[j].add(i)
            elif v1 == u2:
                # Shared vertex v1 (from e1) and u2 (from e2).
                # Other endpoints: u1 and v2.
                if (min(u1, v2), max(u1, v2)) in base_set:
                    conflict[i].add(j)
                    conflict[j].add(i)
            elif v1 == v2:
                # Shared vertex v1/v2.
                # Other endpoints: u1 and u2.
                if (min(u1, u2), max(u1, u2)) in base_set:
                    conflict[i].add(j)
                    conflict[j].add(i)
            
    # Now we have a conflict graph.
    # We want to find the smallest set of edges (independent set in conflict graph? No, independent set in conflict graph means no conflicts. So yes, we want an independent set in the conflict graph)
    # such that the diameter of the resulting graph is <= 4.
    
    # Wait, "independent set in conflict graph" means a set of edges where NO two edges conflict.
    # This is exactly what we want for triangle-free!
    # So we need to find the smallest independent set (in the conflict graph of valid candidates) that reduces diameter to <= 4.
    
    # This is a "Minimum Independent Set with Constraint" problem.
    # Or, we can search for the smallest size k, and check if there exists an independent set of size k that works.
    
    # Since n=24, the number of valid candidates is ~240.
    # The conflict graph is sparse.
    # We can use a backtracking search (branch and bound) to find the minimum size independent set that satisfies the diameter constraint.
    
    # However, checking diameter is expensive.
    # We can precompute the "coverage" of each edge?
    # Or, we can iterate k = 1, 2, 3... and try to find a valid set of size k.
    
    # Lower bound on k:
    # The hub solution has 10 edges.
    # Can we do better?
    # Let's try k from 1 up to 10.
    
    def check_diameter(edge_set):
        # Build adj
        adj = {i: set() for i in range(n)}
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for idx in edge_set:
            u, v = valid_candidates[idx]
            adj[u].add(v)
            adj[v].add(u)
        
        max_diam = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            if any(d == -1 for d in dist):
                return 999
            for d in dist:
                if d > max_diam:
                    max_diam = d
        return max_diam

    # Backtracking search for independent set of size k
    # We want to find ANY independent set of size k that gives diameter <= 4.
    
    def search_independent_set(k, current_set, start_idx):
        # current_set: list of indices of chosen edges
        # start_idx: index to start checking for next edge
        if len(current_set) == k:
            diam = check_diameter(current_set)
            if diam <= 4:
                return current_set[:]
            return None
        
        # Pruning: if not enough edges left
        remaining = num_valid - start_idx
        if remaining < k - len(current_set):
            return None
            
        # Heuristic: Pick candidates that cover "far" pairs?
        # Or just iterate.
        
        for i in range(start_idx, num_valid):
            # Check if i is compatible with all in current_set
            compatible = True
            for j in current_set:
                if i in conflict[j]:
                    compatible = False
                    break
            if not compatible:
                continue
                
            new_set = current_set + [i]
            res = search_independent_set(k, new_set, i + 1)
            if res:
                return res
        return None

    # Try k from 1 to 10
    for k in range(1, 11):
        if time.time() > deadline:
            break
        res = search_independent_set(k, [], 0)
        if res:
            # Convert indices to edges
            added_edges = [valid_candidates[i] for i in res]
            return [[int(u), int(v)] for u, v in added_edges]
            
    # Fallback to hub if search fails or times out
    added_edges = []
    hub = 11
    targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
    for v in targets:
        edge = (min(hub, v), max(hub, v))
        if edge not in base_set:
            added_edges.append([hub, v])
    return [[int(u), int(v)] for u, v in added_edges]

# EVOLVE-BLOCK-END
