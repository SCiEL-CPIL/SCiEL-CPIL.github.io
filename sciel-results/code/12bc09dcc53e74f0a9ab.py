   def add_edge_safe(adj, n, u, v):
       if adj[u][v]:
           return True
       
       # Check if u and v already have >= 2 common neighbors.
       # If so, C4 already exists (or we are adding a chord to an existing C4, which is fine for C4-free? 
       # No, if C4 exists, graph is invalid. We must maintain validity.)
       # Actually, if we maintain validity at each step, u and v should have <= 1 common neighbor.
       # If they have 2, the graph was already invalid. 
       # But wait, if they have 2 common neighbors w, x, the cycle u-w-v-x-u exists.
       # So if we reach this state, the graph is already bad.
       # So we should never be in a state where u and v have 2 common neighbors if we started with empty and added safely.
       
       # The real constraint for adding (u,v):
       # 1. u and v must have at most 1 common neighbor in the current graph.
       #    (If they have 2, say w, x, then u-w-v-x-u is a C4. The graph is already invalid. 
       #     But if we assume current graph is valid, this count is <= 1.)
       
       # 2. Adding (u,v) must not create a new C4.
       #    A new C4 would involve the edge (u,v) and a path of length 3 from u to v.
       #    Path: u - w - x - v.
       #    This requires (u,w), (w,x), (x,v) to be edges.
       #    Note that w and x are distinct and not u,v.
       #    Also, this implies w and v have common neighbor x? No.
       #    Let's check the "common neighbor" criterion again.
       #    Adding (u,v) creates a C4 if and only if there exists a vertex w such that:
       #    - (w,u) is an edge AND (w,v) is NOT an edge? No.
       #    Let's stick to the common neighbor count.
       #    After adding (u,v):
       #    For any pair (a,b), their common neighbors might change.
       #    Only pairs involving u or v change.
       #    Pair (u, w) where (u,w) is an edge:
       #       Common neighbors of u and w in new graph:
       #       Old common neighbors + {v} if (w,v) is an edge.
       #       So if (w,v) is an edge, v becomes a new common neighbor.
       #       We must ensure that u and w did not already have 1 common neighbor (other than v? no, v wasn't connected to u).
       #       So if (w,v) is an edge, and u and w already have 1 common neighbor x (x != v), then after adding (u,v), they have x and v. Count = 2. C4.
       #    Pair (v, w) where (v,w) is an edge:
       #       Similarly, if (u,w) is an edge, u becomes a new common neighbor.
       #       If v and w already have 1 common neighbor x, then count becomes 2. C4.
       
       # So the check is:
       # 1. Count common neighbors of u and v. If >= 2, return False (graph already invalid or adding edge to existing C4 structure).
       #    Actually, if current graph is valid, this count is <= 1.
       # 2. For each neighbor w of u:
       #      If (w, v) is an edge:
       #         Check if u and w have any other common neighbor x (x != u, v, w).
       #         If yes, return False.
       # 3. For each neighbor w of v:
       #      If (w, u) is an edge:
       #         Check if v and w have any other common neighbor x (x != u, v, w).
       #         If yes, return False.
       
       # Wait, steps 2 and 3 are symmetric and cover the same cases?
       # If (w,u) and (w,v) are edges, w is a common neighbor of u and v.
       # In step 2, we look at neighbors of u. w is a neighbor of u.
       # We check if (w,v) is edge. Yes.
       # We check if u and w have other common neighbors.
       # u's neighbors: ..., w, ...
       # w's neighbors: ..., u, ...
       # Common neighbors of u and w: vertices connected to both.
       # v is connected to u and w. So v is a common neighbor.
       # Are there others?
       # If there is another x connected to u and w, then u-w-x-v-u is a C4?
       # Edges: (u,w), (w,x), (x,u) -> Triangle u-w-x.
       # And (u,v), (v,w).
       # Cycle: u-x-w-v-u.
       # Edges: (u,x), (x,w), (w,v), (v,u).
       # Yes, this is a C4.
       # So if u and w have a common neighbor x (x!=v), then adding (u,v) creates a C4.
       
       # So the condition is:
       # For all w in neighbors(u):
       #   If (w,v) is an edge:
       #      If common_neighbors(u, w) > 0: return False
       #      (Note: common_neighbors(u,w) in current graph. v is not in it because (u,v) not added yet. 
       #       But wait, is v a neighbor of w? Yes. Is v a neighbor of u? No. 
       #       So v is NOT a common neighbor of u and w in the current graph.
       #       So we just check if count is > 0. If it is 1, say x, then x is a common neighbor. 
       #       Then u-x-w-v-u is C4.)
       
       # Similarly for neighbors of v.
       
       # Also, we must ensure u and v don't have 2 common neighbors already?
       # If they have 2, say a,b. Then u-a-v-b-u is C4. Graph already invalid.
       # So if we maintain validity, this count is always <= 1.
       
       # Implementation:
       # Precompute neighbor lists for speed? Or just use adj matrix.
       
       common_uv = 0
       for w in range(n):
           if w != u and w != v and adj[u][w] and adj[v][w]:
               common_uv += 1
               if common_uv >= 2:
                   return False
       
       # Check neighbors of u
       for w in range(n):
           if w != u and w != v and adj[u][w]:
               if adj[w][v]:
                   # w is common neighbor of u and v.
                   # Check if u and w have any other common neighbor.
                   # If yes, C4 formed.
                   # Common neighbors of u and w:
                   # v is one (in new graph).
                   # Are there others?
                   # Check if any x (x != u, v, w) is connected to both u and w.
                   for x in range(n):
                       if x != u and x != v and x != w and adj[u][x] and adj[w][x]:
                           return False
       
       # Check neighbors of v
       for w in range(n):
           if w != u and w != v and adj[v][w]:
               if adj[w][u]:
                   # w is common neighbor of v and u.
                   # Check if v and w have any other common neighbor.
                   for x in range(n):
                       if x != u and x != v and x != w and adj[v][x] and adj[w][x]:
                           return False
       
       adj[u][v] = 1
       adj[v][u] = 1
       return True
   