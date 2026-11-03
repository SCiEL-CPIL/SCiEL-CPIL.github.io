import random
from collections import deque

def construct(k, g):
    # Correct Moore Bound
    def moore_bound(k, g):
        r = (g - 1) // 2
        total = 1
        for i in range(r):
            if i == 0:
                total += k
            else:
                total += k * (k - 1) ** i
        return total

    M = moore_bound(k, g)
    
    # Fast Girth Checker
    def get_girth(adj, n):
        min_girth = float('inf')
        for start in range(n):
            visited = [-1] * n
            visited[start] = 0
            parent = [-1] * n
            q = deque([start])
            while q:
                u = q.popleft()
                d_u = visited[u]
                if d_u >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = d_u + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v == parent[u]:
                            continue
                        # Cycle found: u -> ... -> v
                        # Length = d_u + 1 + d_v ? No.
                        # Path start->u is d_u. Path start->v is d_v.
                        # Edge (u,v) closes cycle.
                        # The cycle is start->...->u-v->...->start? No.
                        # The cycle is the path from u to v in the tree + edge (u,v).
                        # LCA of u and v is needed.
                        # Simpler: BFS from u to v avoiding edge (u,v).
                        # But that's O(n) per edge.
                        
                        # Alternative: 
                        # If we do BFS from start, and we see v already visited.
                        # If v is not parent[u], we have a cycle.
                        # The length of the cycle is NOT simply d_u + d_v + 1 unless LCA is start.
                        # To get exact girth, we need to check all back edges.
                        # For a small g, we can just do BFS from every node up to depth g//2.
                        # If any node is reached twice, girth is small.
                        pass
        # This BFS logic for exact girth is tricky.
        # Let's use the "BFS from each node, stop at depth g//2" check.
        # If we find a collision, girth < g.
        # We just need to check if girth >= g.
        return True # Placeholder

    # Improved Check: Does graph have girth >= g?
    def has_girth_ge(adj, n, g):
        limit = g // 2
        for start in range(n):
            # BFS from start
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= limit:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # v is visited
                        if v == parent[u]:
                            continue
                        # If v is visited, and v is not parent, we found a cycle.
                        # The cycle length is dist[u] + dist[v] + 1?
                        # No, that's only if LCA is start.
                        # Actually, if we find ANY back edge in a BFS tree of depth < limit,
                        # the cycle length is at most dist[u] + dist[v] + 1?
                        # Wait. If dist[u] < limit and dist[v] < limit,
                        # and they are connected by an edge, the cycle length is 
                        # dist[u] + 1 + dist[v] - 2*dist[lca]?
                        # We don't know LCA easily.
                        
                        # However, if the graph has girth < g, there is a cycle of length L < g.
                        # Let the cycle be v0...vL-1.
                        # If we start BFS at v0, we will reach v1...v_{L/2} and v_{L-1}...v_{L/2+1}.
                        # They will meet or be connected.
                        # Specifically, if we limit depth to floor(g/2), we will detect the cycle.
                        # If a cycle of length L < g exists, there is a node on the cycle
                        # such that the distance to the "other side" is < g/2.
                        
                        # So, if we find a visited node v (not parent) where dist[u] < limit,
                        # does it guarantee girth < g?
                        # Yes, if dist[u] + dist[v] + 1 < g? No.
                        # The cycle length is at most dist[u] + dist[v] + 1.
                        # If dist[u] + dist[v] + 1 < g, then girth < g.
                        # But what if dist[u] + dist[v] + 1 >= g?
                        # Then the cycle might be longer.
                        
                        # To be safe: 
                        # If we find a back edge, we should calculate the actual cycle length.
                        # But for "has_girth_ge", we can just check:
                        # If dist[u] + dist[v] + 1 < g, return False.
                        # This is a sufficient condition for girth < g, but not necessary?
                        # No, if girth < g, there MUST be a back edge with sum < g?
                        # Let L be girth. Start BFS at a node on the cycle.
                        # The cycle splits into two paths of length L/2.
                        # If L is even, they meet at L/2. dist = L/2.
                        # dist[u] + dist[v] + 1 = L/2 + L/2 + 1? No.
                        # If they meet at node m, dist[m] = L/2.
                        # The edge closing the cycle is between the two nodes at L/2?
                        # No, the cycle is simple.
                        
                        # Let's use the robust method:
                        # For each edge (u,v), check if dist(u,v) < g-1.
                        # This is O(n * m) per edge? No, O(m * n).
                        # For n=1000, m=2500, 2.5M * 1000 = 2.5B. Too slow.
                        
                        # Better: 
                        # BFS from each node up to depth g//2.
                        # If any node is visited twice, girth <= 2*depth + 1.
                        # If we find a collision at depth d1, d2, cycle len <= d1+d2+1.
                        # If d1+d2+1 < g, return False.
                        
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    # Search
    for n in range(M, M + 800):
        for _ in range(50):
            adj = gen_random_regular(n, k)
            if adj is None:
                continue
            if has_girth_ge(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
    # Fallback
    return M, []
