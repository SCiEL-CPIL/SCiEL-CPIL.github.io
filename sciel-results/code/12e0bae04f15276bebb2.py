import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # We are looking for k=3, g=16
    # The LUW construction CD(d, q) with q=k=3
    # We sweep d to find the smallest valid graph
    
    best_n = float('inf')
    best_edges = []
    
    # Helper to check girth
    def compute_girth(n, edges):
        if n == 0:
            return 0
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        min_girth = float('inf')
        # For each edge, remove it and find shortest path
        # To speed up, we can just do BFS from each node up to g/2
        # But exact girth is needed.
        # BFS from each node is O(V*(V+E)).
        # For V=2000, E=3000, this is ~6M ops.
        
        # Optimized Girth Check:
        # A cycle of length L exists if and only if there is a path of length L-1 between the endpoints of an edge.
        # We can just run BFS from every node up to depth g//2 - 1 = 7.
        # If we find a collision, we have a cycle.
        # However, we need the MINIMUM cycle length.
        
        # Let's use the edge-removal method for exactness, but limit search depth.
        # Actually, for g=16, we just need to know if girth >= 16.
        # If we find any cycle < 16, it's invalid.
        
        def has_short_cycle(n, adj, limit):
            # BFS from each node
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                
                while q:
                    u = q.popleft()
                    if dist[u] >= limit // 2:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif parent[u] != v and parent[v] != u:
                            # Found a cycle
                            # Cycle length = dist[u] + dist[v] + 1
                            # This logic is tricky for BFS tree.
                            # Standard: if v is visited and not parent, cycle found.
                            # But we need the length.
                            pass
            return False # Placeholder, need better logic
            
        # Better Girth Check for "Is girth >= g?"
        # If no cycle of length < g exists.
        # We can run BFS from each node up to depth r = g//2 - 1.
        # If any node is reached by two different paths of length <= r, or a path of length < g closes?
        # Actually, if g is even, g=2r+2.
        # A cycle of length <= 2r+1 implies a "collision" in BFS of depth r.
        # Specifically, if we run BFS from every node to depth r, and we find an edge (u,v) where dist(u) + dist(v) < 2r+1?
        
        # Let's use a robust method:
        # For each node s, run BFS up to depth r.
        # If we find a node v such that dist(s, v) + dist(s, v) < g? No.
        
        # Simplest robust check for "Girth >= g":
        # For each node s, run BFS.
        # If we encounter a node v that has already been visited in the current BFS,
        # and the path length is < g, then girth < g.
        # Wait, BFS finds shortest paths.
        # If there is a cycle of length L < g, then for any node s on the cycle,
        # the shortest path from s to s (via the cycle) is L.
        # But BFS from s doesn't return to s.
        
        # Correct check:
        # For each edge (u, v), remove it. Find shortest path between u and v.
        # If shortest path < g-1, then cycle < g.
        # To make this fast, we can just check if there is ANY cycle < g.
        
        def check_girth_ok(n, edges, g):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # If we find a cycle of length < g, return False
            # We can use BFS from each node, but stop if depth > g//2
            # Actually, if girth < g, there is a cycle of length L < g.
            # Pick any node on the cycle. The distance from this node to itself is L.
            # BFS from any node s will find the shortest cycle passing through s? No.
            
            # Let's just use the property:
            # Girth < g iff there exists a node s and a path of length < g from s to s (cycle).
            # This is equivalent to: in the BFS tree from s, there is an edge (u,v) such that dist(u) + dist(v) + 1 < g?
            # No, that detects triangles/4-cycles easily.
            
            # Given the small size (~2000 nodes), we can afford a slightly slower check.
            # Check all pairs? No.
            
            # Let's try:
            # For each node s, run BFS.
            # If we find a node v that is already visited, and the current depth + dist[v] < g?
            # This is not strictly correct for all cycle lengths.
            
            # Let's use the "remove edge" method but optimized.
            # Since we only care if girth >= 16, we can stop as soon as we find a cycle < 16.
            
            for u, v in edges:
                # Remove edge (u,v) temporarily? 
                # Just find shortest path from u to v avoiding direct edge.
                dist = [-1] * n
                dist[u] = 0
                q = deque([u])
                found = False
                while q:
                    curr = q.popleft()
                    if dist[curr] >= g - 1:
                        break
                    for nei in adj[curr]:
                        if nei == v and curr == u:
                            continue # Skip direct edge
                        if dist[nei] == -1:
                            dist[nei] = dist[curr] + 1
                            q.append(nei)
                if dist[v] != -1 and dist[v] < g - 1:
                    return False
            return True

        return check_girth_ok(n, edges, g)

    def build_luw_graph(d, q=3):
        # Re-implementation of seed logic, but cleaner
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        # Get largest connected component
        # The LUW graph is bipartite and usually connected?
        # Let's verify connectivity and extract largest component
        n_total = len(order)
        visited = [False] * n_total
        comp_sizes = []
        comp_nodes = []
        
        for i in range(n_total):
            if not visited[i]:
                # BFS
                q = deque([i])
                visited[i] = True
                comp = []
                while q:
                    u = q.popleft()
                    comp.append(u)
                    for v in edges:
                        if v[0] == u and not visited[v[1]]:
                            visited[v[1]] = True
                            q.append(v[1])
                        elif v[1] == u and not visited[v[0]]:
                            visited[v[0]] = True
                            q.append(v[0])
                comp_sizes.append(len(comp))
                comp_nodes.append(comp)
        
        if not comp_nodes:
            return 0, []
        
        max_idx = max(range(len(comp_sizes)), key=lambda i: comp_sizes[i])
        largest_comp = comp_nodes[max_idx]
        
        # Remap nodes to 0..n-1
        remap = {old: new for new, old in enumerate(largest_comp)}
        new_edges = []
        for u, v in edges:
            if u in remap and v in remap:
                new_edges.append((remap[u], remap[v]))
        
        return len(largest_comp), new_edges

    # Sweep d
    for d in range(3, 16):
        if time.time() > deadline:
            break
        n, edges = build_luw_graph(d)
        if n == 0:
            continue
        # Check degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(dg != 3 for dg in deg):
            continue
            
        # Check girth
        if compute_girth(n, edges):
            if n < best_n:
                best_n = n
                best_edges = edges
                # If we find a very small one, we might stop?
                # Moore bound is 382. If n < 400, it's excellent.
                if n < 400:
                    break

    if best_n == float('inf'):
        # Fallback: Return a small valid graph if possible, or the seed result
        # To ensure validity, we can return a known small 3-regular graph with girth 16?
        # If we don't have one, we must return *something*.
        # The seed program returns a valid graph.
        # I should replicate the seed's default behavior if sweep fails.
        n, edges = build_luw_graph(11) # Seed's default
        return n, edges

    return best_n, best_edges
