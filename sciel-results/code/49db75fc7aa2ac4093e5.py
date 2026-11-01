import random
import time

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds safety margin
    
    # Fallback: The Lazebnik-Ustimenko-Woldar graph from the seed
    # This returns 320 vertices for k=4, g=10.
    def luw_fallback():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
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
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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
        return len(order), edges

    # Try Voltage Graph Construction
    # Base graph: K5 (5 vertices, 4-regular)
    base_vertices = 5
    base_edges = [(0, 1), (0, 2), (0, 3), (0, 4),
                  (1, 2), (1, 3), (1, 4),
                  (2, 3), (2, 4),
                  (3, 4)]
    
    # We want N = 5 * M < 320 => M < 64
    # Try M from small to large to find the smallest valid graph
    # But smaller M is harder to satisfy constraints?
    # Actually, larger group makes it easier to avoid identity products.
    # So start with M that gives N close to 320 but smaller?
    # Or start with small M and increase?
    # If M is too small, it's impossible.
    # Let's try M values that result in N < 320.
    # Candidates for M: 10, 12, 14, ..., 62.
    
    random.seed(42)
    
    def check_girth(n, adj, min_girth):
        # Check if graph has girth >= min_girth
        # BFS from each vertex
        for u in range(n):
            visited = [-1] * n
            parent = [-1] * n
            depth = [-1] * n
            queue = [u]
            visited[u] = u
            depth[u] = 0
            idx = 0
            while idx < len(queue):
                curr = queue[idx]
                idx += 1
                for v in adj[curr]:
                    if visited[v] == -1:
                        visited[v] = curr
                        parent[v] = curr
                        depth[v] = depth[curr] + 1
                        queue.append(v)
                    else:
                        # If v is visited and not parent, we found a cycle
                        # The cycle length is depth[curr] + depth[v] + 1?
                        # No, in BFS tree, if edge (curr, v) exists and v is visited,
                        # and parent[curr] != v and parent[v] != curr,
                        # then we have a cycle.
                        # Length = depth[curr] + depth[v] + 1 is NOT correct for general graph.
                        # Correctly: The path from u to curr and u to v are known.
                        # The cycle is formed by the path u->...->curr, edge curr-v, path v->...->u.
                        # But we need the length of the cycle.
                        # Actually, simpler: just check if depth[curr] + depth[v] + 1 < min_girth?
                        # This works if the graph is a tree, but it's not.
                        # However, if we find ANY cycle of length < min_girth, we fail.
                        # The shortest cycle passing through u can be found by checking back edges.
                        # If we see an edge (curr, v) where v is an ancestor of curr in BFS tree,
                        # cycle length is depth[curr] - depth[v] + 1.
                        # If v is not ancestor, it's a cross edge, which also forms a cycle.
                        
                        # Simpler robust check:
                        # Since N is small (< 320), we can just run BFS up to depth min_girth/2 - 1?
                        # No, girth 10 means no cycles 3-9.
                        # We can just check all pairs?
                        # Let's use a standard girth check:
                        # For each vertex, BFS up to depth g/2 - 1.
                        # If we hit a visited node at depth d2, and current depth is d1,
                        # and the nodes are not parent-child, we have a cycle of length d1+d2+1?
                        # Actually, if we stop BFS at depth 4 (for girth 10),
                        # any cycle of length < 10 must be detected?
                        # A cycle of length 9 has diameter 4.
                        # So BFS depth 4 is sufficient to detect cycles up to length 9?
                        # If a cycle of length 9 exists, there is a vertex on it with distance 4 to two other vertices on the cycle?
                        # Yes.
                        
                        # So: BFS from u up to depth 4.
                        # If we encounter a vertex already in the BFS tree (at depth <= 4),
                        # and it's not the parent, we have a cycle.
                        # The length is depth[curr] + depth[v] + 1?
                        # Not exactly, but if depth[curr] + depth[v] + 1 < 10, it's a short cycle.
                        # We just need to know if ANY cycle < 10 exists.
                        
                        if parent[curr] != v and parent[v] != curr:
                            # Potential cycle
                            # The cycle length is at least 3.
                            # We need to be sure it's < 10.
                            # Since we only go to depth 4, the max distance is 4.
                            # Max cycle length detected this way: 4+4+1 = 9.
                            # So any cycle detected here has length <= 9.
                            # Thus, if we detect ANY such cycle, girth < 10.
                            return False
            # End while
        return True

    def build_lift(M):
        # M is the size of the group Z_M
        n = base_vertices * M
        adj = [[] for _ in range(n)]
        
        # Assign random voltages to base edges
        voltages = [random.randint(0, M-1) for _ in range(len(base_edges))]
        
        # Map base vertex to lift vertex index
        # Lift vertex (v, x) -> v*M + x
        
        for i, (u, v) in enumerate(base_edges):
            g = voltages[i]
            # Edge (u, x) -> (v, x+g)
            # Edge (v, x) -> (u, x-g)
            for x in range(M):
                u_lift = u * M + x
                v_lift = v * M + (x + g) % M
                
                # Add undirected edge
                adj[u_lift].append(v_lift)
                adj[v_lift].append(u_lift)
                
        return n, adj

    def get_edges_from_adj(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges

    # Try different group sizes M
    # We want 5*M < 320 => M < 64
    # Try M in descending order? No, smaller N is better.
    # But smaller M is harder.
    # Let's try M from 20 to 63.
    # If we find one, return it.
    
    # To save time, we can limit trials per M.
    trials_per_m = 200
    
    for M in range(20, 64):
        if time.time() > deadline:
            break
        for _ in range(trials_per_m):
            if time.time() > deadline:
                break
            n, adj = build_lift(M)
            # Quick check: is it simple? (Should be)
            # Check girth
            if check_girth(n, adj, 10):
                edges = get_edges_from_adj(adj, n)
                return n, edges
                
    # If no lift found, fallback to LUW
    return luw_fallback()
