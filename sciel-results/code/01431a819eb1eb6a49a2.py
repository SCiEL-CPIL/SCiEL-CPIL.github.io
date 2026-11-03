import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Moore Bound for even girth g = 2r
    # n >= 2 * sum_{i=0}^{r-1} (k-1)^i
    r = g // 2
    moore_bound = 2 * sum((k-1)**i for i in range(r))
    
    # LUW Fallback
    def luw_fallback():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
            r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

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
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Random Greedy Search
    def try_random_greedy(n):
        # n must be even for bipartite? Not necessarily, but k-regular graphs with odd n
        # require k to be even? No.
        # But girth >= 4 implies no triangles.
        # For k=4, n can be odd?
        # Sum of degrees = 4n. Must be even. 4n is always even.
        # So n can be any integer.
        
        adj = [[] for _ in range(n)]
        deg = [0] * n
        edges_set = set()
        edges_list = []
        target_edges = n * k // 2
        
        # Pre-compute potential edges? No, just random sampling.
        
        def has_short_path(u, v, max_dist):
            # BFS from u to v, limit depth to max_dist
            if u == v:
                return 0
            visited = {u: 0}
            q = deque([(u, 0)])
            while q:
                node, dist = q.popleft()
                if dist >= max_dist:
                    continue
                for nei in adj[node]:
                    if nei == v:
                        return dist + 1
                    if nei not in visited:
                        visited[nei] = dist + 1
                        q.append((nei, dist + 1))
            return None

        while len(edges_list) < target_edges:
            # Pick random u, v
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or deg[u] >= k or deg[v] >= k:
                continue
            if (u, v) in edges_set or (v, u) in edges_set:
                continue
            
            # Check if adding (u,v) creates a cycle < g
            # This happens if there is a path between u and v of length < g-1
            # Since we are adding the edge, the cycle length is path_len + 1.
            # We need cycle_len >= g => path_len >= g-1.
            # So we reject if path_len < g-1.
            # Max path length to check is g-2.
            
            if g > 2:
                path_len = has_short_path(u, v, g - 2)
                if path_len is not None:
                    continue
            
            # Add edge
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            edges_set.add((u, v))
            edges_set.add((v, u))
            edges_list.append((u, v))
            
        return n, edges_list

    # Strategy: Try increasing n from Moore bound
    # Start with a small offset
    n_try = moore_bound
    if n_try % 2 != 0 and k % 2 != 0:
        n_try += 1 # Actually k=4 is even, so n can be odd.
    # But for bipartite graphs (girth >= 4), n must be even?
    # No, bipartite graphs can have odd number of vertices?
    # No, bipartite graphs have even number of edges?
    # A bipartite graph with parts A and B.
    # If it's regular, |A|=|B|?
    # Yes, for k-regular bipartite, |A|=|B|=n/2.
    # So n must be even.
    
    if n_try % 2 != 0:
        n_try += 1
        
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        try:
            n, edges = try_random_greedy(n_try)
            # Verify
            if len(edges) == n * k // 2:
                # Check degrees
                deg_check = [0] * n
                for u, v in edges:
                    deg_check[u] += 1
                    deg_check[v] += 1
                if all(d == k for d in deg_check):
                    return n, edges
        except Exception:
            pass
        
        # If stuck, increase n slightly
        # Moore bound is a lower bound.
        # Often the cage is larger.
        # Increase by 2 to maintain evenness.
        n_try += 2
        # Cap n_try to avoid infinite loop if search is bad
        if n_try > 5000:
            break
            
    # Fallback
    return luw_fallback()
