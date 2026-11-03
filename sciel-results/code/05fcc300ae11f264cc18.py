import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: The original seed logic
    q = k
    # Calculate d for the seed's logic
    d_seed = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Helper to build LUW graph for given d
    def build_luw(d):
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

    # Helper to check girth
    def check_girth(n, edges, min_girth):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex
        for start in range(n):
            # dist[u][prev] is not needed, just dist and parent
            # To detect cycles, we need to check if a neighbor is visited and not parent
            # But for girth, we just need the shortest cycle.
            # BFS level by level.
            visited = [-1] * n
            parent = [-1] * n
            queue = [(start, 0)]
            visited[start] = 0
            while queue:
                u, depth = queue.pop(0)
                if depth >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = depth + 1
                        parent[v] = u
                        queue.append((v, depth + 1))
                    elif v != parent[u]:
                        # Found a cycle
                        # Length is depth + 1 + (distance from start to v)
                        # Actually, if we find a cross edge, cycle length is depth + 1 + visited[v]?
                        # No, BFS tree.
                        # If u and v are both visited and v != parent[u], 
                        # the cycle length is depth(u) + depth(v) + 1?
                        # Only if they are in the same BFS tree?
                        # Yes, we are doing BFS from start.
                        cycle_len = depth + visited[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    best_n = None
    best_edges = None
    
    # First, compute the seed's result as fallback
    n_seed, edges_seed = build_luw(d_seed)
    best_n = n_seed
    best_edges = edges_seed
    
    # Try smaller d
    # Minimum d for girth g is roughly g/2
    # Start from max(3, g//2 - 2)
    min_d = max(3, g // 2 - 2)
    
    for d in range(min_d, d_seed + 1):
        if time.time() > deadline:
            break
            
        n, edges = build_luw(d)
        
        # If n is already larger than best, skip check (but d is increasing, so n increases)
        # Wait, we are iterating d upwards.
        # We want the SMALLEST n, so SMALLEST d that works.
        # So we should iterate d from small to large.
        
        # Check girth
        # Optimization: If n is large, checking girth is expensive.
        # But for small d, n is small.
        
        if check_girth(n, edges, g):
            best_n = n
            best_edges = edges
            # Since we iterate d from small to large, the first valid one is the smallest.
            # We can break here?
            # Yes, because n grows exponentially with d.
            break
            
    return best_n, best_edges
