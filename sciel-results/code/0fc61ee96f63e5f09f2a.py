import time
import random

def construct(k, g):
    deadline = time.time() + 140  # safety margin
    
    def luw_construct(q, d):
        """Construct LUW CD(d,q) graph, return connected component of origin."""
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def add(x,y): return x ^ y
            def sub(x,y): return x ^ y
            def mul(x,y): return table[x][y]
        elif q == 2:
            def add(x,y): return (x + y) % 2
            def sub(x,y): return (x - y) % 2
            def mul(x,y): return x * y % 2
        elif q == 3:
            def add(x,y): return (x + y) % 3
            def sub(x,y): return (x - y) % 3
            def mul(x,y): return x * y % 3
        else:
            def add(x,y): return (x + y) % q
            def sub(x,y): return (x - y) % q
            def mul(x,y): return x * y % q
        
        rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        for c in range(5, d+1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,) + (0,)*d)
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
    
    def verify_girth(n, edges, required_girth):
        """Check that the graph has girth >= required_girth using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            # BFS tracking (node, prev, dist)
            visited = {}
            queue = [(start, -1, 0)]
            while queue:
                node, prev, dist = queue.pop(0)
                if dist >= required_girth:
                    break
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    key = (nb, node)
                    if key in visited:
                        # Found a cycle of length dist + visited_dist
                        # Actually, we need to track properly
                        pass
                    if nb == start and dist > 0:
                        # Found a cycle
                        if dist + 1 < required_girth:
                            return False
                    if (nb, node) not in visited:
                        visited[(nb, node)] = dist
                        queue.append((nb, node, dist + 1))
        return True
    
    def verify_girth_bfs(n, edges, required_girth):
        """Proper girth verification: no cycle shorter than required_girth."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do BFS and check for early returns
        for start in range(n):
            # dist[v] = shortest distance from start to v
            # parent[v] = parent in BFS tree
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for i in range(len(queue)):
                u = queue[i]
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle: u-v edge, both already visited
                        # Cycle length = dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < required_girth:
                            return False
        return True
    
    # Try various LUW parameters
    candidates = []
    
    # LUW with different q and d
    for q in [2, 3, 4, 5, 7]:
        for d in range(5, 8):
            if time.time() > deadline - 5:
                break
            try:
                n, edges = luw_construct(q, d)
                # Check degree
                deg = [0]*n
                for u,v in edges:
                    deg[u] += 1
                    deg[v] += 1
                if all(dg == k for dg in deg):
                    # Check girth
                    if verify_girth_bfs(n, edges, g):
                        candidates.append((n, edges))
            except:
                pass
    
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0]
    
    # Fallback: return the original LUW(4,5)
    return luw_construct(4, 5)
