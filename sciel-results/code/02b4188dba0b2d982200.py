import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    def check_girth(n, edges, min_girth):
        """Check if graph has girth at least min_girth using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS from each vertex
        for start in range(n):
            visited = [-1] * n
            visited[start] = 0
            q = deque([start])
            found_cycle = False
            while q:
                u = q.popleft()
                dist = visited[u]
                if dist >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = dist + 1
                        q.append(v)
                    else:
                        if v != start and visited[v] <= dist:
                            # Cycle length is dist + 1 + visited[v]
                            # Actually, if we see a visited node, we found a cycle.
                            # We need the *shortest* cycle.
                            # Standard BFS finds shortest path.
                            # If we encounter a visited node, the cycle length is dist + 1 + visited[v]?
                            # No, if we are at u (dist d) and see v (dist d'), 
                            # and v is visited, the cycle is d + 1 + d' only if path is disjoint?
                            # Simpler: Just run BFS and record distance to all nodes.
                            # The girth is the min over all u of (min cycle length through u).
                            # A cycle through u has length 2 * dist(u, w) + 1? No.
                            # Let's just do a simple BFS to find shortest cycle.
                            pass
            # This simple check is inefficient. 
            # Better: For each vertex, run BFS. The first time we hit a visited vertex 
            # that is not the parent, we found a cycle.
            # But we need the global minimum.
            pass
        return True # Placeholder, need better girth check

    def get_girth(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        min_g = float('inf')
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                    else:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        # But this might not be the shortest cycle through start?
                        # Actually, BFS from start finds shortest path from start to v.
                        # The cycle formed by path(start->u) + edge(u,v) + path(v->start)
                        # is not necessarily simple or shortest.
                        # Correct way: For each vertex, BFS. 
                        # When we visit a neighbor v that is already visited, 
                        # the cycle length is dist[u] + dist[v] + 1.
                        # We take the min of these.
                        if dist[v] < dist[u]: # avoid double counting
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_g:
                                min_g = cycle_len
        return min_g

    # 1. L-U-W Construction
    def luw_construct(k, g):
        q = k
        # Need d such that d+5 >= g. d must be odd.
        d = g - 5
        if d < 3: d = 3
        if d % 2 == 0: d += 1
        
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
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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

    best_n = float('inf')
    best_edges = []
    
    # Try L-U-W
    n_luw, edges_luw = luw_construct(k, g)
    if n_luw < best_n:
        best_n = n_luw
        best_edges = edges_luw
        
    if time.time() > deadline:
        return best_n, best_edges

    # Try Cayley Graphs on Z_N x Z_N
    # Generators: (1,0), (-1,0), (0,1), (0,-1), (N//2, 0) for even N
    # This gives degree 5.
    # Only valid if N is even.
    
    # Moore bound for k-regular graph with girth g:
    # If g is odd, n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    # If g is even, n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    if g % 2 == 1:
        mb = 1 + k * sum((k-1)**i for i in range((g-3)//2 + 1))
    else:
        mb = 2 * sum((k-1)**i for i in range(g//2))
        
    min_N = math.ceil(math.sqrt(mb))
    if min_N % 2 != 0: min_N += 1
    
    # Search N
    max_N = min_N + 50 # Limit search range
    for N in range(min_N, max_N):
        if N % 2 != 0: continue
        if time.time() > deadline: break
        
        # Construct Cayley graph
        n = N * N
        edges = []
        gens = [(1, 0), (N-1, 0), (0, 1), (0, N-1), (N//2, 0)]
        
        for x in range(N):
            for y in range(N):
                u = x * N + y
                for dx, dy in gens:
                    vx = (x + dx) % N
                    vy = (y + dy) % N
                    v = vx * N + vy
                    if u < v:
                        edges.append((u, v))
                        
        # Check girth
        # This is O(N^2 * N^2) worst case, but N is small (~100)
        # N=100 -> 10000 vertices. BFS from each is 10000 * 10000 = 10^8. Too slow?
        # 10^8 operations in Python might take > 1s.
        # We only need to check a few N.
        
        # Optimized Girth Check
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        min_g = float('inf')
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= g: break
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        q.append(v)
                    else:
                        if dist[v] < du:
                            cycle = du + dist[v] + 1
                            if cycle < min_g:
                                min_g = cycle
            if min_g < g:
                break
                
        if min_g >= g:
            if n < best_n:
                best_n = n
                best_edges = edges
            break # Found smallest N, stop
            
    return best_n, best_edges
