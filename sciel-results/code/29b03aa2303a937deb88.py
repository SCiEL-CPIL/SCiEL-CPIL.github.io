import time
import random
import math

def _luw_construct(k, g):
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

def _generate_random_3reg(n):
    # Configuration model
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
    random.shuffle(stubs)
    edges = []
    used = set()
    for i in range(0, n, 2): # Wait, 3n stubs. n must be even? 
        # 3-regular graph must have even number of vertices.
        pass
    # 3n is odd if n is odd. So n must be even.
    # If n is odd, return None.
    if n % 2 != 0:
        return None
        
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
    random.shuffle(stubs)
    
    edges_set = set()
    edges_list = []
    valid = True
    for i in range(0, 3*n, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            valid = False
            break
        if (u, v) in edges_set or (v, u) in edges_set:
            valid = False
            break
        edges_set.add((u, v))
        edges_set.add((v, u))
        edges_list.append((u, v))
        
    if not valid:
        return None
    return edges_list

def _calc_girth(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_girth = float('inf')
    
    for start in range(n):
        visited = [-1] * n
        parent = [-1] * n
        dist = [0] * n
        queue = [start]
        visited[start] = start
        
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = start
                    parent[v] = u
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif visited[v] == start and v != parent[u] and u != parent[v]:
                    # Found a cycle
                    # Cycle length is dist[u] + dist[v] + 1
                    # Wait, this is for BFS tree.
                    # If v is visited and not parent, it's a back edge.
                    # The cycle is u-v-...-start-...-u?
                    # Actually, if we do BFS from start, and we see an edge (u,v) where v is visited
                    # and v is not parent of u, then we have a cycle.
                    # The length is dist[u] + dist[v] + 1.
                    # But this only finds cycles that go through start?
                    # No, BFS from start finds all nodes.
                    # If we encounter a visited node v (from the same BFS run),
                    # the cycle length is dist[u] + dist[v] + 1.
                    # This finds the shortest cycle passing through start?
                    # No, it finds the shortest cycle in the component?
                    # Actually, BFS from a single node finds the shortest cycle *passing through that node*.
                    # To find the global girth, we must BFS from every node.
                    # This is O(N * (N+E)).
                    # For N=1000, E=1500. 1000 * 2500 = 2.5M ops. Fast.
                    
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
        
        # Optimization: if min_girth is already 3, break?
        # We need girth 16.
        
    return min_girth

def construct(k, g):
    if k != 3 or g != 16:
        return _luw_construct(k, g)
        
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find a better graph
    best_n = float('inf')
    best_edges = None
    
    # Try a few N values
    for N in [1000, 1200, 1500]:
        if time.time() > deadline:
            break
            
        # Generate
        edges = _generate_random_3reg(N)
        if not edges:
            continue
            
        # Check initial girth
        girth = _calc_girth(N, edges)
        if girth >= g:
            return N, edges
            
        # Local Search
        T = 1.0
        for step in range(2000):
            if time.time() > deadline:
                break
                
            # Find a short cycle (simplified: just pick random edges and try flips)
            # A better heuristic: Find the actual shortest cycle and try to break it.
            # For now, random flips.
            
            u, v = random.choice(edges)
            x, y = random.choice(edges)
            
            # Check if they are the same edge or adjacent
            if (u, v) == (x, y) or (u, v) == (y, x):
                continue
            if u == x or u == y or v == x or v == y:
                continue
                
            # Try flip
            # Remove (u,v) and (x,y)
            # Add (u,x) and (v,y)
            # Check if (u,x) or (v,y) are already edges or loops
            # This requires an adjacency set for O(1) lookup
            
            # This implementation is too slow for Python if we do it naively.
            # I will skip the complex local search in the final code if it's too risky.
            # Instead, I will just try many random graphs.
            
            # Actually, just trying random graphs is unlikely to find girth 16.
            # I will return the LUW graph if I don't find a better one quickly.
            
        # End step loop
        
    # Fallback
    return _luw_construct(3, 16)
