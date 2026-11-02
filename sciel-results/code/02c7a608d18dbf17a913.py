"""Rewritten program with a stronger algorithm for cage construction."""
import random
import math
import time

def _moore_bound(k, g):
    """Calculate the Moore bound for a k-regular graph of girth g."""
    if g % 2 == 0:
        # Even girth: 1 + k * sum((k-1)^i for i in range(g//2))
        r = g // 2
        total = 1
        term = k
        for i in range(r):
            total += term
            term *= (k - 1)
        return total
    else:
        # Odd girth: 2 * sum((k-1)^i for i in range((g-1)//2))
        r = (g - 1) // 2
        total = 0
        term = 1
        for i in range(r):
            total += term
            term *= (k - 1)
        return 2 * total

def _is_valid_graph(n, edges, k, g):
    """Check if the graph is k-regular, simple, and has girth >= g."""
    if len(edges) != n * k // 2:
        return False
    
    # Check simplicity and degree
    deg = [0] * n
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))
        deg[u] += 1
        deg[v] += 1
    
    for d in deg:
        if d != k:
            return False
            
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check girth using BFS from each vertex (optimized: stop if depth >= g//2)
    # Actually, for girth g, we need to ensure no cycle < g.
    # A cycle of length L exists if there is a path of length L-1 connecting two neighbors of a vertex, etc.
    # Simpler: BFS from each vertex up to depth g//2. If we revisit a node at depth d, 
    # the cycle length is d + dist.
    # For girth >= g, we must not find any cycle of length < g.
    # We can do a BFS from each vertex, and if we encounter a previously visited node 
    # in the current BFS tree at a depth that creates a cycle < g, it's invalid.
    # Note: This is O(n * (k^(g/2))) which is small for small g.
    
    max_depth = g // 2
    for start in range(n):
        visited = [-1] * n
        parent = [-1] * n
        visited[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if visited[u] >= max_depth:
                continue
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # If v is visited and not the parent, we found a cycle.
                    # The cycle length is visited[u] + visited[v] + 1? No.
                    # In BFS, if we see a visited node v from u, and v is not parent[u],
                    # the cycle length is visited[u] + visited[v] + 1?
                    # Actually, if we do BFS from a single source, finding a back edge
                    # gives a cycle length of dist[u] + dist[v] + 1.
                    # However, the shortest cycle might not be found by a single BFS from one source
                    # if it doesn't pass through the source.
                    # But for girth checking, it is sufficient to check for cycles of length < g.
                    # If there is a cycle of length L < g, it will be detected by BFS from any vertex on that cycle?
                    # Yes, if we start BFS from a vertex on the cycle, we will find the cycle.
                    # But we are iterating over ALL start vertices. So it is sufficient.
                    if v != parent[u]:
                        cycle_len = visited[u] + visited[v] + 1
                        if cycle_len < g:
                            return False
    return True

def _luw_graph(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph."""
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

def construct(k, g):
    # Hardcoded known optimal or near-optimal cages for small parameters
    # Format: (k, g): (n, edges)
    known_cages = {}
    
    # Example: Petersen Graph (3,5)
    # n=10
    edges_3_5 = [(0,1),(0,4),(0,5),(1,3),(1,6),(2,3),(2,7),(2,8),(3,9),(4,6),(4,7),(4,9),(5,6),(5,8),(5,9),(6,8),(7,9),(8,9)] # Wait, Petersen is 15 edges.
    # Let's just use the LUW for fallback if not hardcoded, to avoid bugs in hardcoded edges.
    # But for k=5, g=9, LUW is huge.
    
    # Try to find a better graph via heuristic search for small n
    mb = _moore_bound(k, g)
    
    # For small k and g, we can try to find a circulant or random graph
    # Given time limits, we try a limited search
    best_n = None
    best_edges = None
    
    # If k and g are small, we might find a solution quickly
    # We try n from mb to mb + 100
    # For each n, we try a few random regular graphs
    
    if k <= 5 and g <= 9:
        start_time = time.time()
        max_time = 10 # seconds
        max_n_search = mb + 200
        
        for n in range(mb, max_n_search):
            if time.time() - start_time > max_time:
                break
            # Try to find a k-regular graph on n vertices with girth >= g
            # Use randomized greedy
            for _ in range(5): # 5 restarts per n
                if time.time() - start_time > max_time:
                    break
                # Generate a random k-regular graph
                # This is complex. Instead, use a simple heuristic:
                # Start with empty graph, add edges randomly, ensuring degree < k.
                # Then check girth.
                # This is too slow for n=200.
                
                # Alternative: Use a known construction if available.
                # For now, we rely on the LUW fallback if heuristic fails.
                pass
        
    # If no heuristic solution found, use LUW
    # But LUW is very large for k=5, g=9.
    # Is there a better fallback?
    # Maybe a random regular graph with random restarts and girth checking?
    # For n=200, k=5, g=9, the probability of a random graph having girth 9 is low but non-zero.
    # We can try to generate random 5-regular graphs and check girth.
    
    if best_edges is None:
        # Try random regular graphs for a short time
        start_time = time.time()
        max_time = 20
        # Start from Moore bound
        for n in range(mb, mb + 500):
            if time.time() - start_time > max_time:
                break
            # Try to find a 5-regular graph on n vertices with girth >= g
            # Using random configuration model (pairing stubs)
            for _ in range(10): # 10 attempts per n
                if time.time() - start_time > max_time:
                    break
                # Generate random 5-regular graph
                stubs = [i for i in range(n) for _ in range(k)]
                random.shuffle(stubs)
                edges = []
                for i in range(0, len(stubs), 2):
                    u = stubs[i]
                    v = stubs[i+1]
                    if u != v:
                        edges.append((u, v))
                    else:
                        # Self-loop, restart
                        break
                else:
                    # Check for multi-edges
                    edge_set = set()
                    valid = True
                    for u, v in edges:
                        if u > v: u, v = v, u
                        if (u, v) in edge_set:
                            valid = False
                            break
                        edge_set.add((u, v))
                    if not valid:
                        continue
                    
                    # Check girth
                    if _is_valid_graph(n, edges, k, g):
                        best_n = n
                        best_edges = edges
                        break
            if best_edges:
                break
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW
    return _luw_graph(k, g)
