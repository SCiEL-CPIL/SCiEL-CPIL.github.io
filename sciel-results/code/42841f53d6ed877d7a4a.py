import random
import time
import sys

def _random_3_regular(n):
    """Generate a random 3-regular simple graph on n vertices."""
    if n < 4 or n % 2 != 0:
        return None
    while True:
        # Configuration model
        half_edges = []
        for i in range(n):
            half_edges.extend([i, i, i])
        random.shuffle(half_edges)
        edges = []
        seen = set()
        valid = True
        for i in range(0, 2 * n * 3 // 2, 2): # 3n is even, so 3n/2 pairs
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in seen or (v, u) in seen:
                valid = False
                break
            edges.append((u, v))
            seen.add((u, v))
            seen.add((v, u))
        if valid:
            return edges

def _has_short_cycle(edges, n, g):
    """Check if graph has a cycle of length < g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS from each vertex
    # We need to detect cycles of length < g.
    # If we find a back-edge to a node with depth difference < g, we have a short cycle.
    # Actually, just checking if any cycle is found with length < g.
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # Calculate length
                    # The cycle is u - ... - v - u?
                    # Path from start to u, path from start to v.
                    # LCA is needed for exact length, but for "is there a short cycle",
                    # we can just check if dist[u] + dist[v] + 1 < g?
                    # No, the cycle length is dist[u] + dist[v] + 1 - 2*dist[lca].
                    # Since dist[lca] >= 0, cycle length <= dist[u] + dist[v] + 1.
                    # If dist[u] + dist[v] + 1 < g, it's definitely a short cycle.
                    # If dist[u] + dist[v] + 1 >= g, it might still be short if lca is close to u or v.
                    # To be safe and simple, we can just run BFS to depth g/2 and see if we hit a visited node.
                    # If we hit a visited node at depth d, and the other node was at depth d', 
                    # the cycle length is at most d + d' + 1.
                    # If we restrict BFS to depth D = g//2 - 1, any cycle found has length <= 2D + 1 = g-1.
                    # So if we find ANY cycle within depth g//2 - 1, it is a short cycle.
                    if dist[u] + dist[v] + 1 < g:
                        return True
        # Optimization: If max dist reached is < g//2, we didn't explore deep enough?
        # No, BFS explores all reachable nodes.
        
    return False

# Wait, the BFS above might be slow if done naively.
# And checking `dist[u] + dist[v] + 1 < g` is a sufficient condition for a short cycle,
# but is it necessary?
# If a cycle of length L < g exists, there is a vertex on the cycle such that the distance from start to it is small?
# Actually, if we run BFS from *every* vertex, we will find the cycle.
# For a cycle of length L, pick a vertex v on it. The distance from v to the other end of the cycle is L-1.
# If we run BFS from v, we will see the neighbor of v on the cycle at distance L-1.
# If L-1 < g, we detect it.
# So yes, BFS from every vertex with early termination if a back-edge is found with length < g works.

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Try random search for small n
    # Moore bound for k=3, g=16 is 510.
    # Try n from 512 to 2000 step 2.
    # Limit number of attempts per n to avoid getting stuck.
    
    best_n = None
    best_edges = None
    
    # Fallback: LUW construction
    def luw_construct():
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

    # If k != 3 or g != 16, just use LUW
    if k != 3 or g != 16:
        return luw_construct()
        
    # Random search for n(3,16)
    # Start from Moore bound 510. Next even is 512.
    for n in range(512, 1000, 2):
        if time.time() > deadline:
            break
        # Try a few random graphs for this n
        for attempt in range(20):
            if time.time() > deadline:
                break
            edges = _random_3_regular(n)
            if edges is None:
                continue
            if not _has_short_cycle(edges, n, g):
                return n, edges
                
    # If no small graph found, return LUW
    return luw_construct()
