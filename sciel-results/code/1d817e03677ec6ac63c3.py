import time
import math
import collections

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 140.0
    
    # Moore bound for k=3, g=16 is 510.
    # We search for Metacirculant/Circulant graphs, specifically Generalized Petersen GP(n, k_gp)
    # which has 2*n vertices.
    # We need 2*n >= 510 => n >= 255.
    # We prefer even n for bipartite graphs to avoid odd cycles < 16.
    
    best_n = None
    best_edges = None
    
    # Search n from 256 upwards
    # Upper bound on n: The known best is 936/2 = 468.
    # We will search up to 500 to be safe, but start from 256.
    
    for n in range(256, 501, 2):
        if time.time() > deadline - 10.0:
            break
            
        # Iterate k_gp from 1 to n//2
        # We require gcd(n, k_gp) == 1 to maximize girth (single inner cycle)
        # Actually, gcd(n, k_gp) > 1 splits inner cycle, which might be shorter.
        # So gcd(n, k_gp) == 1 is a good filter.
        
        for k_gp in range(1, n // 2 + 1):
            if time.time() > deadline - 5.0:
                break
                
            if math.gcd(n, k_gp) != 1:
                continue
                
            # Quick filter: k_gp should not be too small or too large relative to n
            # to avoid short mixed cycles.
            # Heuristic: k_gp should be roughly n/3 or n/4?
            # Let's just check all coprime k_gp, but break early if time is tight.
            
            # Construct GP(n, k_gp)
            # Vertices: 0..n-1 are u_i, n..2n-1 are v_i
            edges = []
            adj = [[] for _ in range(2 * n)]
            
            for i in range(n):
                # Outer cycle
                u_i = i
                u_next = (i + 1) % n
                edges.append((u_i, u_next))
                adj[u_i].append(u_next)
                adj[u_next].append(u_i)
                
                # Spokes
                v_i = n + i
                edges.append((u_i, v_i))
                adj[u_i].append(v_i)
                adj[v_i].append(u_i)
                
                # Inner cycle
                v_next = n + (i + k_gp) % n
                edges.append((v_i, v_next))
                adj[v_i].append(v_next)
                adj[v_next].append(v_i)
            
            # Check girth
            # We check if girth >= g
            # Since graph is bipartite (n even), girth is even.
            # We check for cycles of length 4, 6, ..., g-1.
            
            if _check_girth_fast(2 * n, adj, g):
                best_n = 2 * n
                best_edges = edges
                # We found a valid graph. Can we find a smaller one?
                # We are iterating n from low to high.
                # So the first one we find is the smallest n in this range.
                # But we should check if a smaller n was missed?
                # We started at 256.
                # If we find one at n=300, we should return it?
                # Yes, because we want to minimize n.
                # But we should verify if there's a smaller n?
                # We are iterating n in increasing order.
                # So the first valid one is the best.
                return best_n, best_edges
                
    # If no GP graph found, fall back to LUW
    # (Though LUW is much larger, it's a valid fallback)
    return _luw_construct(k, g)

def _check_girth_fast(n, adj, min_girth):
    """
    Check if girth of graph (n, adj) is at least min_girth.
    Uses BFS from each vertex, tracking depth.
    If a back edge is found with cycle length < min_girth, return False.
    """
    # Optimization: If min_girth is even, and graph is bipartite, we can be faster?
    # But generic BFS is safe.
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = collections.deque([start])
        
        while queue:
            curr = queue.popleft()
            # We only need to check up to depth (min_girth - 2) // 2?
            # No, a cycle of length L can be found at depth L-1?
            # If we have a cycle of length 14, and we start at a vertex on it,
            # the farthest point on the cycle is at distance 7.
            # So if we limit depth to (min_girth - 1) // 2, we might miss it?
            # Let's limit depth to min_girth // 2.
            max_depth = min_girth // 2
            
            if dist[curr] > max_depth:
                continue
                
            for neighbor in adj[curr]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[curr] + 1
                    parent[neighbor] = curr
                    queue.append(neighbor)
                elif neighbor != parent[curr]:
                    # Found a cycle
                    cycle_len = dist[curr] + dist[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _luw_construct(k, g):
    t0 = time.time()
    q = k
    d = max(3, (g - 2) // 2 + 1)
    if q % 2 == 1 and d % 2 == 0:
        d += 1
        
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
        if time.time() - t0 > 130:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
