import time
import sys
import math
import collections

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety

    # Fallback: LUW construction
    q = k
    # Calculate d for the seed's logic
    # For k=3, g=16, d needs to be large enough.
    # d >= g/2 is a rough estimate.
    d_seed = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    best_n = None
    best_edges = None

    # 1. Search for Generalized Petersen Graphs GP(n, k_gp)
    # Vertices: 2n. 
    # Moore bound for k=3, g=16 is 512. So 2n >= 512 => n >= 256.
    # We search n from 256 to 600.
    # We prefer even n to ensure bipartiteness (if k_gp is odd), which helps girth.
    
    max_n_search = 600
    
    for n in range(256, max_n_search + 1, 2):
        if time.time() > deadline - 5:
            break
            
        # k_gp must be coprime to n for the inner cycle to be a single cycle of length n.
        # If gcd(n, k_gp) > 1, the inner graph splits into cycles of length n/gcd.
        # If n/gcd < 16, it's a short cycle. 
        # So we require n/gcd(n, k_gp) >= 16.
        
        for k_gp in range(1, n // 2 + 1):
            if time.time() > deadline - 2:
                break
                
            g_c = math.gcd(n, k_gp)
            inner_cycle_len = n // g_c
            if inner_cycle_len < 16:
                continue
                
            # Construct GP(n, k_gp)
            # Vertices 0..n-1 are outer u_i, n..2n-1 are inner v_i
            edges = []
            adj = [[] for _ in range(2 * n)]
            
            for i in range(n):
                u_i = i
                u_next = (i + 1) % n
                edges.append((u_i, u_next))
                adj[u_i].append(u_next)
                adj[u_next].append(u_i)
                
                v_i = n + i
                edges.append((u_i, v_i))
                adj[u_i].append(v_i)
                adj[v_i].append(u_i)
                
                v_next = n + (i + k_gp) % n
                edges.append((v_i, v_next))
                adj[v_i].append(v_next)
                adj[v_next].append(v_i)
            
            # Check girth
            if _check_girth(2 * n, adj, g):
                best_n = 2 * n
                best_edges = edges
                # Since we iterate n upwards, the first valid one is the smallest.
                return best_n, best_edges
                
    # 2. Fallback to LUW if no GP found
    n_seed, edges_seed = build_luw(d_seed)
    return n_seed, edges_seed

def _check_girth(n, adj, min_girth):
    """
    Check if girth of graph is at least min_girth.
    Uses BFS from each vertex, limiting depth to min_girth // 2.
    """
    max_depth = min_girth // 2 - 1
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = collections.deque([start])
        
        while queue:
            curr = queue.popleft()
            if dist[curr] > max_depth:
                continue
                
            for neighbor in adj[curr]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[curr] + 1
                    parent[neighbor] = curr
                    queue.append(neighbor)
                elif neighbor != parent[curr]:
                    # Found a back edge
                    cycle_len = dist[curr] + dist[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True
