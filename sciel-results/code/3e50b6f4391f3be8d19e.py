import time
import random
import sys

def _is_valid_4reg_girth(n, edges, g):
    """
    Checks if the graph is 4-regular and has girth at least g.
    """
    # Check 4-regular
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    if any(d != 4 for d in degree):
        return False
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for simple graph
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        pair = (min(u, v), max(u, v))
        if pair in edge_set:
            return False
        edge_set.add(pair)

    # Check girth >= g
    # BFS from each vertex up to depth (g-1)//2
    depth_limit = (g - 1) // 2
    
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        for _ in range(depth_limit):
            new_queue = []
            for u in queue:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        new_queue.append(v)
            queue = new_queue
        
        for u, v in edges:
            du = dist[u]
            dv = dist[v]
            if du != -1 and dv != -1:
                if du + dv + 1 < g:
                    return False
    return True

def _generate_random_4reg(n):
    """
    Generates a random 4-regular graph on n vertices.
    """
    if n < 4 or n % 2 != 0: # Note: 4-regular graphs can have odd n (e.g. K5), but configuration model is easier for even n? No, 4n is even, so n can be odd.
        # Actually, configuration model works for any n where 4n is even.
        pass
        
    for attempt in range(100):
        stubs = []
        for i in range(n):
            for _ in range(4):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        edges = []
        adj = [[] for _ in range(n)]
        valid = True
        
        for i in range(0, 4 * n, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            edges.append((u, v))
            adj[u].append(v)
            adj[v].append(u)
            
        if valid:
            return edges
    return None

def _luw_construct():
    """
    Constructs the known (4,10)-cage with 320 vertices.
    """
    q = 4
    d = 5 # g=10 -> d=5
    
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: table[x][y]
    
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
    """
    Constructs a k-regular graph with girth at least g.
    """
    if k != 4 or g != 10:
        raise NotImplementedError("Only k=4, g=10 is supported")
        
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strategy:
    # 1. Try to find a 4-regular graph with girth 5 on small n (starting from Moore bound 17).
    # 2. If found, return its bipartite double cover (girth 10, size 2n).
    # 3. Fallback to LUW construction (size 320).
    
    # Moore bound for k=4, g=5 is 17.
    # We try n from 17 to 40.
    # For each n, we try a few random graphs.
    
    best_n = 320
    best_edges = None
    
    # Pre-compute LUW as fallback
    n_luw, edges_luw = _luw_construct()
    best_n = n_luw
    best_edges = edges_luw
    
    # Search for smaller
    # The (4,5)-cage is known to have 16 vertices? No, Moore bound is 17.
    # Let's try n=17 to 30.
    for n in range(17, 31):
        if time.time() > deadline:
            break
            
        # Try a few random graphs
        for _ in range(20):
            if time.time() > deadline:
                break
            edges = _generate_random_4reg(n)
            if edges is None:
                continue
            if _is_valid_4reg_girth(n, edges, 5):
                # Found a graph with girth 5
                # Construct double cover
                n_double = 2 * n
                edges_double = []
                for u, v in edges:
                    # (u, 0) -- (v, 1)
                    edges_double.append((u, n + v))
                    # (v, 0) -- (u, 1)
                    edges_double.append((v, n + u))
                
                # Verify the double cover has girth 10
                if _is_valid_4reg_girth(n_double, edges_double, 10):
                    return n_double, edges_double
                    
    return best_n, best_edges
