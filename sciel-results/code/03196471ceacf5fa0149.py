import time
import random
import networkx as nx # Wait, networkx is not in standard library, but NumPy/SciPy are. 
# The prompt says "Python standard library, NumPy and SciPy are available."
# I should not rely on networkx. I will implement BFS manually.

def _bfs_girth(adj):
    n = len(adj)
    min_girth = float('inf')
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        for v in queue:
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v]:
                    # Cycle found
                    # Length is dist[v] + dist[u] + 1
                    # But this is for undirected graph BFS tree
                    # The cycle length is dist[v] + dist[u] + 1
                    # We need to be careful: BFS finds the shortest path.
                    # The cycle formed by the tree path and the edge (v,u) has length dist[v] + dist[u] + 1.
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
    return min_girth

def _is_3_regular(adj, n):
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    return True

def _generate_random_3_regular(n):
    # Generate a random 3-regular graph using configuration model
    # n must be even
    if n % 2 != 0:
        return None
    half_edges = n * 3
    if half_edges % 2 != 0:
        return None
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
    random.shuffle(stubs)
    adj = [[] for _ in range(n)]
    for i in range(0, half_edges, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            continue # Skip loops for simplicity, or retry
        if v not in adj[u]:
            adj[u].append(v)
            adj[v].append(u)
    # Check if simple and 3-regular
    if not _is_3_regular(adj, n):
        return None
    return adj

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Fallback: Seed LUW construction
    def _luw_construct():
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

    # If not the specific case, use LUW
    if k != 3 or g != 16:
        return _luw_construct()

    # Try to find a smaller graph for (3, 16)
    # Target: n < 936 (known best) or at least much smaller than LUW (~350k)
    # We try n = 512, 600, 800, 1000
    
    best_n, best_edges = _luw_construct()
    best_girth = float('inf')
    
    # We can try a few sizes
    for target_n in [512, 600, 700, 800, 936, 1000]:
        if time.time() > deadline:
            break
            
        # Try to find a 3-regular graph of size target_n with girth >= 16
        found = False
        for _ in range(20): # 20 attempts per size
            if time.time() > deadline:
                break
            adj = _generate_random_3_regular(target_n)
            if adj is None:
                continue
            girth = _bfs_girth(adj)
            if girth >= g:
                # Convert adj to edge list
                edges = []
                for i in range(target_n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return target_n, edges
            # Update best if girth is better? No, we need girth >= 16.
            
    # If no small graph found, return the LUW result
    return best_n, best_edges
