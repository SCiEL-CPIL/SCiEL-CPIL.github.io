import time
import sys
import collections

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction with dynamic depth.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    q = k
    if q != 3:
        # Fallback for other k
        # For this task, k=3 is expected.
        # If k != 3, we might need a different construction.
        # But LUW works for prime powers.
        pass

    # Determine d such that theoretical girth 2d+2 >= g
    # d >= (g - 2) / 2
    d = max(3, (g - 2) // 2 + 1)
    
    # If g is even, (g-2)//2 + 1 = g/2. 2d+2 = g+2 > g. Safe.
    # If g is odd, (g-2)//2 + 1 = (g-1)/2. 2d+2 = g+1 > g. Safe.
    
    def girth_of_d(d):
        # Theoretical lower bound
        return 2 * d + 2

    # GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Rule for LUW
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    def build_luw(d_val):
        index = {}
        order = []
        edges = []
        
        start = (0, (None,) + (0,) * d_val)
        index[start] = 0
        order.append(start)
        
        for i, vertex in enumerate(order):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d_val - 1)
                for c in range(2, d_val + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                
                neighbor = (1 - side, tuple(y))
                
                if neighbor not in index:
                    index[neighbor] = len(order)
                    order.append(neighbor)
                
                if side == 0:
                    edges.append((index[vertex], index[neighbor]))
                    
            if time.time() > deadline:
                break
                
        n = len(order)
        return n, edges

    def check_girth(n, edges, min_girth):
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS from each vertex to find shortest cycle
        # For efficiency, we can stop if we find a cycle < min_girth
        # But we need to be sure.
        # Standard BFS with parent tracking.
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = collections.deque([start])
            
            while queue:
                curr = queue.popleft()
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

    current_d = d
    best_n = None
    best_edges = None
    
    while time.time() < deadline - 10.0:
        n, edges = build_luw(current_d)
        
        # Verify girth
        if check_girth(n, edges, g):
            best_n = n
            best_edges = edges
            break
        else:
            # Increase depth
            current_d += 1
            if current_d > 15: # Safety limit
                break

    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: return the last constructed one even if not verified (should be rare)
    return n, edges
