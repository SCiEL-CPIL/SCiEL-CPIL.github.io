import time
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Target n based on known best upper bound
    # For k=4, g=10, best known is 320.
    # We search for a circulant graph C_n({±a, ±b}) with girth >= g
    
    def is_circulant_valid(n, a, b, g):
        # Check girth of circulant graph C_n(±a, ±b)
        # Build adjacency
        adj = [[] for _ in range(n)]
        for i in range(n):
            adj[i].append((i + a) % n)
            adj[i].append((i - a) % n)
            adj[i].append((i + b) % n)
            adj[i].append((i - b) % n)
        
        # Check for self-loops or multi-edges
        # Simple graph check
        if a == 0 or b == 0 or a == b or a == n - b or a == n - a:
             return False
        # Actually, steps must be distinct mod n.
        steps = {a, n-a, b, n-b}
        if len(steps) < 4:
            return False
            
        # Girth check using BFS from each node
        # Optimization: BFS from 0 is not enough, need min over all.
        # But for circulant, symmetry allows checking from 0?
        # No, girth is global min. But due to symmetry, the shortest cycle 
        # might not pass through 0? 
        # Actually, in a vertex-transitive graph (circulant is), 
        # the girth is the shortest cycle through any vertex.
        # So BFS from 0 is sufficient to find the girth!
        
        visited = [-1] * n
        parent = [-1] * n
        queue = [0]
        visited[0] = 0
        dist = [0] * n
        # BFS
        head = 0
        min_cycle = float('inf')
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = u
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    # Length is dist[u] + dist[v] + 1
                    # But this finds *a* cycle, not necessarily the shortest.
                    # Standard BFS girth:
                    # When we visit a neighbor v that is already visited:
                    # If v is not parent of u, we have a cycle.
                    # The length is dist[u] + dist[v] + 1.
                    # We take the min over all such encounters.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        
        return min_cycle >= g

    # Search for n in range
    # Start with 320
    for n in range(320, 350):
        if time.time() > deadline:
            break
        # a, b must be such that steps are distinct
        # 1 <= a < b < n/2
        for a in range(1, n // 2):
            if time.time() > deadline:
                break
            for b in range(a + 1, n // 2):
                if is_circulant_valid(n, a, b, g):
                    # Construct edges
                    edges = []
                    for i in range(n):
                        edges.append((i, (i + a) % n))
                        edges.append((i, (i - a) % n))
                        edges.append((i, (i + b) % n))
                        edges.append((i, (i - b) % n))
                    # Remove duplicates (undirected)
                    # The above adds (i, j) and (j, i) implicitly if we iterate all i.
                    # We need a unique list of edges.
                    unique_edges = set()
                    for i in range(n):
                        for j in [(i + a) % n, (i - a) % n, (i + b) % n, (i - b) % n]:
                            if i < j:
                                unique_edges.add((i, j))
                            elif i > j:
                                unique_edges.add((j, i))
                    return n, list(unique_edges)
                    
    # Fallback: LUW Graph
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
