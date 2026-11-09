import random
import time

def construct(k, g):
    start_time = time.time()
    deadline = 110.0
    
    # Try random 3-regular graphs
    n_target = 512
    # Moore bound for k=3, g=16 is 510.
    # Try n = 512, 514, ...
    
    def generate_random_3reg(n):
        if n % 2 != 0:
            return None
        # Configuration model
        half_edges = n * 3 // 2
        # Each vertex has 3 stubs
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        edges = []
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u != v:
                edges.append((u, v))
            else:
                # Self-loop, regenerate? Or just swap?
                # For simplicity, if self-loop, swap with a neighbor
                if i + 2 < len(stubs):
                    stubs[i+1], stubs[i+2] = stubs[i+2], stubs[i+1]
                    edges.append((u, stubs[i+1]))
        # Check for multi-edges
        edge_set = set()
        valid_edges = []
        for u, v in edges:
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                edge_set.add((u, v))
                valid_edges.append((u, v))
            else:
                return None # Multi-edge found
        # Check degrees
        deg = [0] * n
        for u, v in valid_edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            return None
        return valid_edges

    def check_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            # BFS
            visited = [-1] * n
            parent = [-1] * n
            queue = [(start, 0)]
            visited[start] = 0
            while queue:
                u, dist = queue.pop(0)
                if dist >= g // 2:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = dist + 1
                        parent[v] = u
                        queue.append((v, dist + 1))
                    else:
                        if parent[u] != v and parent[v] != u:
                            # Cycle found
                            # Calculate length
                            # This is a bit complex, just return False if we see a back edge
                            # Actually, BFS finds shortest path. If we see a visited node not parent, it's a cycle.
                            # Length is dist + 1 + (dist from start to v)
                            # But we need exact girth.
                            pass
            # Simplified: Just check if any cycle < g
            # This function is a stub. I need a proper girth checker.
            pass
        return True

    # Proper girth checker
    def get_girth(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_girth = float('inf')
        for start in range(n):
            # BFS from start
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                if dist[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if parent[u] != v:
                            # Cycle
                            # Find LCA
                            # Simplified: if dist[u] + dist[v] + 1 < min_girth
                            # This is not accurate for all cases, but good enough for heuristic
                            # Actually, BFS from each node finds the shortest cycle passing through that node?
                            # No, BFS from start finds shortest path from start to v.
                            # If we see a back edge, the cycle length is dist[u] + dist[v] + 1.
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
        return min_girth

    # Try random graphs
    for n_try in range(512, 600, 2):
        if time.time() > deadline:
            break
        for _ in range(10):
            if time.time() > deadline:
                break
            edges = generate_random_3reg(n_try)
            if edges is None:
                continue
            girth = get_girth(n_try, edges)
            if girth >= g:
                return n_try, edges

    # Fallback to LUW
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw
