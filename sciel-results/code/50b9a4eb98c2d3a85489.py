import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if k != 3:
        return _lww_fallback(k, g)

    # Search for Generalized Petersen Graph G(n, k_gpg)
    # Note: using k_gpg to avoid confusion with degree k
    
    def check_girth_gpg(n, k_gpg, g):
        # Build adjacency list
        # Vertices: 0..n-1 (outer), n..2n-1 (inner)
        adj = [[] for _ in range(2*n)]
        for i in range(n):
            adj[i].append((i+1)%n)
            adj[i].append((i-1)%n)
            adj[i].append(n+i)
            
            adj[n+i].append(n+(i+k_gpg)%n)
            adj[n+i].append(n+(i-k_gpg)%n)
            adj[n+i].append(i)
            
        # Check girth using BFS from each vertex
        # We only need to check if girth < g
        for s in range(2*n):
            # BFS
            visited = [-1] * (2*n)
            parent = [-1] * (2*n)
            depth = [0] * (2*n)
            queue = [s]
            visited[s] = s
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = s
                        parent[v] = u
                        depth[v] = depth[u] + 1
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = depth[u] + depth[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Search n from Moore bound
    # Moore bound for k=3, g=16 is 510.
    # GPG has 2n vertices. So 2n >= 510 => n >= 255.
    for n in range(255, 1001):
        if time.time() > deadline:
            break
        for k_gpg in range(1, n//2 + 1):
            if time.time() > deadline:
                break
            if check_girth_gpg(n, k_gpg, g):
                # Construct edges
                edges = []
                for i in range(n):
                    edges.append((i, (i+1)%n))
                    edges.append((n+i, n+(i+k_gpg)%n))
                    edges.append((i, n+i))
                return 2*n, edges
                
    # Fallback to LWW
    return _lww_fallback(3, g)

def _lww_fallback(k, g):
    # Original LWW code
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
