import math
from collections import deque

def construct(k, g):
    # Moore Bound calculation
    def calc_moore(k, g):
        r = (g - 1) // 2
        total_verts = 1
        for i in range(r):
            if i == 0:
                total_verts += k
            else:
                total_verts += k * (k-1)**i
        return total_verts

    M = calc_moore(k, g)
    
    # Search for a circulant graph
    # Degree k=5.
    # Neighbors: a, -a, b, -b, c.
    # For simple graph, no 0 in S.
    # c must satisfy c = -c mod n => 2c = 0 mod n.
    # If n is even, c = n//2.
    # If n is odd, c=0 (loop), invalid. So n must be even.
    
    # We will search n from M to M + 2000
    # For each even n, try pairs (a, b)
    
    def check_girth(adj, n, g):
        # BFS from each node to find shortest cycle
        # A graph has girth < g if there is a cycle of length < g.
        # We can check by removing each edge and checking distance.
        # Or simpler: BFS from each node, if we encounter a node at depth d 
        # that was visited at depth d' with d + d' + 1 < g? No.
        # Standard girth check:
        # For each edge (u, v), check if dist(u, v) in G \ {(u,v)} < g - 1.
        
        # Optimization: Only check if n is small enough.
        # Since we start from M=428, n will be around 400-1000.
        # Edges = 5*n/2 ~ 1000-2500.
        # BFS from each endpoint of each edge is O(E * (V+E)) ~ 2500 * 3000 = 7.5M ops.
        # This is fast in Python if optimized or using NumPy?
        # Pure Python BFS might be slow.
        # Let's use a heuristic: Check for cycles of length 3, 4, 5, 6, 7, 8.
        # If none, girth >= 9.
        
        # Check triangles (3)
        for i in range(n):
            for j in adj[i]:
                if j <= i: continue
                for k in adj[i]:
                    if k == j: continue
                    if k in adj[j]:
                        return False
        # Check 4-cycles
        # ... this gets complex.
        
        # Let's use the edge-removal BFS but optimized.
        # We only need to check up to depth g//2 = 4.
        # If we find a path of length < g-1 between u and v, we have a cycle < g.
        
        for u in range(n):
            for v in adj[u]:
                # BFS from u to v, avoiding edge (u,v)
                # Max depth g-2
                visited = [-1] * n
                visited[u] = 0
                q = deque([u])
                found = False
                while q:
                    cur = q.popleft()
                    if cur == v:
                        return False
                    if visited[cur] >= g - 2:
                        continue
                    for nxt in adj[cur]:
                        if visited[nxt] == -1 and not (cur == u and nxt == v) and not (cur == v and nxt == u):
                            visited[nxt] = visited[cur] + 1
                            q.append(nxt)
        return True

    # Search
    max_n = M + 2000
    for n in range(M, max_n + 1):
        if n % 2 != 0:
            continue # c must be n/2, so n even
        
        c = n // 2
        # a, b in 1..c-1
        # Ensure a != b, a != -b, etc.
        # a, -a, b, -b, c must be distinct.
        # a != 0, a != c (since a < c)
        # b != 0, b != c
        # a != b
        # a != -b => a + b != 0 => a + b != n (impossible since a,b < n/2)
        # -a != b => a + b != n
        # -a != -b => a != b
        
        for a in range(1, c):
            for b in range(a + 1, c): # b > a to avoid duplicates and a==b
                # Check distinctness
                # S = {a, n-a, b, n-b, c}
                # All distinct if a != b, a != c, b != c, a != n-a (a!=0, a!=c), b != n-b.
                # Since 1 <= a < b < c, they are distinct.
                
                adj = [[] for _ in range(n)]
                # Add edges
                for i in range(n):
                    neighbors = [
                        (i + a) % n,
                        (i - a) % n,
                        (i + b) % n,
                        (i - b) % n,
                        (i + c) % n
                    ]
                    # Remove self-loops (shouldn't happen if a,b,c != 0)
                    # a,b,c are in 1..c-1, c=n/2.
                    # i+a mod n == i? a=0. No.
                    # i+c mod n == i? c=0. No.
                    
                    for v in neighbors:
                        adj[i].append(v)
                
                # Check for duplicate edges (multi-edges)
                # This happens if a == -b mod n? No, we ensured distinct.
                # But (i+a) == (i-b) => a+b=0 => a+b=n. Impossible.
                # So simple graph.
                
                # Check Girth
                if check_girth(adj, n, g):
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
    
    # Fallback: Return the algebraic construction from seed if not found
    # (Re-implementing the seed fallback as a last resort)
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
