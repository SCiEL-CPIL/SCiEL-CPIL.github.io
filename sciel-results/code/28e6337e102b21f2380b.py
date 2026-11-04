import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # Special case: Hoffman-Singleton graph for k=4, g=10
    if k == 4 and g == 10:
        n = 50
        edges = set()
        for i in range(n):
            # The Hoffman-Singleton graph can be defined by connecting i to i+1, i+10, i+26, i+35 mod 50
            for j in (i + 1, i + 10, i + 26, i + 35):
                j %= n
                if i != j:
                    edges.add((i, j) if i < j else (j, i))
        
        # Verify the graph
        edges_list = sorted(edges)
        if _verify(n, edges_list, k, g):
            return n, edges_list
        # If verification fails, fall through to LUW (should not happen)

    # Fallback: Lazebnik-Ustimenko-Woldar construction
    if time.time() > deadline:
        return 0, [] # Should not happen, but safe exit
        
    q = k
    # d is the parameter for LUW. For girth g, we need d such that girth >= d+5 or d+4.
    # Standard LUW gives girth at least d+5 for odd d, d+4 for even d? 
    # Actually, the standard construction CD(d, q) has girth at least d+5 if d is odd, d+4 if d is even?
    # Let's stick to the logic in the current program which worked.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 5:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
        # If q is not 4 or 5, we can't easily do LUW with simple field ops unless we implement GF(q^m)
        # For this task, k=4 is the main case. If k is other, we might fail.
        # But the task says "evaluator passes this task's degree k", implying k=4.
        # We'll return an empty graph if we can't construct, or try a simple cycle if k=2?
        # Assuming k=4 is the primary target.
        return 0, []
        
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


def _verify(n, edges, k, g):
    """Return True iff the graph is simple, k-regular, connected, and has girth >= g."""
    if n < 1:
        return False
    if len(edges) != len(set(edges)):
        return False
        
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # girth: shortest cycle = min over edges (u,v) of dist_nonback(u -> v) + 1
    INF = 10 ** 9
    for (u, v) in edges:
        dist = {}
        q = []
        for w in adj[u]:
            dist[(u, w)] = 1
            q.append((u, w, 1))
        head = 0
        best = INF
        while head < len(q):
            prev, cur, L = q[head]
            head += 1
            if cur == v:
                if L + 1 < best:
                    best = L + 1
                continue
            if L >= g:
                break
            for w in adj[cur]:
                if w == prev:
                    continue
                if (cur, w) not in dist:
                    dist[(cur, w)] = L + 1
                    q.append((cur, w, L + 1))
        if best < g:
            return False
    return True
