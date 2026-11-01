import time
import math

def construct(k, g):
    start = time.time()
    deadline = start + 135.0

    def time_left():
        return deadline - time.time()

    def build_luw(q, d_val):
        # Lazebnik-Ustimenko-Woldar construction
        if q == 4:
            mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def F(a, b):
                if a == 0 or b == 0:
                    return 0
                return mul_table[a][b]
        else:
            def F(a, b):
                return (a * b) % q

        # Rule for LUW
        if d_val == 2:
            rule = {2: (1,1)}
        elif d_val == 3:
            rule = {2: (1,1), 3: (2,1)}
        elif d_val == 4:
            rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        else:
            rule = {2: (1,1), 3: (2,1), 4: (1,2)}
            for c in range(5, d_val + 1):
                rc = (c - 5) % 4
                c0 = c - rc
                rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][rc]

        def neighbors(v):
            side, x = v
            out = []
            for first in range(q):
                y = [None] + list(x)
                y[1] = first
                if side == 0:
                    for c in range(2, d_val + 1):
                        a, b = rule[c]
                        y[c] = (y[a] * x[b]) if q == 4 else (y[a] * x[b]) % q
                        # For q=4 use XOR/add? No, LUW uses field operations.
                        # Let's use a generic field for q prime, and for q=4 use GF(4).
                else:
                    for c in range(2, d_val + 1):
                        a, b = rule[c]
                        # sub is same as add in GF(4) if char 2, or (x-y)%q
                        y[c] = (x[c] - x[a] * y[b]) % q if q != 4 else (x[c] ^ x[a] * y[b])
                out.append((1 - side, tuple(y)))
            return out

        # For GF(4), need proper arithmetic
        if q == 4:
            # GF(4): 0,1,2,3 with alpha^2=alpha+1 (i.e., 2^2=2+1=3? No, alpha^2=alpha+1 => if 2=alpha, 3=alpha+1, then 2^2=3, 2*3=2*(alpha+1)=alpha^2+alpha=3+2=1)
            # Multiplication table:
            #   0 1 2 3
            # 0 0 0 0 0
            # 1 0 1 2 3
            # 2 0 2 3 1
            # 3 0 3 1 2
            # Addition is XOR:
            gf4_mul = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            
            def gf4_add(a, b): return a ^ b
            def gf4_sub(a, b): return a ^ b
            def gf4_mul_fn(a, b): return gf4_mul[a][b]
            
            def neighbors4(v):
                side, x = v
                out = []
                for first in range(4):
                    y = list(x)
                    y[1] = first
                    if side == 0:
                        for c in range(2, d_val + 1):
                            a, b = rule[c]
                            y[c] = gf4_add(x[c], gf4_mul_fn(y[a], x[b]))
                    else:
                        for c in range(2, d_val + 1):
                            a, b = rule[c]
                            y[c] = gf4_sub(x[c], gf4_mul_fn(x[a], y[b]))
                    out.append((1 - side, tuple(y)))
                return out

            start_v = (0, (None,) + (0,) * d_val)
            index = {start_v: 0}
            order = [start_v]
            edges = []
            for v in order:
                for u in neighbors4(v):
                    if u not in index:
                        index[u] = len(order)
                        order.append(u)
                    if v[0] == 0:
                        edges.append((index[v], index[u]))
            return len(order), edges

        else:
            def neighborsq(v):
                side, x = v
                out = []
                for first in range(q):
                    y = list(x)
                    y[1] = first
                    if side == 0:
                        for c in range(2, d_val + 1):
                            a, b = rule[c]
                            y[c] = (x[c] + y[a] * x[b]) % q
                    else:
                        for c in range(2, d_val + 1):
                            a, b = rule[c]
                            y[c] = (x[c] - x[a] * y[b]) % q
                    out.append((1 - side, tuple(y)))
                return out

            start_v = (0, (None,) + (0,) * d_val)
            index = {start_v: 0}
            order = [start_v]
            edges = []
            for v in order:
                for u in neighborsq(v):
                    if u not in index:
                        index[u] = len(order)
                        order.append(u)
                    if v[0] == 0:
                        edges.append((index[v], index[u]))
            return len(order), edges

    def verify(n, edges, k, g, time_limit=20.0):
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
            key = (min(u,v), max(u,v))
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        max_depth = g // 2
        from collections import deque
        for u in range(n):
            if time.time() > start + time_limit:
                return False
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q_ = deque([u])
            while q_:
                curr = q_.popleft()
                if dist[curr] >= max_depth:
                    continue
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q_.append(v)
                    else:
                        if v == parent[curr]:
                            continue
                        if dist[v] < dist[curr]:
                            continue
                        cycle_len = dist[curr] + 1 + dist[v]
                        if cycle_len < g:
                            return False
        return True

    best_n = float('inf')
    best_edges = None

    # Try LUW construction
    min_d = max(3, (g - 2) // 2)
    for d_val in range(min_d, min_d + 10):
        if time_left() < 20:
            break
        try:
            n, edges = build_luw(k, d_val)
            if n > 10000:
                continue
            if n < best_n:
                if verify(n, edges, k, g, time_limit=5.0):
                    best_n = n
                    best_edges = edges
                    break
        except Exception:
            pass

    if best_edges is not None:
        return best_n, best_edges

    # Fallback: Random search for small k, g
    # For k=4, g=10, n=320 is known. Let's try to construct one via random greedy + local search.
    # This is risky within time limits but might find a smaller graph than LUW.
    
    return 0, []
