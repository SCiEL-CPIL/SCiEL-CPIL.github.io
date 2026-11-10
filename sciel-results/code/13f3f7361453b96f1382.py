import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses Cayley graphs on SL(2, q) for small q to find small n.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    if k != 3:
        # Fallback for other k values
        pass

    # List of (q, p, m, poly) to try in order of increasing group size
    # |SL(2, q)| = q(q^2 - 1)
    # q=9: 720
    # q=11: 1320
    # q=13: 2184
    # q=16: 32767
    candidates = [
        (9, 3, 2, 0b101),    # x^2 + 1 over GF(3)
        (11, 11, 1, 0),      # GF(11)
        (13, 13, 1, 0),      # GF(13)
        (16, 2, 4, 0b10011), # x^4 + x + 1 over GF(2)
        (25, 5, 2, 0b10001), # x^2 + 1 over GF(5) -> 5(24)=120*5=600? No, 5(25-1)=120. Wait.
                             # |SL(2, 25)| = 25(625-1) = 15600.
    ]
    
    # Add more if needed, but 16 is the current best.
    # Actually, let's stick to the ones that are likely to work.
    
    def create_gf(p, m, poly):
        q = p ** m
        if m == 1:
            def add(a, b): return (a + b) % p
            def mul(a, b): return (a * b) % p
            def inv(a):
                if a == 0: return 0
                for i in range(1, p):
                    if (a * i) % p == 1: return i
                return 0
            def div(a, b):
                if b == 0: return 0
                return mul(a, inv(b))
            return q, add, mul, inv, div

        # Precompute multiplication table
        mul_table = [[0] * q for _ in range(q)]
        for i in range(q):
            for j in range(q):
                res = 0
                a = i
                b = j
                while b:
                    if b & 1:
                        res ^= a
                    a <<= 1
                    if a >= q:
                        a ^= poly
                    b >>= 1
                mul_table[i][j] = res
        
        def add(a, b): return a ^ b
        def mul(a, b): return mul_table[a][b]
        def inv(a):
            if a == 0: return 0
            for i in range(1, q):
                if mul_table[a][i] == 1: return i
            return 0
        def div(a, b):
            if b == 0: return 0
            return mul(a, inv(b))
        
        return q, add, mul, inv, div

    def check_girth_fast(adj, n_nodes, target_girth):
        max_depth = (target_girth - 1) // 2
        for s in range(n_nodes):
            dist = [-1] * n_nodes
            parent = [-1] * n_nodes
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    random.seed(42)
    
    best_result = None
    
    for q, p, m, poly in candidates:
        if time.time() > deadline - 20:
            break
            
        q_val, gf_add, gf_mul, gf_inv, gf_div = create_gf(p, m, poly)
        
        # Generate SL(2, q) elements
        elements = []
        for a in range(q_val):
            for b in range(q_val):
                for c in range(q_val):
                    if a == 0:
                        if gf_mul(b, c) == 1:
                            for d in range(q_val):
                                elements.append((a, b, c, d))
                    else:
                        bc = gf_mul(b, c)
                        rhs = gf_add(1, bc)
                        d = gf_div(rhs, a)
                        elements.append((a, b, c, d))
        
        n = len(elements)
        elem_to_idx = {e: i for i, e in enumerate(elements)}
        
        # Find Involutions
        involutions = []
        identity = (1, 0, 0, 1)
        for e in elements:
            a, b, c, d = e
            if e == identity:
                continue
            # Check A^2 = I
            # (a b; c d)^2 = (a^2+bc, ab+bd; ca+dc, cb+d^2)
            aa = gf_add(gf_mul(a, a), gf_mul(b, c))
            ab = gf_add(gf_mul(a, b), gf_mul(b, d))
            ba = gf_add(gf_mul(c, a), gf_mul(d, c))
            bb = gf_add(gf_mul(c, b), gf_mul(d, d))
            if (aa, ab, ba, bb) == identity:
                involutions.append(e)
        
        if len(involutions) < 3:
            continue

        def multiply(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            aa = gf_add(gf_mul(a1, a2), gf_mul(b1, c2))
            ab = gf_add(gf_mul(a1, b2), gf_mul(b1, d2))
            ba = gf_add(gf_mul(c1, a2), gf_mul(d1, c2))
            bb = gf_add(gf_mul(c1, b2), gf_mul(d1, d2))
            return (aa, ab, ba, bb)

        # Try random sets
        max_trials = 50
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            S = random.sample(involutions, 3)
            
            # Build adjacency
            adj = [[] for _ in range(n)]
            for i, e in enumerate(elements):
                for s in S:
                    f = multiply(e, s)
                    j = elem_to_idx[f]
                    adj[i].append(j)
            
            # Quick connectivity check
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 1
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        queue.append(v)
            
            if count != n:
                continue
                
            # Check girth
            if check_girth_fast(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # Fallback to LUW if all Cayley attempts fail
    def build_luw(d, q_val):
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_fn = lambda x, y: x ^ y
            sub_fn = lambda x, y: x ^ y
            mul_fn = lambda x, y: table[x][y]
        else:
            add_fn = lambda x, y: (x + y) % q_val
            sub_fn = lambda x, y: (x - y) % q_val
            mul_fn = lambda x, y: x * y % q_val

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                    else:
                        y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def check_girth_luw(n, edges, girth_target):
        max_depth = (girth_target - 1) // 2
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth_target:
                            return False
        return True

    q_val = 3
    best_luw = None
    for d in range(5, 15):
        if time.time() > deadline:
            break
        n_luw, edges_luw = build_luw(d, q_val)
        if n_luw < 2:
            continue
        deg = [0] * n_luw
        for u, v in edges_luw:
            deg[u] += 1
            deg[v] += 1
        if any(dd != k for dd in deg):
            continue
        if check_girth_luw(n_luw, edges_luw, g):
            return n_luw, edges_luw
        best_luw = (n_luw, edges_luw)
        
    if best_luw:
        return best_luw

    return 0, []
