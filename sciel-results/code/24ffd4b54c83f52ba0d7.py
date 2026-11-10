# EVOLVE-BLOCK-START
import time
import random
import collections

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    # Try Cayley graph on SL(2, 16) for k=3, g=16
    if k == 3 and g == 16:
        try:
            result = _try_cayley_sl2_16(deadline)
            if result:
                return result
        except Exception:
            pass

    # Fallback to LUW
    return _luw_construct(k, g)

def _try_cayley_sl2_16(deadline):
    # GF(16) implementation
    def gf_add(x, y):
        return x ^ y

    def gf_mul(x, y):
        if x == 0 or y == 0:
            return 0
        z = 0
        while y:
            if y & 1:
                z ^= x
            x <<= 1
            if x & 16:
                x ^= 0x11
            y >>= 1
        return z

    def gf_inv(x):
        if x == 0:
            return 0
        # x^14 = 1
        res = 1
        base = x
        n = 14
        while n:
            if n & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            n >>= 1
        return res

    # SL(2, 16) elements
    elements = []
    elem_to_idx = {}
    
    for a in range(16):
        for b in range(16):
            for c in range(16):
                if a != 0:
                    d = gf_mul(gf_inv(a), gf_add(1, gf_mul(b, c)))
                    m = (a, b, c, d)
                    if m not in elem_to_idx:
                        elem_to_idx[m] = len(elements)
                        elements.append(m)
                else:
                    if b == 0:
                        continue
                    if gf_mul(b, c) == 1:
                        for d in range(16):
                            m = (0, b, c, d)
                            if m not in elem_to_idx:
                                elem_to_idx[m] = len(elements)
                                elements.append(m)
    
    n = len(elements)
    identity = (1, 0, 0, 1)
    
    def mat_mul(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return (
            gf_mul(a1, a2) ^ gf_mul(b1, c2),
            gf_mul(a1, b2) ^ gf_mul(b1, d2),
            gf_mul(c1, a2) ^ gf_mul(d1, c2),
            gf_mul(c1, b2) ^ gf_mul(d1, d2)
        )
    
    def mat_inv(m):
        a, b, c, d = m
        return (d, b, c, a)
    
    inv_map = {}
    sq_map = {}
    for m in elements:
        inv_map[m] = mat_inv(m)
        sq_map[m] = mat_mul(m, m)
    
    involutions = [m for m in elements if sq_map[m] == identity and m != identity]
    
    if not involutions:
        return None

    def check_girth(g1, g3):
        g2 = inv_map[g1]
        if g1 == g2:
            return False
        S = [g1, g2, g3]
        
        adj = [[] for _ in range(n)]
        for i, m in enumerate(elements):
            for s in S:
                neighbor = mat_mul(m, s)
                j = elem_to_idx[neighbor]
                adj[i].append(j)
        
        # BFS up to depth 7
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = collections.deque([u])
            
            while q:
                v = q.popleft()
                if dist[v] >= 7:
                    continue
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        q.append(w)
                    else:
                        if w != parent[v]:
                            cycle_len = dist[v] + dist[w] + 1
                            if cycle_len < 16:
                                return False
        return True

    random.shuffle(involutions)
    elements_list = elements[:]
    random.shuffle(elements_list)
    
    max_iters = 5000
    for i in range(max_iters):
        if time.time() > deadline:
            return None
        g1 = elements_list[i % len(elements_list)]
        if sq_map[g1] == identity:
            continue
        g3 = involutions[i % len(involutions)]
        
        if check_girth(g1, g3):
            edges = []
            for i, m in enumerate(elements):
                for s in [g1, inv_map[g1], g3]:
                    neighbor = mat_mul(m, s)
                    j = elem_to_idx[neighbor]
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
    return None

def _luw_construct(k, g):
    # Existing LUW code
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
# EVOLVE-BLOCK-END
