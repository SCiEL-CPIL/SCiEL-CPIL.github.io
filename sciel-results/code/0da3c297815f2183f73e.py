import time
import random
import itertools

# EVOLVE-BLOCK-START

def _girth_fast(n, edges, g_target):
    """Check if girth is at least g_target. Returns True if no cycle shorter than g_target."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for s in range(n):
        visited = {}
        queue = []
        for nb in adj[s]:
            state = (nb, s)
            visited[state] = 1
            queue.append((nb, s, 1))
        
        head = 0
        while head < len(queue):
            v, p, depth = queue[head]
            head += 1
            
            if depth >= g_target - 1:
                continue
            
            for nb in adj[v]:
                if nb == p:
                    continue
                if nb == s:
                    return False
                state = (nb, v)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((nb, v, depth + 1))
    
    return True

def _build_lift(base_adj, M):
    n = M * len(base_adj)
    edges = []
    for i in range(len(base_adj)):
        for (j, v) in base_adj[i]:
            if i < j:
                for x in range(M):
                    u = i * M + x
                    w = j * M + (x + v) % M
                    edges.append((u, w))
    return n, edges

def construct(k, g):
    deadline = time.time() + 140
    
    # Specific case for (4, 10) cage
    if k == 4 and g == 10:
        # Try K4,4 with M=40
        base_adj = [[] for _ in range(8)]
        for i in range(4):
            for j in range(4):
                base_adj[i].append((j+4, 0))
                base_adj[j+4].append((i, 0))
        
        M = 40
        for attempt in range(2000):
            if time.time() > deadline - 10:
                break
            voltages = [random.randint(0, M-1) for _ in range(16)]
            # Rebuild base_adj with voltages
            base_adj_v = [[] for _ in range(8)]
            idx = 0
            for i in range(4):
                for j in range(4):
                    v = voltages[idx]
                    base_adj_v[i].append((j+4, v))
                    base_adj_v[j+4].append((i, M - v if v != 0 else 0))
                    idx += 1
            
            n, edges = _build_lift(base_adj_v, M)
            if _girth_fast(n, edges, g):
                return n, edges
                
        # Fallback: Try K5 with M=64
        base_adj = [[] for _ in range(5)]
        for i in range(5):
            for j in range(i+1, 5):
                base_adj[i].append((j, 0))
                base_adj[j].append((i, 0))
        
        M = 64
        for attempt in range(2000):
            if time.time() > deadline - 10:
                break
            voltages = [random.randint(0, M-1) for _ in range(10)]
            base_adj_v = [[] for _ in range(5)]
            idx = 0
            for i in range(5):
                for j in range(i+1, 5):
                    v = voltages[idx]
                    base_adj_v[i].append((j, v))
                    base_adj_v[j].append((i, M - v if v != 0 else 0))
                    idx += 1
            
            n, edges = _build_lift(base_adj_v, M)
            if _girth_fast(n, edges, g):
                return n, edges

    # General fallback: LUW
    return _luz(k, g)

def _luz(k, g):
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
