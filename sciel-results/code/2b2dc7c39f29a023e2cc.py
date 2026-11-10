# EVOLVE-BLOCK-START
import random
import time

def _random_regular_graph(n, k):
    if n * k % 2 != 0:
        return None
    stubs = [i for i in range(n) for _ in range(k)]
    random.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None
        if (u, v) in edges or (v, u) in edges:
            return None
        edges.add((min(u, v), max(u, v)))
    return list(edges)

def _has_short_cycles(n, edges, max_len=5):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        for nei in adj[start]:
            stack = [(nei, start, 1)]
            while stack:
                node, prev, depth = stack.pop()
                if depth >= max_len:
                    continue
                for next_node in adj[node]:
                    if next_node == prev:
                        continue
                    if next_node == start:
                        if depth + 1 <= max_len:
                            return True
                    else:
                        stack.append((next_node, node, depth + 1))
    return False

def _check_girth_fast(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        for nei in adj[start]:
            stack = [(nei, start, 1)]
            while stack:
                node, prev, depth = stack.pop()
                if depth >= g - 1:
                    continue
                for next_node in adj[node]:
                    if next_node == prev:
                        continue
                    if next_node == start:
                        if depth + 1 < g:
                            return False
                    else:
                        stack.append((next_node, node, depth + 1))
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 4 and g == 10:
        best_n = float('inf')
        best_edges = None
        
        for n_base in range(10, 31, 2):
            if time.time() > deadline:
                break
            for attempt in range(50):
                if time.time() > deadline:
                    break
                base_edges = _random_regular_graph(n_base, 4)
                if not base_edges:
                    continue
                if _has_short_cycles(n_base, base_edges, 5):
                    continue
                
                # Try lifts
                for L in range(10, 40):
                    if time.time() > deadline:
                        break
                    n = n_base * L
                    if n < 243:
                        continue
                    if n >= best_n:
                        continue
                        
                    # Random voltage
                    voltages = {}
                    for u, v in base_edges:
                        voltages[(u, v)] = random.randint(0, L - 1)
                        voltages[(v, u)] = (-voltages[(u, v)]) % L
                    
                    lift_edges_set = set()
                    for u, v in base_edges:
                        w = voltages[(u, v)]
                        for i in range(L):
                            u_lift = u * L + i
                            v_lift = v * L + (i + w) % L
                            lift_edges_set.add((min(u_lift, v_lift), max(u_lift, v_lift)))
                    
                    lift_edges = list(lift_edges_set)
                    
                    if _check_girth_fast(n, lift_edges, g):
                        if n < best_n:
                            best_n = n
                            best_edges = lift_edges
                            # We can return immediately if n is close to lower bound?
                            # Or just keep searching for smaller n?
                            # Since we want to minimize n, we should keep searching 
                            # but we have a deadline.
                            # If we find n=320, it's good.
                            # If we find n=300, better.
                            pass
        
        if best_edges:
            return best_n, best_edges
    
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
