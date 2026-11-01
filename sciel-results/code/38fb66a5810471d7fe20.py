import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    def luw_construct(k, g):
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

    if not (k == 5 and g == 9):
        return luw_construct(k, g)

    # Random search for k=5, g=9
    # Start n from 1116, step by 2
    n = 1116
    found = False
    best_n = float('inf')
    best_edges = None
    
    while time.time() < deadline - 10:
        if n % 2 != 0: n += 1 # 5n must be even
        
        # Try to generate a random 5-regular graph
        attempts = 0
        while time.time() < deadline - 10 and attempts < 20:
            attempts += 1
            # Configuration model
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            edges_set = set()
            is_simple = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    is_simple = False
                    break
                if u > v:
                    u, v = v, u
                if (u, v) in edges_set:
                    is_simple = False
                    break
                edges_set.add((u, v))
            
            if not is_simple:
                continue
                
            edges = list(edges_set)
            
            # Check girth
            # Build adj
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            valid = True
            # BFS check
            # For each start node
            for s in range(n):
                dist = [-1] * n
                dist[s] = 0
                queue = [s]
                # BFS to depth 4 (g//2 = 4)
                while queue:
                    u = queue.pop(0)
                    if dist[u] >= 4:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                            
                # Check all edges
                for u, v in edges:
                    if dist[u] != -1 and dist[v] != -1:
                        if dist[u] + dist[v] + 1 < g:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                best_n = n
                best_edges = edges
                found = True
                break # Found a valid graph for this n, try smaller n? 
                # Actually, we started at min known. If we find one, it's likely optimal or close.
                # We can break and return.
        
        if found:
            break
            
        # If not found, increase n
        n += 2
        
    if found:
        return best_n, best_edges
    else:
        return luw_construct(k, g)
