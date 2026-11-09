import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: LUW construction
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

    # Try to find a smaller graph using random search
    # Target n: start with 2^g and increase
    # For g=16, 2^16 = 65536
    # Let's try a few values
    
    best_n, best_edges = None, None
    
    # Try n = 80000, 100000, etc.
    # But generating and checking is expensive.
    # Let's try one candidate: n = 100000
    
    n_target = 100000
    
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        # Generate random 3-regular graph
        # Configuration model
        stubs = [i for i in range(n_target) for _ in range(k)]
        random.shuffle(stubs)
        edges = []
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u != v:
                edges.append((u, v))
        
        # Check for multi-edges
        # This is O(E log E) or O(E) with set
        edge_set = set()
        valid = True
        adj = [[] for _ in range(n_target)]
        for u, v in edges:
            if (u, v) in edge_set or (v, u) in edge_set:
                valid = False
                break
            edge_set.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
        
        if not valid:
            continue
            
        # Check girth
        # BFS from each vertex to depth g//2
        # If we find a cycle < g, fail
        girth_ok = True
        max_depth = g // 2
        dist = [-1] * n_target
        parent = [-1] * n_target
        
        for u in range(n_target):
            if time.time() > deadline:
                break
            # Reset dist
            # To avoid O(n) reset, use a "seen" array with timestamps
            # But simple reset is O(n). n=100k. 100k * 100k = 10^10. Too slow.
            # We need a faster check.
            # Actually, we can just do BFS and stop if we find a short cycle.
            # Most graphs will have short cycles, so we stop early.
            
            dist = [-1] * n_target
            dist[u] = 0
            q = deque([u])
            
            while q:
                curr = q.popleft()
                d_curr = dist[curr]
                if d_curr >= max_depth:
                    continue
                for nb in adj[curr]:
                    if dist[nb] == -1:
                        dist[nb] = d_curr + 1
                        q.append(nb)
                    else:
                        # Found a back edge
                        # Cycle length = d_curr + 1 + dist[nb]
                        # Wait, this logic is for simple BFS.
                        # If nb is visited, the cycle length is d_curr + 1 + dist[nb]?
                        # No, if nb is visited at depth d_nb, and we are at d_curr,
                        # the cycle is d_curr + d_nb + 1.
                        # But this is only if the path to nb is disjoint from the path to curr.
                        # In a tree, yes. But BFS tree is a tree.
                        # So if we see a visited node, we have a cycle.
                        cycle_len = d_curr + dist[nb] + 1
                        if cycle_len < g:
                            girth_ok = False
                            break
                if not girth_ok:
                    break
            
            if not girth_ok:
                break
        
        if girth_ok:
            best_n, best_edges = n_target, edges
            break
    
    if best_n is not None:
        return best_n, best_edges
    else:
        return luw_construct(k, g)
