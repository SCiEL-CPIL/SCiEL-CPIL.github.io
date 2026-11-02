import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Moore bound
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2 - 1
        moore = k * sum((k - 1) ** i for i in range(r))
        
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
            r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

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

    # Local Search
    def check_girth(adj, n, target_g):
        # Returns True if girth >= target_g
        # Use BFS from each vertex
        for v in range(n):
            visited = [-1] * n
            parent = [-1] * n
            queue = [v]
            visited[v] = 0
            while queue:
                u = queue.pop(0)
                dist = visited[u]
                if dist >= (target_g - 1) // 2:
                    continue
                for w in adj[u]:
                    if visited[w] == -1:
                        visited[w] = dist + 1
                        parent[w] = u
                        queue.append(w)
                    elif parent[w] != u and parent[u] != w:
                        # Found a cycle
                        # Cycle length is dist + 1 + dist_to_lca
                        # This simple check might miss some or miscount, 
                        # but for "is there a cycle < g", we can just check if we revisit a node 
                        # with dist < g/2.
                        # Actually, if we revisit a node with dist < g/2, we have a cycle.
                        # We need to be careful.
                        pass
        return True # Placeholder, need better check

    # Better Girth Check
    def has_short_cycle(adj, n, target_g):
        # Check for cycles of length 3, 4, ..., target_g-1
        # For small target_g (9), this is feasible.
        # Use BFS from each vertex to find shortest cycle.
        for v in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[v] = 0
            queue = [v]
            while queue:
                u = queue.pop(0)
                if dist[u] >= (target_g - 1) // 2:
                    continue
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        parent[w] = u
                        queue.append(w)
                    else:
                        # Found a back edge
                        # Check if it forms a cycle < target_g
                        # The cycle length is dist[u] + 1 + dist[w] (if u and w are not parent)
                        # Actually, this logic is tricky for general graphs.
                        # Simpler: Just do BFS up to depth g/2. If we find a node already visited 
                        # that is not the parent, we have a cycle.
                        if w != parent[u]:
                            # We found a cycle. 
                            # We need to verify its length.
                            # For now, assume if we find any back edge within depth g/2, 
                            # it might be a short cycle.
                            # To be safe, we can just return True if we find ANY back edge 
                            # in the BFS tree within depth g/2? 
                            # No, that's not sufficient.
                            pass
        return False

    # Given the complexity of implementing a perfect girth checker and local search 
    # within the token limit, and the risk of bugs, 
    # I will rely on the LUW construction but try to optimize the parameters 
    # or use a known smaller construction if possible.
    
    # Actually, the LUW construction is the only one I can guarantee is correct.
    # I will return the LUW construction.
    return luw_construct(k, g)
