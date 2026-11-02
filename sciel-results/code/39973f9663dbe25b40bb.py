import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    # Try to find a smaller circulant graph
    def check_girth_circulant(n, generators):
        # generators: list of 3 ints
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            for s in generators:
                j1 = (i + s) % n
                j2 = (i - s) % n
                adj[i].append(j1)
                adj[i].append(j2)
        
        # Check for duplicate edges or self loops
        # If s == n-s, then j1 == j2, which is a multi-edge (loop in simple graph terms? No, 2-edge).
        # Simple graph requires no multi-edges.
        # So s != n-s mod n => 2s != 0 mod n.
        # Also s_i != s_j, s_i != -s_j.
        
        # Girth check via BFS from each node
        # We need to ensure no cycle of length < g
        # Since graph is bipartite? No, circulant graphs are not necessarily bipartite.
        # But if we choose generators such that the graph is bipartite, it's easier?
        # For now, general BFS.
        
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g // 2: # If we go deeper, we might find cycles, but we can stop early if we've covered enough?
                    # Actually, to certify girth >= g, we need to ensure no cycles of length < g.
                    # A cycle of length L is found if we have an edge (u, v) such that dist[u] + dist[v] + 1 = L < g.
                    # Or if we encounter a visited node v from u, and dist[u] + dist[v] + 1 < g.
                    pass
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    else:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    # Search for circulant
    # We want 3-regular.
    # Try n from 512 upwards.
    # To save time, only try a few random generator sets for each n.
    
    # Note: For k=3, g=16, the graph must be bipartite to have large girth?
    # Not necessarily, but bipartite graphs have no odd cycles, so we only check even cycles 4..14.
    # This makes the check slightly faster (or same).
    
    # Heuristic: Use generators that are "far apart".
    # E.g., 1, n/3, n/2?
    
    # Given time constraints, this search might not find a solution.
    # But if it does, it's a huge win.
    
    # Let's try a very limited search.
    if k == 3 and g == 16:
        for n in range(512, 1000, 16): # Step 16 to keep n even (bipartite potential)
            if time.time() > deadline:
                break
            # Try a few specific generator sets
            # Set 1: {1, n//3, n//2} (if valid)
            # Set 2: Random
            for _ in range(5):
                if time.time() > deadline:
                    break
                gen = [random.randint(1, n-1) for _ in range(3)]
                # Check validity: distinct, not 0, not n/2 (if n even, n/2 is its own inverse, creates multi-edge? No, 2*(n/2)=n=0. So edge i--i+n/2. This is a valid simple edge. But if we have two generators s and n-s, we get multi-edge.
                # So we must ensure s_i != s_j and s_i != n - s_j.
                valid = True
                for i in range(3):
                    for j in range(i+1, 3):
                        if gen[i] == gen[j] or gen[i] == n - gen[j]:
                            valid = False
                if not valid:
                    continue
                if check_girth_circulant(n, gen):
                    # Build edges
                    edges = []
                    for i in range(n):
                        for s in gen:
                            edges.append((i, (i + s) % n))
                    return n, edges
    
    # Fallback
    return luw_construct(k, g)
