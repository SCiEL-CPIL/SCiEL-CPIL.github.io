import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def luw_construct():
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

    if k != 4 or g != 10:
        return luw_construct()

    # Fast girth check using non-backtracking BFS from each vertex
    def girth_ok(adj, n):
        # For girth >= 10, we need no cycles of length 3, 4, 5, 6, 7, 8, 9
        # Use BFS from each vertex, tracking parent to avoid backtracking
        # A cycle of length L < g is found if we reach a vertex at distance
        # >= 2 from start via a non-parent edge where dist[u] + dist[v] + 1 - 2*dist[lca] < g
        # Simpler: BFS from each start, if we find any back-edge (to a non-parent ancestor)
        # where the cycle length < g, return False.
        # For efficiency, we can limit BFS depth to (g-1)//2 = 4.
        # If no cycle is found within depth 4 from any start, girth >= 10.
        # Actually, a cycle of length 9 has diameter 4, so BFS depth 4 from any vertex on the cycle
        # will find it. But we need to be careful.
        # Let's use the standard approach: for each edge (u,v), remove it and check if shortest path u->v < g-1.
        # But that's O(E*(V+E)) which is too slow for many trials.
        
        # Better: BFS from each vertex with parent tracking.
        # If we encounter a vertex that was visited and is not the parent, and the cycle length < g, fail.
        # Cycle length when u-v is an edge and v was visited at dist[v]:
        # The cycle goes from v up to LCA then back to u.
        # To get exact cycle length, we need LCA. This is complex.
        
        # Alternative: just check for cycles of length 3, 4, 5, 6, 8 (even lengths up to 8, and 3, 5, 7, 9).
        # For k=4, checking up to depth 4 from each vertex:
        # Depth 1: 4 neighbors
        # Depth 2: 4*3 = 12
        # Depth 3: 12*3 = 36
        # Depth 4: 36*3 = 108
        # Total per vertex: ~160 nodes visited.
        # For n=320: 320 * 160 = 51,200 operations.
        # This is fast enough for ~10,000 trials.
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= 4:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # Found a cycle. The cycle length is at most dist[u] + dist[v] + 1
                        # but could be shorter depending on LCA.
                        # However, if dist[u] + dist[v] + 1 < g, it's definitely a short cycle.
                        # If dist[u] >= 2 and dist[v] >= 2, and the path from u to v through their
                        # common ancestor is short, we have a cycle.
                        # Simple check: if min(dist[u], dist[v]) <= 3 and dist[u] + dist[v] + 1 < g
                        # This is conservative (might miss some short cycles but won't false-positive).
                        # Actually, we need to be exact. Let's check if the edge (u,v) creates
                        # a cycle of length < g.
                        # The cycle length is at least 3 (triangle) and at most dist[u]+dist[v]+1.
                        # If dist[u]+dist[v]+1 < g, it's a fail.
                        # If dist[u]+dist[v]+1 >= g, it might still be a fail if the actual cycle is shorter.
                        # For g=10, if dist[u]+dist[v]+1 < 10, fail.
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    def generate_random_4_reg(n):
        # Configuration model with rejection
        stubs = []
        for i in range(n):
            stubs.extend([i] * 4)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, 4 * n, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Strategy: Try n from 242 upward (even only for 4-regular).
    # For each n, try random generation + girth check.
    # If found, return immediately.
    # If time runs out, return LUW (320).
    
    # Moore bound for k=4, g=10: n >= 2*(4*(4^3 - 1)/(4-1)) = 2*(4*63/3) = 2*84 = 168? 
    # Wait, Moore bound for even girth g=2m: n >= k * sum_{i=0}^{m-1} (k-1)^i = 4 * (1 + 4 + 16 + 64) = 4 * 85 = 340?
    # No, that's for the Moore bound. Let me recalculate.
    # For k-regular graph with girth g = 2m (even), the Moore bound is:
    # n >= k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(m-1))
    # For k=4, m=5: n >= 4 * (1 + 4 + 16 + 64 + 256) = 4 * 341 = 1364? That seems too high.
    # Wait, the Moore bound formula: for girth g, the minimum number of vertices is bounded below by
    # the number of vertices in a k-regular tree of depth (g-1)/2.
    # For g=10, depth = 4.5, so we consider depth 4 on one side.
    # Actually, the standard Moore bound for even girth g=2m:
    # n >= k * sum_{i=0}^{m-1} (k-1)^i
    # For k=4, m=5: n >= 4 * (1 + 4 + 16 + 64 + 256) = 4 * 341 = 1364.
    # But the known cage is 320, so the Moore bound must be different.
    # Let me look up: The Moore bound for (k, g) cages:
    # For g even, g = 2m: n >= k * (k-1)^(m-1) * (k-1 + 1) / (k-1 - 1)? No.
    # The correct formula: n >= k * sum_{i=0}^{m-1} (k-1)^i for the Moore bound.
    # Hmm, but 320 < 1364. So I'm using the wrong formula.
    # Actually, the lower bound is: n >= k * ( (k-1)^m - 1 ) / (k-2) for g=2m? 
    # Let me just use the known fact that the best lower bound is 243, so n >= 244 (even).
    
    # We'll try n from 244 to 318 in steps of 2.
    # For each n, try up to 5000 random graphs.
    # If girth check passes, return.
    
    best_result = luw_construct()
    best_n = best_result[0]
    
    for n in range(244, min(best_n, 320), 2):
        if time.time() > deadline - 5:
            break
        max_trials = 5000
        for trial in range(max_trials):
            if time.time() > deadline - 2:
                break
            adj = generate_random_4_reg(n)
            if adj is None:
                continue
            if girth_ok(adj, n):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    return best_result
