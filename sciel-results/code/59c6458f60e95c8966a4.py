"""Combined program: Fast search for small 3-regular graphs + LUW fallback."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    # --- LUW fallback with minimal d ---
    def _luw_graph(k, g):
        q = k
        # Minimal d such that girth >= g
        # Girth of LUW(q, d) is 2d+2
        # So 2d+2 >= g => d >= (g-2)/2
        d = max(3, (g - 2) // 2)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        # Rule for LUW
        # Standard LUW rule:
        # y_c = x_c + y_{c-2} * x_{c-1} for c >= 2
        # But the seed code had a specific rule. Let's use the standard recursive definition.
        # Actually, the seed code's rule was:
        # rule[c] = (a, b) such that y[c] = op(x[c], mul(y[a], x[b]))
        # For LUW, it is typically y_c = x_c + y_{c-2} x_{c-1}.
        # Let's stick to the seed's rule generation if it works, or use the standard one.
        # The seed's rule for c=2 is (1,1) -> y2 = x2 + y1*x1.
        # c=3: (2,1) -> y3 = x3 + y2*x1.
        # This looks like y_c = x_c + y_{c-1} x_{c-2}? No.
        # Let's just use the seed's logic for LUW but with optimized d.
        
        rule = {}
        # Replicate seed's rule logic
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
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

    # --- Fast girth verification ---
    def _check_girth(adj, n, g_limit):
        max_depth = (g_limit - 1) // 2
        for start in range(n):
            dist = np.full(n, -1, dtype=np.int32)
            parent = np.full(n, -1, dtype=np.int32)
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g_limit:
                            return False
        return True

    # --- Bipartite Cayley Graph Search ---
    def _try_bipartite_cayley(m):
        """Search for a 3-regular bipartite Cayley graph on Z_m x {0,1} with girth >= g."""
        # Generate random connection sets S of size 3
        # S must not contain 0
        # Also, to avoid small cycles, elements should be "independent"
        
        # Heuristic: try sets with large elements first? Or random?
        # Random is better for avoiding specific algebraic relations.
        
        for _ in range(100): # Try 100 random sets per m
            if time.time() > deadline - 10:
                return None
            
            S = random.sample(range(1, m), 3)
            
            # Quick check: no 2s = 0 mod m (avoid girth 2, though bipartite handles it, 
            # but we need simple graph, so s != 0 and s != m/2 if m even? 
            # If s = m/2, then u -> u+m/2 -> u. Cycle of length 2 in the underlying multigraph? 
            # In bipartite graph, edge (u, v) and (v, u) are distinct? 
            # No, it's a simple graph. 
            # If s = m/2, then u connects to u+m/2. 
            # u+m/2 connects to u+m/2+m/2 = u. 
            # So edge (u, u+m/2) is the same as (u+m/2, u). 
            # This is fine for a simple graph, but it creates a "cycle" of length 2 in the walk sense? 
            # No, it's just an edge. 
            # But it reduces the effective connectivity. 
            # Let's avoid s = m/2 if m is even to be safe.
            if m % 2 == 0:
                if m//2 in S:
                    continue

            # Build graph
            n = 2 * m
            adj = [[] for _ in range(n)]
            edges = []
            for u in range(m):
                for s in S:
                    v = (u + s) % m
                    # Left u connects to Right v
                    adj[u].append(m + v)
                    adj[m + v].append(u)
                    edges.append((u, m + v))
            
            # Check degrees
            for i in range(n):
                if len(adj[i]) != 3:
                    break
            else:
                # Check girth
                if _check_girth(adj, n, g):
                    return n, edges
        return None

    # --- Main Logic ---
    best_n = float('inf')
    best_edges = []

    if k == 3 and g == 16:
        # Try to find a small graph using bipartite Cayley
        # Target n ~ 936 => m ~ 468
        # Search m from 468 to 1000
        found = False
        for m in range(468, 1000, 2):
            if time.time() > deadline - 20:
                break
            result = _try_bipartite_cayley(m)
            if result:
                n, edges = result
                if n < best_n:
                    best_n = n
                    best_edges = edges
                    found = True
                    # We can stop if we find a very good one, e.g. <= 1000
                    if n <= 1000:
                        break
        
        if found:
            return best_n, best_edges

    # Fallback: LUW with minimal d
    n_luw, edges_luw = _luw_graph(k, g)
    return n_luw, edges_luw
# EVOLVE-BLOCK-END
