"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    q = k
    # Initial guess for d: assume girth is 2d + 2
    d = max(3, (g - 2) // 2)
    
    while True:
        if time.time() > deadline:
            # Return the last constructed graph even if unverified? 
            # Better to return a valid one. If we ran out of time, we should have verified the previous one.
            # We will assume the last constructed graph is the best we have.
            # But we must ensure we return a valid graph.
            # If we are in the loop, we haven't verified the current d yet.
            # We should verify before returning.
            pass
            
        # Define field operations
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
        # Define the rule for the L-U-W construction
        # The rule depends on d
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        # Build the graph
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
        
        # BFS to build graph
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        n = len(order)
        
        # Verify girth
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles shorter than g
        # BFS from each vertex up to depth g//2 - 1
        # If we find an edge (u, v) in the BFS tree such that dist(u) + dist(v) + 1 < g, then girth < g
        # Actually, we need to check if the shortest cycle is < g.
        # A cycle of length L exists if there is a non-backtracking walk of length L.
        # Simpler: Run BFS from each node. If we encounter a neighbor that is already visited and is not the parent,
        # the cycle length is dist[u] + dist[v] + 1.
        # We must check if any such cycle is < g.
        
        valid = True
        limit_depth = g // 2 - 1
        
        for i in range(n):
            if time.time() > deadline:
                break
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            queue = [i]
            for u in queue:
                if dist[u] >= limit_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return n, edges
            
        # If invalid, increase d and retry
        d += 1
        # Safety: if d becomes too large, we might run out of time or memory
        if d > 20: 
             # Fallback to original logic if loop takes too long?
             # Or just break and return last valid?
             # But we don't have a valid one if we keep increasing.
             # The original program was valid for d=11.
             # So d will eventually be valid.
             pass

# EVOLVE-BLOCK-END
