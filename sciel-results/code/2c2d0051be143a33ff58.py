"""Deterministic algebraic construction for regular graphs of large girth."""
import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # safety margin before 150s kill

    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar construction for k-regular graphs of large girth."""
        q = k
        # Choose depth d such that girth >= g
        # LUW graph has girth at least 2*(d-1) for the construction
        # We need 2*(d-1) >= g, so d >= g//2 + 1
        d = max(3, g // 2 + 1)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return x * y % q
        
        # Rule table for the construction
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        # Build the graph by BFS from the root
        # Vertices are (side, x) where side in {0,1} and x is a tuple of length d+1
        # x[0] is None (marker), x[1..d] are values in range(q)
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        qi = 0
        while qi < len(order):
            vertex = order[qi]
            qi += 1
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                other = (1 - side, tuple(y))
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                # Only add edge from side 0 to avoid duplicates
                if side == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    def lps_construct(k, g):
        """
        Lubotzky-Phillips-Sarnak Ramanujan graph construction.
        For k=3, uses Cayley graph of PSL(2, p) with specific generators.
        The girth is related to the smallest eigenvalue of the adjacency matrix.
        """
        if k != 3:
            return None
        
        # For 3-regular LPS graphs, we use Cayley graphs of PSL(2, p)
        # where p is a prime with p ≡ 1 mod 4 (or use the variant)
        # The graph has p(p^2 - 1)/2 vertices... that's too many for small p.
        # 
        # Actually, for 3-regular Ramanujan graphs, the standard LPS construction
        # gives (p-1)-regular graphs. For 3-regular, we need a different approach.
        # 
        # Alternative: Use the (2,3)-Ramanujan graphs or the Margulis-Marulis construction.
        # 
        # For now, this is a placeholder - the LUW construction is more practical.
        return None

    def verify_girth(n, edges, min_girth):
        """
        Fast girth verification using non-backtracking walks from each vertex.
        Returns True if girth >= min_girth.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each starting vertex, do a non-backtracking BFS up to depth min_girth//2
        # If we revisit a vertex at depth d1 from start s and another vertex at depth d2
        # where d1 + d2 + 1 < min_girth, we have a short cycle.
        
        half = min_girth // 2
        for start in range(n):
            # dist[v] = minimum non-backtracking distance from start to v
            # prev[v] = previous vertex on the shortest non-backtracking path
            dist = [-1] * n
            prev = [-1] * n
            dist[start] = 0
            queue = deque()
            queue.append(start)
            
            while queue:
                u = queue.popleft()
                if dist[u] >= half:
                    continue
                for v in adj[u]:
                    if v == prev[u]:
                        continue  # non-backtracking
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle: start -> ... -> u -> v -> ... -> start
                        # The cycle length is dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        
        return True

    # Main logic
    if k == 3 and g == 16:
        # Use LUW construction as primary method
        # For k=3, g=16: d = 16//2 + 1 = 9
        # This gives a specific number of vertices
        n, edges = luw_construct(k, g)
        
        # Verify girth (should be fast for the constructed graph)
        if time.time() - t0 < deadline - 10:
            if not verify_girth(n, edges, g):
                # If verification fails (shouldn't happen with correct LUW), try larger d
                # Try incrementally larger depths
                for extra in range(1, 5):
                    if time.time() - t0 > deadline - 10:
                        break
                    # Rebuild with larger depth
                    old_d = g // 2 + 1
                    new_d = old_d + extra
                    # Monkey-patch: rebuild with new depth
                    q = k
                    d = new_d
                    if q == 4:
                        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
                        def add(x, y): return x ^ y
                        def sub(x, y): return x ^ y
                        def mul(x, y): return table[x][y]
                    else:
                        def add(x, y): return (x + y) % q
                        def sub(x, y): return (x - y) % q
                        def mul(x, y): return x * y % q
                    
                    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
                    for c in range(5, d + 1):
                        r = (c - 5) % 4
                        c0 = c - (c - 5) % 4
                        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
                    
                    start_v = (0, (None,) + (0,) * d)
                    index = {start_v: 0}
                    order = [start_v]
                    edges_new = []
                    qi = 0
                    while qi < len(order):
                        vertex = order[qi]
                        qi += 1
                        side, x = vertex
                        for first in range(q):
                            y = [None, first] + [0] * (d - 1)
                            for c in range(2, d + 1):
                                a, b = rule[c]
                                if side == 0:
                                    y[c] = add(x[c], mul(y[a], x[b]))
                                else:
                                    y[c] = sub(x[c], mul(x[a], y[b]))
                            other = (1 - side, tuple(y))
                            if other not in index:
                                index[other] = len(order)
                                order.append(other)
                            if side == 0:
                                edges_new.append((index[vertex], index[other]))
                    
                    n = len(order)
                    edges = edges_new
                    if verify_girth(n, edges, g):
                        return n, edges
        
        return n, edges
    else:
        n, edges = luw_construct(k, g)
        return n, edges
# EVOLVE-BLOCK-END
