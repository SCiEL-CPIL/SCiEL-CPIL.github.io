import time
import random
import math
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- LUW Fallback Construction ----------------
    def _luw_graph(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # ---------------- Girth Verification ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                d = dist[node]
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < g:
                            return cycle_len
                    else:
                        dist[nb] = d + 1
                        parent[nb] = node
                        if d + 1 < g:
                            queue.append(nb)
        return g

    # ---------------- Girth-Constrained Random Regular Graph ----------------
    def girth_constrained_rrg(k, g, max_n, time_limit):
        """
        Build a k-regular graph with girth >= g using the configuration model
        with dynamic edge rejection. Start with a small seed graph and grow it
        by adding vertices one at a time, ensuring no short cycles are created.
        """
        if k != 3:
            return None

        # Moore bound for (3,16): 2 + 3*(1+2+4+8+16+32) = 2 + 3*63 = 191
        # But we know n >= 512 from theory. Try various n values.
        # We'll try to find the smallest n that works.
        
        # For 3-regular graphs, n must be even
        # Start from a reasonable lower bound
        moore_bound = 2 + k * sum(k**(i-1) for i in range(1, (g-1)//2 + 1))
        min_n = max(moore_bound, 512)
        if min_n % 2 != 0:
            min_n += 1
        
        # Try to build graphs with increasing n
        for target_n in range(min_n, max_n + 1, 2):
            if remaining() < 10:
                break
            
            # Build using configuration model with girth constraint
            # We use a "growth" approach: start with a path and extend
            # Actually, let's use a different approach:
            # Start with a known small girth-16 graph fragment and try to complete it
            
            # Alternative: Use the "random regular graph with girth constraint" 
            # by building edge-by-edge with backtracking
            
            success = _build_girth_rrg(target_n, k, g, time_limit - 5)
            if success is not None:
                return success
        
        return None

    def _build_girth_rrg(n, k, g, time_limit):
        """
        Build a k-regular graph on n vertices with girth >= g using
        a randomized approach with backtracking.
        
        Strategy: 
        1. Create a random configuration (stubs)
        2. Pair stubs randomly
        3. Check if girth constraint is satisfied
        4. If not, reject and retry (with modifications)
        
        For large n, pure rejection is too slow. Instead, we use:
        - Start with a partial graph
        - Add edges one at a time, checking girth constraint
        - Use local search to fix violations
        """
        if time.time() > time_limit:
            return None
        
        # For n up to ~1000, we can try a direct approach
        # Build a random 3-regular graph and check/fix girth
        
        # Method: Start with a cycle and add chords carefully
        # Actually, let's use a simpler approach:
        # Generate random 3-regular graphs and check girth
        # If n is small enough, this might work
        
        if n > 2000:
            return None  # Too large for this approach
        
        num_attempts = 200
        for attempt in range(num_attempts):
            if time.time() > time_limit:
                return None
            
            # Generate a random 3-regular graph
            edges = _random_regular_graph(n, k)
            if edges is None:
                continue
            
            # Check girth
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Quick girth check using BFS from a few random vertices
            girth_ok = True
            sample_vertices = random.sample(range(n), min(n, 50))
            for s in sample_vertices:
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                queue = deque([s])
                while queue:
                    node = queue.popleft()
                    d = dist[node]
                    if d >= (g - 1) // 2:
                        continue
                    for nb in adj[node]:
                        if nb == parent[node]:
                            continue
                        if dist[nb] != -1:
                            cycle_len = dist[node] + dist[nb] + 1
                            if cycle_len < g:
                                girth_ok = False
                                break
                        else:
                            dist[nb] = d + 1
                            parent[nb] = node
                            queue.append(nb)
                    if not girth_ok:
                        break
                if not girth_ok:
                    break
            
            if girth_ok:
                # Full verification
                gc = check_girth(n, edges, g, time_limit)
                if gc >= g:
                    return (n, edges)
        
        return None

    def _random_regular_graph(n, k):
        """Generate a random k-regular graph using the configuration model."""
        if n * k % 2 != 0:
            return None
        
        # Create stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        # Randomly pair stubs
        for _ in range(100):
            random.shuffle(stubs)
            edges = []
            valid = True
            edge_set = set()
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append(key)
            
            if valid:
                return edges
        
        return None

    # Try girth-constrained RRG first
    res = girth_constrained_rrg(k, g, max_n=1200, time_limit=deadline)
    if res is not None:
        return res

    # Fallback: LUW construction
    n_luw, edges_luw = _luw_graph(k, g)
    return (n_luw, edges_luw)


if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
