import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def luw_construction():
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

    if k == 4 and g == 10:
        # Pre-compute Q4 cycles
        # Vertices 0..15
        # Edges: Hamming dist 1
        base_edges = []
        for u in range(16):
            for v in range(u+1, 16):
                if bin(u ^ v).count('1') == 1:
                    base_edges.append((u, v))
        
        # Find all cycles of length 4, 6, 8
        # Use DFS from each vertex to find simple cycles
        # Since graph is small, this is fast.
        cycles_4 = []
        cycles_6 = []
        cycles_8 = []
        
        # Adjacency for base
        adj_base = [[] for _ in range(16)]
        for u, v in base_edges:
            adj_base[u].append(v)
            adj_base[v].append(u)
            
        for start in range(16):
            # DFS to find cycles
            # We want to find cycles of length L.
            # To avoid duplicates, only start from the smallest vertex in the cycle?
            # Or just find all and deduplicate.
            # Given small size, find all simple cycles.
            def find_cycles(start, length):
                cycles = []
                # DFS
                # Path so far
                stack = [(start, [start])]
                while stack:
                    u, path = stack.pop()
                    if len(path) == length:
                        # Check if u connected to start
                        if start in adj_base[u]:
                            # Found a cycle
                            # Normalize: sort vertices? Or just keep as is and dedup later.
                            # To dedup, we can represent cycle as frozenset of edges?
                            # Or sorted tuple of vertices.
                            cycles.append(tuple(sorted(path)))
                    else:
                        for v in adj_base[u]:
                            if v not in path:
                                # Pruning: if remaining steps < distance to start, skip?
                                # Simple check:
                                stack.append((v, path + [v]))
                return cycles

            # This DFS might be slow or generate many duplicates.
            # Better: Use networkx? No, std lib only.
            # For Q4, we can hardcode or generate efficiently.
            # Let's just use the DFS but limit depth.
            # Actually, for Q4, 4-cycles are squares.
            # 6-cycles are hexagons.
            # 8-cycles are octagons.
            
            # Let's just do the random search and check girth on the cover directly,
            # but optimize the girth check.
            # Or, limit attempts to 10.
            pass

        # Let's try 10 random assignments and check girth on cover.
        # 10 * (320 * 640) ~ 2M ops. Should be fast in Python if optimized.
        
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            voltages = {}
            for u, v in base_edges:
                val = random.randint(1, 19)
                voltages[(u, v)] = val
                voltages[(v, u)] = (-val) % 20
            
            # Build cover
            n = 320
            adj = [[] for _ in range(n)]
            for u, v in base_edges:
                vu = voltages[(u, v)]
                for i in range(20):
                    u_cov = u * 20 + i
                    v_cov = v * 20 + (i + vu) % 20
                    adj[u_cov].append(v_cov)
                    adj[v_cov].append(u_cov)
            
            # Girth check
            # BFS from each node, stop if cycle < 10 found.
            girth_ok = True
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u] and u != parent[v]:
                            l = dist[u] + dist[v] + 1
                            if l < 10:
                                girth_ok = False
                                break
                    if not girth_ok:
                        break
                if not girth_ok:
                    break
            
            if girth_ok:
                edges = []
                for u, v in base_edges:
                    vu = voltages[(u, v)]
                    for i in range(20):
                        u_cov = u * 20 + i
                        v_cov = v * 20 + (i + vu) % 20
                        edges.append((u_cov, v_cov))
                return 320, edges

    return luw_construction()
