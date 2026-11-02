"""Construct a k-regular graph of girth at least g, minimizing vertex count."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    def time_left():
        return deadline - time.time()

    best_n = None
    best_edges = None

    def verify_girth(n, edges, min_girth):
        """Verify no cycle shorter than min_girth using BFS from each vertex."""
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Check simple: no self-loops, no duplicate edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        # BFS from each vertex to find shortest cycle
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q = deque([start])
            dist[start] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def get_largest_component(n, edges):
        """Return the largest connected component as (n_comp, edges_comp)."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        visited = [False] * n
        best_comp = []
        for i in range(n):
            if not visited[i]:
                comp = []
                queue = deque([i])
                visited[i] = True
                while queue:
                    u = queue.popleft()
                    comp.append(u)
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                if len(comp) > len(best_comp):
                    best_comp = comp
        
        if not best_comp:
            return 0, []
        
        comp_set = set(best_comp)
        comp_edges = []
        for u, v in edges:
            if u in comp_set and v in comp_set:
                comp_edges.append((u, v))
        
        # Remap vertices to 0..n_comp-1
        remap = {v: i for i, v in enumerate(best_comp)}
        remapped_edges = [(remap[u], remap[v]) for u, v in comp_edges]
        
        return len(best_comp), remapped_edges

    def try_luw_construction(q, d):
        """Lazebnik-Ustimenko-Woldar construction CD(d, q)."""
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

    # Try LUW construction with various parameters
    # For girth g, we need d such that the construction has girth >= g
    # CD(d,q) has girth d+5 for odd d, d+4 for even d
    # So for g=9: d=5 (odd, girth>=10) or d=4 (even, girth>=8 - not enough)
    # Let's try d values that might work
    
    candidate_params = []
    # For the target k and g, try q=k first with various d
    for d in range(max(3, g - 6), g - 2):
        candidate_params.append((k, d))
    # Also try larger q values
    for q in [k + 2, k + 4, k + 6]:
        if q > 0 and q % 2 == 1:  # q should be prime for the construction
            for d in range(max(3, g - 6), g - 2):
                candidate_params.append((q, d))

    # Deduplicate and sort by expected size (smaller q and d first)
    seen = set()
    unique_params = []
    for q, d in candidate_params:
        if (q, d) not in seen and q >= 3:
            seen.add((q, d))
            unique_params.append((q, d))
    
    unique_params.sort(key=lambda x: (x[0], x[1]))

    for q, d in unique_params:
        if time_left() < 5:
            break
        try:
            n_exp, edges_exp = try_luw_construction(q, d)
            # Check if it's k-regular
            adj = [[] for _ in range(n_exp)]
            for u, v in edges_exp:
                adj[u].append(v)
                adj[v].append(u)
            if all(len(adj[i]) == k for i in range(n_exp)):
                # Verify girth
                if verify_girth(n_exp, edges_exp, g):
                    if best_n is None or n_exp < best_n:
                        best_n = n_exp
                        best_edges = edges_exp
        except Exception:
            pass

    # If no valid graph found, fall back to the basic LUW with d=g-5 or g-4
    if best_n is None:
        d = g - 5 if (g - 5) % 2 == 1 else g - 4
        if d < 3:
            d = 3
        try:
            n_exp, edges_exp = try_luw_construction(k, d)
            return n_exp, edges_exp
        except Exception:
            pass

    return best_n, best_edges
# EVOLVE-BLOCK-END
