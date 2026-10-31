"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import deque

def construct(k, g):
    # Random lift of Petersen graph with local search to achieve girth g
    # Petersen graph: 10 vertices, 3-regular, girth 5
    # Lift by Z_m: 10m vertices
    
    # Petersen graph edges (standard labeling)
    # Outer cycle: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    base_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),  # outer cycle
        (5,7), (7,9), (9,6), (6,8), (8,5),  # inner star
        (0,5), (1,6), (2,7), (3,8), (4,9)   # spokes
    ]
    base_adj = [[] for _ in range(10)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
    
    # Find all cycles of length < g in Petersen graph
    # Petersen graph is small, we can enumerate
    def find_all_cycles_up_to_len(base_adj, max_len):
        cycles = []
        n_base = len(base_adj)
        for start in range(n_base):
            # BFS/DFS for cycles starting and ending at start, not revisiting nodes
            stack = [(start, [start], [start])]
            while stack:
                node, path, edges_in_path = stack.pop()
                if len(path) > max_len:
                    continue
                for nb in base_adj[node]:
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        cycle = list(path)
                        # Normalize: start from min vertex, ensure consistent direction
                        min_idx = cycle.index(min(cycle))
                        norm = cycle[min_idx:] + cycle[:min_idx]
                        # Also check reverse
                        rev = list(reversed(norm))
                        rev_min_idx = rev.index(min(rev))
                        rev_norm = rev[rev_min_idx:] + rev[:rev_min_idx]
                        canon = tuple(min(norm, rev_norm))
                        if canon not in [tuple(c) for c in cycles]:
                            cycles.append(list(canon))
                    elif nb not in path:
                        stack.append((nb, path + [nb], edges_in_path + [(node, nb)]))
        return cycles
    
    short_cycles = find_all_cycles_up_to_len(base_adj, g - 1)
    
    # For each cycle, we need sum of twists != 0 mod m
    # Assign random twists to edges, check constraints, local search to fix
    
    # Determine m: we want 10*m vertices, aim for reasonable size
    # Try m such that 10*m is in a good range
    # The known best is 936, so m ~ 94. But let's try various m.
    
    def compute_girth(n, edges):
        """Compute exact girth using BFS from each vertex (non-backtracking)."""
        if n == 0:
            return float('inf')
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_girth = float('inf')
        for start in range(n):
            # BFS with parent tracking to avoid backtracking
            dist = [-1] * n
            parent = [-1] * n
            queue = deque([start])
            dist[start] = 0
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue  # non-backtracking
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth
    
    def build_lift(m, twists):
        """Build the lifted graph. twists is a dict mapping (u,v) to twist value."""
        n = 10 * m
        edges = []
        for u, v in base_edges:
            key = (min(u,v), max(u,v))
            t = twists.get(key, 0)
            for i in range(m):
                u_lift = u * m + i
                v_lift = v * m + (i + t) % m
                edges.append((min(u_lift, v_lift), max(u_lift, v_lift)))
        return n, edges
    
    def check_girth(n, edges, target_g):
        """Check if girth >= target_g."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = deque([start])
            dist[start] = 0
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_g:
                            return False
        return True
    
    # Try different m values and random twist assignments
    best_result = None
    best_n = float('inf')
    
    # The number of edges in base graph
    edge_keys = [(min(u,v), max(u,v)) for u, v in base_edges]
    
    # For each cycle, get the sequence of edge keys
    def cycle_edge_keys(cycle):
        keys = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            keys.append((min(u,v), max(u,v)))
        return keys
    
    cycle_keys = [cycle_edge_keys(c) for c in short_cycles]
    
    # Try m from small to larger
    for m in range(40, 120, 2):
        n = 10 * m
        if n >= best_n:
            continue
        
        # Try random twist assignments
        for attempt in range(50):
            random.seed(42 + m * 100 + attempt)
            twists = {key: random.randint(0, m-1) for key in edge_keys}
            
            # Check constraints: for each short cycle, sum of twists != 0 mod m
            valid = True
            for keys in cycle_keys:
                s = sum(twists[key] for key in keys) % m
                if s == 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Build and check girth
            n_lift, edges = build_lift(m, twists)
            if check_girth(n_lift, edges, g):
                if n_lift < best_n:
                    best_n = n_lift
                    best_result = (n_lift, edges)
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: use a larger random construction
    # Just return a valid 3-regular graph with girth at least g
    # Use a simple approach: random 3-regular graph and check
    # For safety, use a known construction
    
    # Fallback to LUW-like construction (simplified)
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
# EVOLVE-BLOCK-END
