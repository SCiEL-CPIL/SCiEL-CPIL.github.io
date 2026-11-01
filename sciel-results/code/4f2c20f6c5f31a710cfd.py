"""Voltage graph lift approach for cage problem n(4,10)."""
import numpy as np
from itertools import product
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g using voltage graph lifting.
    
    Strategy: Use a small base 4-regular graph and lift it over Z_m with
    carefully chosen voltages to ensure girth >= g. The lifted graph has
    n = |V_base| * m vertices. We search for voltage assignments that
    maximize the shortest non-backtracking cycle voltage sum.
    """
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    # Try voltage graph lift with small base graphs
    best_n, best_edges = None, None
    
    # Try base graph sizes: 8, 10, 12, 14, 16 vertices
    for base_size in [8, 10, 12, 14, 16]:
        # Generate candidate base graphs (4-regular)
        base_graphs = _generate_base_graphs(base_size)
        
        for base_edges in base_graphs:
            # Try different lift sizes m
            # n = base_size * m, we want n as small as possible but >= Moore bound ~243
            # So m >= ceil(243 / base_size)
            min_m = (243 + base_size - 1) // base_size
            
            for m in range(min_m, min_m + 20):
                n_total = base_size * m
                if n_total > 320:
                    continue
                
                # Try to find a voltage assignment that gives girth >= g
                voltages = _find_good_voltages(base_edges, m, g)
                if voltages is not None:
                    edges = _lift_graph(base_edges, voltages, m)
                    if best_n is None or n_total < best_n:
                        best_n = n_total
                        best_edges = edges
    
    if best_edges is not None:
        # Verify girth
        if _verify_girth(best_n, best_edges, g):
            return best_n, best_edges
    
    # Fallback to LUW construction
    return _luw_construct(k, g)


def _verify_girth(n, edges, g):
    """Verify that the graph has girth >= g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 4:
            return False
    
    # Check no self-loops or multi-edges
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if (min(u,v), max(u,v)) in edge_set:
            return False
        edge_set.add((min(u,v), max(u,v)))
    
    # BFS from each vertex to find shortest cycle
    # For efficiency, we can use a multi-source approach or just check
    # For n up to 320, this is feasible
    for start in range(n):
        # BFS tracking parent to detect cycles
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    # Length of cycle = dist[u] + dist[v] + 1
                    # But this may not be the shortest cycle through this edge
                    # For a proper girth check, we need to be more careful
                    pass
        
        # Better: use the standard approach - for each edge (u,v), the shortest cycle
        # containing it has length = dist_u_to_v_in_graph_without_edge + 1
        # This is expensive. Instead, let's do a simpler check:
        # BFS from each vertex, and when we find a back edge, compute cycle length
    
    # Simpler verification: check that no cycle of length < g exists
    # by doing BFS from each vertex and checking for early returns
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Potential cycle detected
                    # The cycle length is dist[u] + dist[v] + 1, but this overcounts
                    # because the path from start to u and start to v may share prefix
                    # For a rigorous check, we'd need to compute LCA
                    # For now, this is a heuristic check
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # This might be a false positive due to shared prefix
                        # Let's do a more careful check
                        pass
    
    # For a definitive check with n <= 320, let's use the non-backtracking walk approach
    # as specified in the task
    return _check_girth_nonbacktracking(n, edges, g)


def _check_girth_nonbacktracking(n, edges, g):
    """Check girth >= g using non-backtracking walks from every vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do a non-backtracking BFS up to depth g-1
    # If we can reach the start vertex again, we've found a cycle
    for start in range(n):
        # State: (current_vertex, previous_vertex, depth)
        # We want to see if we can return to start with depth < g
        visited = set()
        queue = []
        for v in adj[start]:
            queue.append((v, start, 1))
            visited.add((v, start))
        
        head = 0
        while head < len(queue):
            u, prev, depth = queue[head]
            head += 1
            
            if depth >= g - 1:
                continue
            
            for v in adj[u]:
                if v == prev:
                    continue  # Non-backtracking
                
                if v == start:
                    # Found a cycle of length depth + 1
                    if depth + 1 < g:
                        return False
                    continue
                
                state = (v, u)
                if state not in visited:
                    visited.add(state)
                    queue.append((v, u, depth + 1))
    
    return True


def _generate_base_graphs(size):
    """Generate small 4-regular graphs on `size` vertices."""
    graphs = []
    if size % 2 != 0:
        return graphs
    
    # For small sizes, enumerate or use known constructions
    # Use a deterministic approach: try all 4-regular graphs for small sizes
    # This is only feasible for size <= 16
    
    if size == 8:
        # The 8-vertex 4-regular graphs: cube and others
        # Cube: vertices 0-7, edges of 3D cube
        cube_edges = [
            (0,1),(0,2),(0,4),(0,5),
            (1,3),(1,5),(1,6),
            (2,3),(2,6),(2,7),
            (3,7),
            (4,5),(4,6),(4,7),
            (5,7),
            (6,7)
        ]
        # Check if 4-regular
        deg = [0]*8
        for u,v in cube_edges:
            deg[u] += 1
            deg[v] += 1
        if all(d == 4 for d in deg):
            graphs.append(cube_edges)
        # Also try other 4-regular graphs on 8 vertices
        # The complete bipartite K4,4 is 4-regular on 8 vertices
        k44_edges = [(i, 4+j) for i in range(4) for j in range(4)]
        graphs.append(k44_edges)
        
    elif size == 10:
        # Generate random 4-regular graphs
        graphs = _random_4reg_graphs(10, num_samples=50)
        
    elif size == 12:
        graphs = _random_4reg_graphs(12, num_samples=50)
        
    elif size == 14:
        graphs = _random_4reg_graphs(14, num_samples=50)
        
    elif size == 16:
        graphs = _random_4reg_graphs(16, num_samples=100)
    
    return graphs


def _random_4reg_graphs(n, num_samples=50):
    """Generate random 4-regular graphs on n vertices using configuration model."""
    graphs = []
    seen = set()
    random.seed(42)
    
    for _ in range(num_samples):
        # Configuration model: each vertex has 4 stubs
        stubs = [i for i in range(n) for _ in range(4)]
        random.shuffle(stubs)
        
        # Pair up stubs to form edges
        edges = []
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if (min(u,v), max(u,v)) in seen:
                valid = False
                break
            seen.add((min(u,v), max(u,v)))
            edges.append((u, v))
        
        if valid and len(edges) == 2*n:
            # Check it's actually 4-regular (should be by construction)
            graphs.append(edges)
    
    return graphs


def _find_good_voltages(base_edges, m, g, max_attempts=500):
    """
    Find voltage assignment for base edges such that no non-backtracking
    cycle of length < g in the base has voltage sum ≡ 0 mod m.
    
    Returns list of voltages for each edge, or None if not found.
    """
    n_base = max(max(u for u,v in base_edges), max(v for u,v in base_edges)) + 1
    num_edges = len(base_edges)
    
    # Build adjacency
    adj = [[] for _ in range(n_base)]
    for idx, (u, v) in enumerate(base_edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    # Precompute all non-backtracking cycles of length 3 to g-1 in the base graph
    # For each such cycle, we record the sequence of edge indices
    short_cycles = []
    
    for start in range(n_base):
        for first_neighbor, first_eidx in adj[start]:
            # DFS to find all non-backtracking paths from first_neighbor back to start
            # with length 2 to g-2 (so total cycle length 3 to g-1)
            stack = [(first_neighbor, start, 1, [first_eidx], 0)]
            while stack:
                vertex, prev, length, edge_seq, volt_sum = stack.pop()
                
                if length >= g - 1:
                    continue
                
                for neighbor, eidx in adj[vertex]:
                    if neighbor == prev:
                        continue
                    
                    new_length = length + 1
                    new_edge_seq = edge_seq + [eidx]
                    
                    if neighbor == start and new_length >= 3 and new_length < g:
                        # Found a cycle of length new_length
                        short_cycles.append(new_edge_seq)
                    elif new_length < g - 1:
                        stack.append((neighbor, vertex, new_length, new_edge_seq, 0))
    
    # Now find voltages such that for every short cycle, sum of voltages != 0 mod m
    random.seed(42)
    for attempt in range(max_attempts):
        voltages = [random.randint(0, m-1) for _ in range(num_edges)]
        
        valid = True
        for cycle in short_cycles:
            total = sum(voltages[e] for e in cycle) % m
            if total == 0:
                valid = False
                break
        
        if valid:
            return voltages
    
    return None


def _lift_graph(base_edges, voltages, m):
    """
    Lift the base graph over Z_m using the given voltages.
    Returns list of edges in the lifted graph.
    """
    n_base = max(max(u for u,v in base_edges), max(v for u,v in base_edges)) + 1
    n_total = n_base * m
    
    edges = []
    edge_set = set()
    
    for idx, (u, v) in enumerate(base_edges):
        voltage = voltages[idx]
        
        for s in range(m):
            # Edge from (u, s) to (v, (s + voltage) % m)
            u_lifted = u * m + s
            v_lifted = v * m + ((s + voltage) % m)
            edge = (min(u_lifted, v_lifted), max(u_lifted, v_lifted))
            if edge not in edge_set:
                edge_set.add(edge)
                edges.append(edge)
    
    return edges


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) construction."""
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
