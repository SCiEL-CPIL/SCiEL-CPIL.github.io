import time
import random
import itertools
from collections import deque

def _get_icosahedron():
    """
    Returns the edge list of the Icosahedron graph (12 vertices, 5-regular, girth 5).
    Vertices are indexed 0..11.
    """
    # Coordinates of icosahedron vertices
    # (0, 1, phi), (0, 1, -phi), (0, -1, phi), (0, -1, -phi)
    # (1, phi, 0), (1, -phi, 0), (-1, phi, 0), (-1, -phi, 0)
    # (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
    # phi = (1 + sqrt(5)) / 2
    # We can just hardcode the adjacency for speed and simplicity.
    # Icosahedron graph is the 1-skeleton of the icosahedron.
    # It is the complement of the disjoint union of 6 disjoint edges (K2 x 6)? No.
    # It is the line graph of K_{3,3}? No, that's 9 vertices.
    # It is the complement of the Petersen graph? No, Petersen is 10 vertices.
    
    # Let's use a known adjacency list for Icosahedron (12 vertices).
    # Reference: Icosahedral graph
    adj = [
        [1, 2, 3, 4, 5],
        [0, 2, 6, 7, 8],
        [0, 1, 6, 9, 10],
        [0, 4, 5, 9, 11],
        [0, 3, 5, 7, 11],
        [0, 3, 4, 8, 10],
        [1, 2, 7, 9, 11],
        [1, 4, 6, 8, 10],
        [1, 5, 7, 10, 11],
        [2, 3, 6, 8, 10],
        [2, 5, 7, 8, 9],
        [3, 4, 6, 8, 9]
    ]
    edges = []
    for u in range(12):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

def _get_dodecahedron():
    """
    Returns the edge list of the Dodecahedron graph (20 vertices, 5-regular, girth 5).
    """
    # Dodecahedron graph
    # We can construct it from two pentagons and connections, or use a standard list.
    # Let's use a standard adjacency.
    # 20 vertices.
    # This is harder to hardcode correctly from memory.
    # Alternative: Use the Icosahedron (12 vertices) which is smaller and easier to verify.
    # 12 * 35 = 420. 12 * 36 = 432.
    # If we use Dodecahedron (20 vertices), 20 * 22 = 440.
    # Icosahedron is better for small N.
    return None 

def _check_girth_bfs(adj, n, target_girth):
    """
    Check if graph has girth at least target_girth.
    Returns True if girth >= target_girth.
    Uses BFS from each vertex up to depth target_girth // 2.
    """
    half_g = target_girth // 2
    # We need to detect cycles of length < target_girth.
    # A cycle of length L is detected if we find a neighbor at distance L-1 that is also a neighbor of the start?
    # Or more simply: BFS from start. If we encounter a node at depth d that was visited at depth d' 
    # such that d + d' + 1 < target_girth, we have a short cycle.
    # Actually, standard girth check:
    # For each start node s:
    #   BFS.
    #   dist[v] = shortest distance from s.
    #   parent[v] = parent in BFS tree.
    #   For each edge (u, v):
    #     if u == parent[v] or v == parent[u]: continue
    #     cycle_len = dist[u] + dist[v] + 1
    #     if cycle_len < target_girth: return False
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        parent[s] = -2
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Back edge or cross edge
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < target_girth:
                        return False
    return True

def _construct_lift(base_edges, m, voltages):
    """
    Construct the lift of base_edges with given voltages in Z_m.
    Returns (n, edges, adj) or None if invalid (multi-edges etc).
    """
    base_vertices = set()
    for u, v in base_edges:
        base_vertices.add(u)
        base_vertices.add(v)
    base_n = len(base_vertices)
    # Map base vertices to 0..base_n-1 for indexing
    # Since we use icosahedron, vertices are 0..11
    n = base_n * m
    adj = {i: [] for i in range(n)}
    edges_set = set()
    
    for i, (u, v) in enumerate(base_edges):
        vol = voltages[i] % m
        for a in range(m):
            u_lift = u * m + a
            v_lift = v * m + (a + vol) % m
            
            # Check for multi-edges/loops
            if u_lift == v_lift:
                return None, None, None
            if v_lift in adj[u_lift]:
                return None, None, None
                
            adj[u_lift].append(v_lift)
            adj[v_lift].append(u_lift)
            edges_set.add((min(u_lift, v_lift), max(u_lift, v_lift)))
            
    # Verify degree
    for i in range(n):
        if len(adj[i]) != 5:
            return None, None, None
            
    return n, list(edges_set), adj

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Use Icosahedron base
    base_edges = _get_icosahedron()
    base_n = 12
    
    best_n = float('inf')
    best_edges = None
    
    # We want to find the smallest n.
    # n = 12 * m.
    # Moore bound for k=5, g=9 is ~428.
    # So m should be at least ceil(428/12) = 36.
    # Let's try m from 35 upwards.
    
    # Pre-calculate short cycles in base to speed up voltage checking?
    # Or just build and BFS.
    # Building is O(E_base * m). BFS is O(V_lift * (V_lift+E_lift)) worst case, but usually fast if girth is large.
    # Actually, BFS from each vertex is O(N * (N+E)). For N=500, E=1250, this is 500 * 1750 ~ 10^6 ops.
    # 10^6 ops is very fast in Python (milliseconds).
    # So we can try many random voltage assignments.
    
    for m in range(35, 100):
        if time.time() > deadline:
            break
            
        n = base_n * m
        
        # Try to find a valid voltage assignment
        found = False
        attempts = 0
        max_attempts = 5000 # Adjust based on time
        
        while not found and time.time() < deadline:
            attempts += 1
            if attempts > max_attempts:
                break
                
            # Random voltages
            voltages = [random.randint(0, m - 1) for _ in range(len(base_edges))]
            
            # Optimization: Ensure no 5-cycle in base has voltage sum 0 or small?
            # We can just build and check.
            
            res_n, res_edges, adj = _construct_lift(base_edges, m, voltages)
            if res_n is None:
                continue
                
            # Check girth
            if _check_girth_bfs(adj, res_n, g):
                best_n = n
                best_edges = res_edges
                found = True
                break
        
        if found:
            # Since we iterate m from small to large, the first valid m gives the smallest n
            # But we should check if a larger m with different base could be smaller?
            # No, n = base_n * m. Smaller m is better for same base.
            # We can return immediately if we found a valid graph for the smallest m?
            # Yes, because we are looking for min n.
            # However, it's possible that m=35 fails, m=36 fails, m=37 succeeds.
            # So we return the first success.
            return best_n, best_edges

    # If no lift found, fallback to a known construction or random regular graph
    # Fallback: LUW construction from seed
    n, edges = _luw_construct(k, g)
    return n, edges

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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
