import time
import itertools
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # For k=4, g=10: use voltage graph approach
    # Base: 4-regular graph on small number of vertices
    # Lift via voltage group Z_m to get n vertices
    
    if k != 4 or g != 10:
        # Fallback to LUW for other parameters
        return _luw_fallback(k, g)
    
    # Strategy: Try a 4-regular base graph on b vertices with voltages in Z_m
    # The lift has b*m vertices. We want b*m ≈ 300-320, girth >= 10.
    # Try base graphs that are "cage-like" themselves.
    
    # Use a known small 4-regular graph as base: e.g., the 4x4 torus (16 vertices)
    # or a 4-regular graph on 8 vertices (cube complement? No, cube is 3-regular)
    # 4-regular on 8 vertices: K8 minus perfect matching (8 vertices, 4-regular)
    
    # Let's try: base = K8 minus perfect matching (8 vertices, 4-regular)
    # Voltages in Z_m, m = 40, giving n = 320
    
    # Actually, let me try a more systematic approach:
    # Build the graph directly using a non-backtracking walk enumeration
    # with a greedy completion strategy.
    
    # Alternative: Use the known 320-vertex cage if we can construct it.
    # The (4,10)-cage is known to exist on 320 vertices.
    # Let me try to construct it via a voltage graph.
    
    # Base graph: 4-regular on 8 vertices (K8 minus 1-factor)
    # Vertices 0-7, edges: all pairs except (0,1),(2,3),(4,5),(6,7)
    base_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if (i, j) not in [(0,1),(2,3),(4,5),(6,7)]:
                base_edges.append((i, j))
    
    # Each vertex has degree 4 in this base.
    # Now assign voltages (elements of Z_m) to each edge.
    # The lift has 8*m vertices. For m=40, n=320.
    
    # The girth of the lift is determined by the shortest non-backtracking
    # closed walk in the base whose total voltage is 0 mod m.
    # We need no such walk of length < 10.
    
    # Let's search for voltage assignments in Z_40.
    # There are 14 edges in the base (K8 has 28 edges, minus 4 = 24... wait)
    # K8 has 8*7/2 = 28 edges. Minus 4 matching edges = 24 edges.
    # But each vertex should have degree 4. K8 minus 1-factor: each vertex loses 1 edge,
    # so degree = 7-1 = 6. That's 6-regular, not 4-regular!
    
    # I need a 4-regular base. Let me use a different base.
    # 4-regular on 10 vertices: Petersen graph!
    # Petersen graph: 10 vertices, 4-regular, girth 5.
    
    # Petersen graph edges (standard labeling):
    # Outer pentagon: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),  # outer
        (5,7), (7,9), (9,6), (6,8), (8,5),  # inner
        (0,5), (1,6), (2,7), (3,8), (4,9)   # spokes
    ]
    
    # 15 edges in Petersen. Assign voltages in Z_m.
    # Lift has 10*m vertices. For m=32, n=320.
    
    m = 32
    n_target = 10 * m  # 320
    
    # We need to assign voltages v_e in Z_m to each edge such that
    # no non-backtracking closed walk of length < 10 has voltage sum 0 mod m.
    
    # A non-backtracking walk of length L in the base corresponds to a walk
    # in the lift that closes iff the sum of voltages (with signs for direction)
    # is 0 mod m.
    
    # For a walk e1, e2, ..., eL (each ei is a directed edge), the voltage sum is
    # sum of v_ei if traversed in forward direction, -v_ei if backward.
    
    # We need: for all non-backtracking closed walks of length 4,6,8 in base,
    # the voltage sum != 0 mod m.
    
    # This is a constraint satisfaction problem. Let's use random search.
    
    import random
    random.seed(42)
    
    def get_lift(voltages):
        """Given voltage assignment (dict edge->voltage in Z_m), build the lift graph."""
        # Lift vertices: (base_vertex, offset) where offset in Z_m
        # Lift edges: for each base edge (u,v) with voltage w,
        # connect (u, a) to (v, a+w mod m) and (v, a) to (u, a-w mod m)
        # But we need undirected edges, so for each undirected base edge (u,v) with voltage w:
        # for a in range(m): edge between (u,a) and (v,(a+w)%m)
        
        n_lift = 10 * m
        lift_edges = []
        edge_set = set()
        
        for (u, v) in petersen_edges:
            w = voltages[(u, v)]
            for a in range(m):
                u_lift = u * m + a
                v_lift = v * m + (a + w) % m
                if u_lift < v_lift:
                    e = (u_lift, v_lift)
                else:
                    e = (v_lift, u_lift)
                if e not in edge_set:
                    edge_set.add(e)
                    lift_edges.append(e)
        
        return n_lift, lift_edges
    
    def check_girth(n, edges, min_girth):
        """Check if graph has girth >= min_girth using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for cycles of length < min_girth
        # Use non-backtracking BFS from each vertex
        for start in range(n):
            # State: (current, prev, depth)
            # We want to find a cycle of length < min_girth
            # A cycle of length L means we return to start after L steps
            # without backtracking.
            visited = {}  # (node, prev) -> depth
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited[(nb, start)] = 1
            
            while queue:
                node, prev, depth = queue.popleft()
                if depth >= min_girth - 1:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == start and depth + 1 < min_girth:
                        return False  # Found cycle shorter than min_girth
                    state = (nb, node)
                    if state in visited:
                        if visited[state] < depth + 1:
                            # Already visited with shorter path, skip
                            continue
                        else:
                            continue
                    visited[state] = depth + 1
                    queue.append((nb, node, depth + 1))
        
        return True
    
    # Random search for voltage assignment
    best_result = None
    attempts = 0
    
    while time.time() < deadline - 10:
        # Random voltage assignment
        voltages = {}
        for (u, v) in petersen_edges:
            voltages[(u, v)] = random.randint(0, m - 1)
        
        n_lift, lift_edges = get_lift(voltages)
        
        # Verify it's 4-regular
        deg = [0] * n_lift
        for u, v in lift_edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 4 for d in deg):
            continue
        
        # Check girth
        if check_girth(n_lift, lift_edges, g):
            best_result = (n_lift, lift_edges)
            break
        
        attempts += 1
        if attempts % 100 == 0:
            pass  # Could print progress
    
    if best_result is None:
        # Try with different m values
        for m_try in [20, 25, 40, 50]:
            if time.time() > deadline - 10:
                break
            m = m_try
            n_target = 10 * m
            
            for attempt in range(200):
                if time.time() > deadline - 10:
                    break
                voltages = {}
                for (u, v) in petersen_edges:
                    voltages[(u, v)] = random.randint(0, m - 1)
                
                n_lift, lift_edges = get_lift(voltages)
                
                deg = [0] * n_lift
                for u, v in lift_edges:
                    deg[u] += 1
                    deg[v] += 1
                if any(d != 4 for d in deg):
                    continue
                
                if check_girth(n_lift, lift_edges, g):
                    best_result = (n_lift, lift_edges)
                    break
            if best_result:
                break
    
    if best_result is None:
        # Fallback: use LUW construction
        return _luw_fallback(k, g)
    
    return best_result


def _luw_fallback(k, g):
    """Lazebnik-Ustimenko-Woldar graph as fallback."""
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
