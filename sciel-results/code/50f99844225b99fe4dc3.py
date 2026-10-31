import time
import random
import sys
from collections import deque

def _luw_construct(k, g):
    # Seed program's construction
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

def _kautz_voltage_lift(k, g, m, seed=42):
    """
    Construct a 3-regular graph via voltage lift of a Kautz-type base graph.
    
    For k=3, the Kautz digraph K(3,1) has 3 vertices (0,1,2) with edges:
    0->1, 1->2, 2->0 (directed cycle)
    Treating as undirected: triangle on 3 vertices.
    
    We use a larger base: the 3-regular Kautz graph K(3,2) has 6 vertices.
    Actually, let's use a simpler approach: voltage lift of a 3-regular base
    graph over Z_m.
    
    For k=3, g=16:
    - Base graph: use a 3-regular graph with girth >= some value
    - Voltage group: Z_m
    - Total vertices: 3 * m (if base has 3 vertices) or more
    
    Strategy: Use the Petersen graph (10 vertices, 3-regular, girth 5) as base.
    Voltage lift over Z_m gives 10*m vertices.
    For m=94: 940 vertices.
    
    Assign voltages to edges such that any closed walk of length < 16 has
    non-zero voltage sum.
    """
    if k != 3:
        return None
    
    # Use Petersen graph as base (10 vertices, 3-regular, girth 5)
    # Petersen graph edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),  # inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    n_base = 10
    n_edges_base = len(petersen_edges)
    
    # We need to find m and voltage assignments
    # For girth 16, we need: no closed walk of length < 16 has voltage sum 0 mod m
    
    # Try different m values
    # The total number of vertices will be n_base * m = 10 * m
    # We want 10*m <= 936, so m <= 93.6, try m = 93, 94, 95...
    
    # Actually, let's be more careful. The base graph has girth 5, so
    # any cycle in the lift corresponds to a closed walk in the base.
    # We need voltage assignments such that for every closed walk of length
    # L < 16 in the base, the voltage sum is not 0 mod m.
    
    # Let's try a systematic approach with random voltage assignments
    # and check the girth of the resulting graph.
    
    # For efficiency, we'll try a few m values and voltage assignments
    best_result = None
    best_n = float('inf')
    
    start_time = time.time()
    deadline = start_time + 80  # Leave 20s for verification
    
    # Try m values that give reasonable graph sizes
    # m should be around 90-100 for n around 900-1000
    m_candidates = [93, 94, 95, 96, 97, 98, 99, 100]
    
    random.seed(seed)
    
    for m in m_candidates:
        if time.time() > deadline:
            break
        
        n_total = n_base * m
        
        # Try several random voltage assignments
        max_tries = 50
        for attempt in range(max_tries):
            if time.time() > deadline:
                break
            
            # Assign random voltages to edges
            # Voltage is in Z_m
            voltages = {}
            for u, v in petersen_edges:
                voltages[(u, v)] = random.randint(1, m - 1)
                voltages[(v, u)] = (-voltages[(u, v)]) % m
            
            # Build the lifted graph
            # Vertex (base_v, volt) where base_v in 0..9, volt in 0..m-1
            # Edge from (u, a) to (v, a + voltage(u,v)) mod m
            adj = [[] for _ in range(n_total)]
            edges_set = set()
            
            for u, v in petersen_edges:
                volt = voltages[(u, v)]
                for a in range(m):
                    b = (a + volt) % m
                    vid1 = u * m + a
                    vid2 = v * m + b
                    adj[vid1].append(vid2)
                    adj[vid2].append(vid1)
                    if vid1 < vid2:
                        edges_set.add((vid1, vid2))
                    else:
                        edges_set.add((vid2, vid1))
            
            # Verify: check degree and girth
            # First check all vertices have degree 3
            valid = True
            for i in range(n_total):
                if len(adj[i]) != 3:
                    valid = False
                    break
                if len(set(adj[i])) != 3:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check for multi-edges
            if len(edges_set) != n_total * 3 // 2:
                continue
            
            # Check girth using BFS from each vertex (but only up to depth 7)
            # If we find a cycle of length < 16, reject
            girth_ok = True
            for start_v in range(n_total):
                if time.time() > deadline:
                    girth_ok = False
                    break
                # BFS with parent tracking
                dist = [-1] * n_total
                parent = [-1] * n_total
                dist[start_v] = 0
                queue = deque([start_v])
                found_short = False
                while queue:
                    u = queue.popleft()
                    if dist[u] >= 8:  # We only care about cycles < 16
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            if dist[v] < 8:
                                queue.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                found_short = True
                                break
                    if found_short:
                        break
                if found_short:
                    girth_ok = False
                    break
            
            if girth_ok:
                # Success!
                edges = list(edges_set)
                return n_total, edges
    
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, leave 10s margin
    
    # Try voltage lift construction first
    if k == 3:
        result = _kautz_voltage_lift(k, g, 94, seed=42)
        if result:
            return result
        
        # Try with different seeds
        for seed in [123, 456, 789, 101, 202, 303]:
            if time.time() > deadline:
                break
            result = _kautz_voltage_lift(k, g, 94, seed=seed)
            if result:
                return result
        
        # Try different m values
        for m in [93, 95, 96, 97]:
            if time.time() > deadline:
                break
            for seed in [42, 123, 456]:
                if time.time() > deadline:
                    break
                result = _kautz_voltage_lift(k, g, m, seed=seed)
                if result:
                    return result
    
    # Fallback to L-U-W
    return _luw_construct(k, g)
