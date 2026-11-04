import time
import random
import sys

def construct(k, g):
    deadline = time.time() + 130.0  # safety margin before 150s kill

    # For k=4, g=10: use voltage graph / lift construction
    # Base graph: complement of Q3 (8 vertices, 4-regular, girth 4)
    # Lift by Z_m to get 8*m vertices
    # We need to find voltages so the lift has girth >= 10
    
    # The complement of Q3: vertices 0..7, edges connect vertices at Hamming distance 2 or 3
    # Actually Q3 has 8 vertices, each connected to 3 others (distance 1). 
    # Complement has degree 4, edges at distance 2 or 3.
    # Girth of complement of Q3: shortest cycle is 4 (e.g., 0-3-5-6-0, all dist 2 or 3)
    
    # For the lift to have girth >= 10, we need no non-backtracking closed walk 
    # of length < 10 in the lift.
    
    # Strategy: try small m values, use simulated annealing or random search 
    # to find voltage assignments that avoid short cycles.
    
    if k != 4 or g != 10:
        # Fallback: LUW construction
        def get_luw_graph():
            d = 3
            while True:
                girth_est = d + 5 if d % 2 == 1 else d + 4
                if girth_est >= g:
                    break
                d += 1
            
            q = k
            if q == 4:
                mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
                def add(x, y): return x ^ y
                def sub(x, y): return x ^ y
                def mul(x, y): return mul_table[x][y]
            else:
                def add(x, y): return (x + y) % q
                def sub(x, y): return (x - y) % q
                def mul(x, y): return (x * y) % q

            rule_a = [0] * (d + 1)
            rule_b = [0] * (d + 1)
            rule_a[2], rule_b[2] = 1, 1
            rule_a[3], rule_b[3] = 2, 1
            rule_a[4], rule_b[4] = 1, 2
            for c in range(5, d + 1):
                r = (c - 5) % 4
                c0 = c - r
                if r == 0: a, b = 1, c0 - 2
                elif r == 1: a, b = c0 - 1, 1
                elif r == 2: a, b = c0, 1
                else: a, b = 1, c0 + 1
                rule_a[c] = a
                rule_b[c] = b

            def compute_neighbours(vertex):
                side, x = vertex
                result = []
                for first in range(q):
                    y = [0] * (d + 1)
                    y[1] = first
                    if side == 0:
                        for c in range(2, d + 1):
                            a, b = rule_a[c], rule_b[c]
                            y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        for c in range(2, d + 1):
                            a, b = rule_a[c], rule_b[c]
                            y[c] = sub(x[c], mul(x[a], y[b]))
                    result.append((1 - side, tuple(y)))
                return result

            start = (0, (0,) * (d + 1))
            index = {start: 0}
            order = [start]
            edges = []
            
            for idx in range(len(order)):
                if time.time() > deadline:
                    break
                vertex = order[idx]
                for other in compute_neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
            
            return len(order), edges

        return get_luw_graph()

    # Main: voltage graph approach for k=4, g=10
    # Base: complement of Q3
    n_base = 8
    base_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            d = bin(i ^ j).count('1')
            if d >= 2:  # distance 2 or 3
                base_edges.append((i, j))
    # 12 edges in base graph, each vertex degree 4
    
    m = 40  # target group size, n = 320
    n_total = n_base * m
    
    # Precompute for quick girth checking
    # Build adjacency of base
    base_adj = [[] for _ in range(n_base)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
    
    def check_girth(voltages, max_depth=9):
        """Check if lift has girth < max_depth using BFS from all vertices.
        Returns True if girth >= max_depth."""
        n = n_base * m
        # Build full adjacency
        adj = [[] for _ in range(n)]
        for u, v in base_edges:
            vu = voltages[(u, v)]
            for g1 in range(m):
                g2 = (g1 + vu) % m
                u_id = u * m + g1
                v_id = v * m + g2
                adj[u_id].append(v_id)
                adj[v_id].append(u_id)
        
        # BFS from each vertex, looking for back edges creating cycles < max_depth
        for start in range(n):
            if time.time() > deadline - 5:
                return None  # timeout
            dist = [-1] * n
            parent = [-1] * n
            queue = [start]
            dist[start] = 0
            parent[start] = -2
            
            while queue:
                curr = queue.pop(0)
                if dist[curr] >= max_depth - 1:
                    continue
                for nbr in adj[curr]:
                    if dist[nbr] == -1:
                        dist[nbr] = dist[curr] + 1
                        parent[nbr] = curr
                        queue.append(nbr)
                    elif nbr != parent[curr]:
                        cycle_len = dist[curr] + dist[nbr] + 1
                        if cycle_len < max_depth:
                            return False
        return True
    
    # Try to find voltages using random search with some structure
    # Key insight: for girth 10, we need to avoid closed walks of length 4,6,8
    # in the non-backtracking sense
    
    random.seed(42)
    
    # Start with random voltages and try to fix short cycles
    best_result = None
    
    # Try different m values starting from smaller
    for m_try in [40]:
        if time.time() > deadline - 10:
            break
        n_t = n_base * m_try
        
        # Random initialization
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, m_try - 1)
            voltages[(v, u)] = (m_try - voltages[(u, v)]) % m_try
        
        # Iterate: find short cycles and adjust voltages
        for iteration in range(5000):
            if time.time() > deadline - 15:
                break
            
            result = check_girth(voltages, max_depth=g)
            if result is None:
                break  # timeout
            if result:
                # Found valid assignment
                # Build final edge list
                edges = []
                for u, v in base_edges:
                    vu = voltages[(u, v)]
                    for g1 in range(m_try):
                        g2 = (g1 + vu) % m_try
                        u_id = u * m_try + g1
                        v_id = v * m_try + g2
                        if u_id < v_id:
                            edges.append((u_id, v_id))
                best_result = (n_t, edges)
                break
            
            # Pick a random edge and perturb its voltage
            e = random.choice(base_edges)
            old_v = voltages[e]
            new_v = (old_v + random.randint(1, m_try - 1)) % m_try
            voltages[e] = new_v
            voltages[(e[1], e[0])] = (m_try - new_v) % m_try
        
        if best_result:
            break
    
    if best_result:
        return best_result
    
    # Fallback to LUW if voltage search failed
    return get_luw_graph()
