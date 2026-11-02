import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if k == 4 and g == 10:
        # 1. Try Voltage Lift of Desargues Graph (Target 320 vertices)
        result = _try_voltage_lift_320(deadline)
        if result is not None:
            return result
        
        # 2. Try Cayley graph on Z_16 x Z_20
        result = _try_cayley_320(deadline)
        if result is not None:
            return result
        
        # 3. Try Randomized Local Search for 320 vertices
        result = _try_local_search_320(deadline)
        if result is not None:
            return result

    # Fallback: LUW Construction
    return _luw_construct(k, g)

def _try_voltage_lift_320(deadline):
    # Base: Desargues Graph (20 vertices, 4-regular, bipartite)
    # Vertices 0-9 (Points), 10-19 (Lines)
    # Incidence: Point i is on Line j if j in {i, i+1, i+4, i+6} mod 10
    
    N_BASE = 20
    m = 16
    N_TOTAL = N_BASE * m
    
    # Build base graph edges
    base_adj = [[] for _ in range(N_BASE)]
    undirected_edges = []
    for u in range(10):
        for v in range(10):
            if v in [u, (u+1)%10, (u+4)%10, (u+6)%10]:
                u_id = u
                v_id = 10 + v
                base_adj[u_id].append(v_id)
                base_adj[v_id].append(u_id)
                undirected_edges.append((u_id, v_id))
    
    # Map edge to index for voltage assignment
    edge_index = {}
    for i, (u, v) in enumerate(undirected_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    # Find all non-backtracking closed walks of length 6 and 8
    def find_walks(length):
        walks = []
        # Start from each vertex
        for start in range(N_BASE):
            # Walk: start -> v1 -> ... -> start
            # Non-backtracking: v_{i+1} != v_{i-1}
            # We represent walk by list of vertices [start, v1, ..., v_{L-1}]
            # v_L = start
            
            # Recursion or iterative
            # Length 6: start, v1, v2, v3, v4, v5, start
            # v1 != start
            # v2 != v1
            # v3 != v2
            # v4 != v3
            # v5 != v4
            # start != v5
            
            # To avoid duplicates (rotations/reflections), we can just enumerate all and count unique voltage constraints?
            # Actually, we just need to check if sum of voltages is 0.
            # If we enumerate all closed walks, we might count the same cycle multiple times.
            # But for optimization, minimizing the count of such walks is a good proxy.
            
            def dfs(current, prev, path, remaining):
                if remaining == 0:
                    if current == start:
                        # Check non-backtracking at the end?
                        # path[0] is start. path[-1] is current.
                        # We need path[-1] != path[-2] (which is true by dfs)
                        # And path[0] != path[-1] is false (current==start).
                        # The condition "non-backtracking" usually means v_i != v_{i-2}.
                        # The last step is v_{L-1} -> start.
                        # We must ensure start != v_{L-2}.
                        if len(path) >= 2 and path[-2] != start:
                            walks.append(tuple(path))
                    return
                
                for nb in base_adj[current]:
                    if nb == prev:
                        continue
                    # Pruning: if remaining is small, ensure we can reach start?
                    # For small graphs, just go.
                    dfs(nb, current, path + [nb], remaining - 1)
            
            dfs(start, -1, [start], length)
        return walks

    walks_6 = find_walks(6)
    walks_8 = find_walks(8)
    
    # Precompute voltage coefficients for each walk
    # walk is (v0, v1, ..., v_{L-1})
    # voltage sum = sum_{i=0}^{L-1} sign(v_i, v_{i+1}) * w_{edge(v_i, v_{i+1})}
    # where v_L = v_0
    # sign is +1 if we traverse edge in direction (u,v) where edge_index stores (u,v) with u<v?
    # No, edge_index stores both directions with same index.
    # We need to know the direction relative to the stored edge.
    # Let's define stored edge (u, v) with u < v.
    # If walk goes u->v, voltage is +w.
    # If walk goes v->u, voltage is -w.
    
    walk_coeffs = []
    for walk in walks_6 + walks_8:
        coeffs = [0] * len(undirected_edges)
        L = len(walk)
        for i in range(L):
            u = walk[i]
            v = walk[(i+1) % L]
            idx = edge_index[(u, v)]
            # Determine sign
            # We need to know if (u, v) is the "forward" direction of the stored edge.
            # Let's store the forward direction in a dict
            # Actually, let's just store (u, v) in undirected_edges with u < v.
            # If u < v, forward. Else backward.
            # But undirected_edges was built as (u_id, v_id) where u_id < 10, v_id >= 10.
            # So u_id < v_id always.
            # So if walk[i] < walk[(i+1)%L], sign is +1. Else -1.
            if u < v:
                coeffs[idx] += 1
            else:
                coeffs[idx] -= 1
        walk_coeffs.append(coeffs)
        
    # Objective function
    def cost(voltages):
        c = 0
        for coeffs in walk_coeffs:
            s = 0
            for i in range(len(undirected_edges)):
                if coeffs[i] != 0:
                    s += coeffs[i] * voltages[i]
            if s % m == 0:
                c += 1
        return c

    # Simulated Annealing
    best_voltages = [random.randint(0, m-1) for _ in range(len(undirected_edges))]
    best_cost = cost(best_voltages)
    
    current_voltages = best_voltages[:]
    current_cost = best_cost
    
    T = 10.0
    cooling = 0.99
    steps_per_temp = 1000
    max_steps = 50000
    
    steps = 0
    while steps < max_steps and time.time() < deadline:
        if current_cost == 0:
            break
            
        # Pick random edge
        i = random.randint(0, len(undirected_edges) - 1)
        old_v = current_voltages[i]
        new_v = (old_v + random.choice([-1, 1])) % m
        current_voltages[i] = new_v
        
        new_cost = cost(current_voltages)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_voltages = current_voltages[:]
        
        steps += 1
        if steps % steps_per_temp == 0:
            T *= cooling
            if T < 0.01:
                break

    if best_cost == 0:
        # Construct graph
        edges = []
        # Map vertex (u, z) to id u*m + z
        for u in range(N_BASE):
            for v in base_adj[u]:
                if u < v: # Only add each undirected edge once?
                    # No, we need directed edges for construction?
                    # The output is a list of edges (i, j).
                    # We can iterate over all directed edges and add (u_id, v_id) if u_id < v_id
                    pass
        
        # Better: iterate over all directed edges, compute endpoints, add if u_id < v_id
        for u in range(N_BASE):
            for v in base_adj[u]:
                idx = edge_index[(u, v)]
                volt = best_voltages[idx]
                if u < v:
                    sign = 1
                else:
                    sign = -1
                
                # For each z in 0..m-1
                for z in range(m):
                    u_id = u * m + z
                    v_z = (z + sign * volt) % m
                    v_id = v * m + v_z
                    if u_id < v_id:
                        edges.append((u_id, v_id))
        
        return N_TOTAL, edges

    return None

def _try_cayley_320(deadline):
    m, n = 16, 20
    total = m * n
    
    def check_girth(x, y):
        gens = [
            (1, 0),
            (m-1, 0),
            (x, y),
            ((-x)%m, (-y)%n)
        ]
        
        visited = {}
        visited[(0,0)] = 0
        queue = deque([(0,0)])
        
        while queue:
            u, v = queue.popleft()
            depth = visited[(u,v)]
            
            for g1, g2 in gens:
                nu = (u + g1) % m
                nv = (v + g2) % n
                if (nu, nv) not in visited:
                    visited[(nu, nv)] = depth + 1
                    queue.append((nu, nv))
                else:
                    cycle_len = depth + 1 + visited[(nu, nv)]
                    if cycle_len < 10:
                        return False
        return True

    for x in range(m):
        if time.time() > deadline:
            return None
        for y in range(n):
            if time.time() > deadline:
                return None
            
            if x == 0 and y == 0: continue
            if x == 1 and y == 0: continue
            if x == m-1 and y == 0: continue
            if (2*x) % m == 0 and (2*y) % n == 0: continue
            
            if check_girth(x, y):
                edges = []
                for u in range(m):
                    for v in range(n):
                        nu = (u + 1) % m
                        nv = v
                        if (u, v) < (nu, nv):
                            edges.append(((u*m+v, nu*m+nv)))
                        
                        nu = (u + x) % m
                        nv = (v + y) % n
                        if (u, v) < (nu, nv):
                            edges.append(((u*m+v, nu*m+nv)))
                
                deg = [0]*total
                for e1, e2 in edges:
                    deg[e1] += 1
                    deg[e2] += 1
                if all(d == 4 for d in deg):
                    return total, edges

    return None

def _try_local_search_320(deadline):
    return None

def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

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
