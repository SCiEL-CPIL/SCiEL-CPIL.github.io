import random
import time
import math
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    def remaining():
        return deadline - time.time()

    # LUW fallback construction
    def _luw_graph(k, g):
        q = k
        d = max(3, (g - 2) // 2)
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

    # Fast girth verification using BFS with cycle detection
    def _verify_girth_fast(n, edges, girth_req):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # For each vertex, do BFS and detect any cycle shorter than girth_req
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                node = queue.popleft()
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                    else:
                        # Found a cycle
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < girth_req:
                            return False
        return True

    # Petersen graph: 10 vertices, 15 edges
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    n_base = 10
    n_edges_base = len(base_edges)
    
    # Build edge index lookup
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    base_adj = [[] for _ in range(n_base)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)

    # Generate all closed non-backtracking walks of length 3..g-1 in Petersen
    # Compute coefficient vectors (signed edge counts)
    def _get_walk_constraints(max_len):
        constraints = set()
        for start_v in range(n_base):
            # DFS on (current_vertex, prev_vertex, length, coeff_vector)
            stack = []
            for next_v in base_adj[start_v]:
                e_id = edge_index[(start_v, next_v)]
                u, v = base_edges[e_id]
                sign = 1 if u == start_v else -1
                coeff = [0] * n_edges_base
                coeff[e_id] = sign
                stack.append((next_v, start_v, 1, coeff))
            
            while stack:
                curr_v, prev_v, length, coeff = stack.pop()
                if length >= max_len:
                    continue
                if curr_v == start_v and length >= 3:
                    # Store normalized constraint
                    c = tuple(coeff)
                    # Normalize: make first non-zero positive
                    for i in range(n_edges_base):
                        if c[i] != 0:
                            if c[i] < 0:
                                c = tuple(-x for x in c)
                            break
                    constraints.add(c)
                
                for next_v in base_adj[curr_v]:
                    if next_v == prev_v:
                        continue
                    e_id = edge_index[(curr_v, next_v)]
                    u, v = base_edges[e_id]
                    sign = 1 if u == curr_v else -1
                    new_coeff = coeff[:]
                    new_coeff[e_id] += sign
                    stack.append((next_v, curr_v, length + 1, new_coeff))
        return list(constraints)

    closed_walk_constraints = _get_walk_constraints(g)
    
    # Build constraint matrix (sparse representation for speed)
    # Each constraint is a tuple of (edge_id, coefficient) for non-zero entries
    sparse_constraints = []
    for c in closed_walk_constraints:
        sparse = [(i, c[i]) for i in range(n_edges_base) if c[i] != 0]
        if sparse:
            sparse_constraints.append(sparse)
    
    # Remove duplicates
    seen = set()
    unique_constraints = []
    for sc in sparse_constraints:
        key = tuple(sorted(sc))
        if key not in seen:
            seen.add(key)
            unique_constraints.append(sc)
    sparse_constraints = unique_constraints

    # Dense matrix for numpy operations
    C_dense = np.zeros((len(sparse_constraints), n_edges_base), dtype=np.int64)
    for i, sc in enumerate(sparse_constraints):
        for eid, c in sc:
            C_dense[i, eid] = c

    best_n = float('inf')
    best_edges = None

    random.seed(42)
    np.random.seed(42)

    # Moore bound: n >= 2 + k*(k-1)^(g/2-1) for even g
    # For k=3, g=16: 2 + 3*2^7 = 2 + 384 = 386, but since n must be multiple of 10, n >= 390
    # Actually the Moore bound gives 512 as stated in the problem
    # n = 10*m, so m >= 52 (since 10*51=510 < 512)
    
    # Try m values starting from minimum feasible
    m_candidates = []
    for m in range(52, 120):
        m_candidates.append(m)
    for m in range(120, 200, 2):
        m_candidates.append(m)
    for m in range(200, 300, 5):
        m_candidates.append(m)

    def check_voltage_valid(volts, m):
        """Check if voltage assignment satisfies all walk constraints."""
        v = np.array(volts, dtype=np.int64)
        residues = np.mod(C_dense @ v, m)
        return np.all(residues != 0)

    def construct_lift(volts, m):
        """Construct the lifted graph from voltage assignment."""
        n = n_base * m
        edges = []
        for i, (u, v) in enumerate(base_edges):
            vol = volts[i]
            for t in range(m):
                u_lift = u * m + t
                v_lift = v * m + (t + vol) % m
                edges.append((u_lift, v_lift))
        return n, edges

    def try_voltage_search(m, max_tries):
        """Try to find valid voltage assignment for given m."""
        if m < 2:
            return None
        
        # Try structured assignments first
        # Arithmetic progression: v[i] = a*(i+1) + b mod m
        for a in range(1, min(m, 20)):
            for b in range(0, min(m, 10)):
                volts = [(a * (i + 1) + b) % m for i in range(n_edges_base)]
                # Ensure gcd condition (graph is connected)
                g_val = m
                for v in volts:
                    g_val = math.gcd(g_val, v)
                    if g_val == 1:
                        break
                if g_val != 1:
                    continue
                if check_voltage_valid(volts, m):
                    return volts
        
        # Random search
        for _ in range(max_tries):
            if remaining() < 5.0:
                break
            volts = [random.randint(1, m - 1) for _ in range(n_edges_base)]
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
                if g_val == 1:
                    break
            if g_val != 1:
                continue
            if check_voltage_valid(volts, m):
                return volts
        
        return None

    for m in m_candidates:
        if remaining() < 15.0:
            break
        candidate_n = n_base * m
        if candidate_n >= best_n:
            continue
        
        max_tries = 50000 if m < 60 else 30000
        volts = try_voltage_search(m, max_tries)
        
        if volts is not None:
            n, edges = construct_lift(volts, m)
            if _verify_girth_fast(n, edges, g):
                best_n = n
                best_edges = edges
                break  # Found smallest valid n, stop
    
    if best_edges is not None:
        return best_n, best_edges

    # Fallback to LUW
    return _luw_graph(k, g)
