import random
import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

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

    # Verify girth of a graph using non-backtracking BFS from every vertex
    def _verify_girth(n, edges, girth_req):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Non-backtracking walk check from every vertex
        for start in range(n):
            # State: (vertex, prev_vertex, length)
            # We want to detect any closed NB walk of length < girth_req
            # Use a set to track visited (vertex, prev) states at each depth
            # Actually, we need to detect if we can return to start via NB walk
            # BFS: (curr, prev, dist)
            visited = set()  # (curr, prev) pairs
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited.add((nb, start))
            while queue:
                curr, prev, dist = queue.popleft()
                if dist >= girth_req:
                    continue
                if curr == start and dist >= 3:
                    return False  # Found cycle shorter than girth_req
                for nb in adj[curr]:
                    if nb == prev:
                        continue  # Non-backtracking
                    if (nb, curr) not in visited:
                        visited.add((nb, curr))
                        queue.append((nb, curr, dist + 1))
        return True

    # Petersen graph base
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    base_adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(base_edges):
        base_adj[u].append((v, i))
        base_adj[v].append((u, i))

    # Generate all closed non-backtracking walks of length 3..g-1 in Petersen
    def _get_closed_nb_walks(base_edges, base_adj, n_base, max_len):
        walks = set()
        for start_v in range(n_base):
            # BFS: (curr_v, prev_e_id, length, path)
            queue = []
            for next_v, e_id in base_adj[start_v]:
                u, v = base_edges[e_id]
                sign = 1 if u == start_v else -1
                queue.append((next_v, e_id, 1, [(e_id, sign)]))
            while queue:
                curr_v, prev_e, length, path = queue.pop(0)
                if length >= max_len:
                    continue
                if curr_v == start_v and length >= 3:
                    walks.add(tuple(path))
                for next_v, e_id in base_adj[curr_v]:
                    if e_id == prev_e:
                        continue
                    u, v = base_edges[e_id]
                    sign = 1 if u == curr_v else -1
                    new_path = path + [(e_id, sign)]
                    queue.append((next_v, e_id, length + 1, new_path))
        return list(walks)

    closed_walks = _get_closed_nb_walks(base_edges, base_adj, n_base, g)

    # Precompute coefficient vectors
    n_edges_base = len(base_edges)
    walk_coeffs = []
    for w in closed_walks:
        coeffs = [0] * n_edges_base
        for e_id, sign in w:
            coeffs[e_id] += sign
        # Skip trivially zero (shouldn't happen for NB walks in simple graph)
        if any(c != 0 for c in coeffs):
            walk_coeffs.append(coeffs)

    best_n = float('inf')
    best_edges = None

    # Search for voltage assignments
    # Moore bound for k=3, g=16 is 512, so n = 10*m >= 512 => m >= 52
    # Try m from 52 upward
    moore_bound = 2 + 2 * sum(2 ** i for i in range(g // 2 - 1))  # For k=3
    # Actually Moore bound for (3, 16): 2 + 2*(1+2+4+8+16+32+64) = 2 + 2*127 = 256? 
    # No: Moore bound n >= 2 + k * sum_{i=0}^{g/2-1} (k-1)^i = 2 + 3*(1+2+4+8+16+32) = 2+3*63=191?
    # Let me just use: for k=3, g=16, lower bound is 512 per the task description.
    # So m >= ceil(512/10) = 52.
    
    # Try various m values
    m_candidates = list(range(52, 120)) + list(range(120, 300, 5)) + list(range(300, 500, 10))
    
    random.seed(42)
    
    for m in m_candidates:
        if time.time() > deadline - 10.0:
            break
        candidate_n = n_base * m
        if candidate_n >= best_n:
            continue
        
        # Try multiple random voltage assignments
        max_tries = 3000 if m < 100 else 1000
        found = False
        for _ in range(max_tries):
            if time.time() > deadline - 5.0:
                break
            volts = [random.randint(1, m - 1) for _ in range(n_edges_base)]
            
            # Check connectivity: gcd of all volts and m must be 1
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
            if g_val != 1:
                continue
            
            # Check all walk conditions
            valid = True
            for coeffs in walk_coeffs:
                s = 0
                for i, c in enumerate(coeffs):
                    if c != 0:
                        s += c * volts[i]
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                # Construct the lift
                lift_edges = []
                for i, (u, v) in enumerate(base_edges):
                    vol = volts[i]
                    for t in range(m):
                        u_lift = u * m + t
                        v_lift = v * m + (t + vol) % m
                        lift_edges.append((u_lift, v_lift))
                
                # Verify girth before accepting
                if _verify_girth(candidate_n, lift_edges, g):
                    best_n = candidate_n
                    best_edges = lift_edges
                    found = True
                    break
        
        if found:
            # Since we go in increasing m order, this is optimal for this strategy
            break
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW
    return _luw_graph(k, g)
