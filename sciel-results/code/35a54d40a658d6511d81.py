import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def luw_construct(k, g, d_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
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

    def verify_girth(n, edges, target_girth):
        """Verify girth >= target_girth using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                v = queue.popleft()
                if dist[v] >= target_girth // 2:
                    continue
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif w != parent[v] and v != parent[w]:
                        cycle_len = dist[v] + dist[w] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    def voltage_graph_search(k, g, time_limit):
        """
        Voltage graph / covering graph approach.
        Use K_{3,3} as base (6 vertices, 3-regular, girth 4).
        Lift over Z_m to get 6m vertices.
        Optimize voltages to maximize girth of the lift.
        """
        # K_{3,3} base: vertices 0-2 (left), 3-5 (right)
        # Edges: (0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)
        base_edges = [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
        n_base = 6
        
        def build_lift(m, voltages):
            """Build the 6m-vertex lift graph."""
            n = n_base * m
            edges = []
            for (u, v) in base_edges:
                # Voltage on edge u->v
                w = voltages[(u, v)]
                # u at level i connects to v at level (i + w) % m
                for i in range(m):
                    u_node = u * m + i
                    v_node = v * m + ((i + w) % m)
                    edges.append((u_node, v_node))
            return n, edges

        def compute_girth_fast(n, edges, max_check):
            """Return the actual girth or max_check+1 if no cycle shorter found."""
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            # Check degree
            for u in range(n):
                if len(adj[u]) != k:
                    return 0
            # BFS from each vertex, track shortest cycle
            best_girth = max_check + 1
            half = max_check // 2
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                found_short = False
                while queue:
                    v = queue.popleft()
                    if dist[v] >= half:
                        continue
                    for w in adj[v]:
                        if dist[w] == -1:
                            dist[w] = dist[v] + 1
                            parent[w] = v
                            queue.append(w)
                        elif w != parent[v] and v != parent[w]:
                            cycle_len = dist[v] + dist[w] + 1
                            if cycle_len < best_girth:
                                best_girth = cycle_len
                                if best_girth <= max_check:
                                    found_short = True
                if found_short:
                    return best_girth
            return best_girth

        # Try different m values
        best_result = None
        best_girth_found = 0
        
        # m determines graph size: 6m vertices
        # For girth 16, we need m large enough
        # Moore bound for (3,16): 2*(1+3+3^2+3^3+3^4) = 2*(1+3+9+27+81) = 216
        # So m >= 36 at minimum (6*36=216)
        # Try m from 36 upward
        for m in range(36, 200):
            if time.time() > time_limit:
                break
            n = 6 * m
            # Initialize with random voltages
            best_local_girth = 0
            best_local_voltages = None
            
            # Simulated annealing on voltage assignments
            n_edges = len(base_edges)
            current_voltages = {}
            for e in base_edges:
                current_voltages[e] = random.randint(0, m - 1)
            
            # Evaluate initial
            _, init_edges = build_lift(m, current_voltages)
            init_girth = compute_girth_fast(n, init_edges, g)
            best_local_girth = init_girth
            best_local_voltages = dict(current_voltages)
            
            if best_local_girth >= g:
                if best_result is None or n < best_result[0]:
                    best_result = (n, init_edges, m)
                break  # Found valid graph with this m
            
            # Try a few random restarts with SA
            for restart in range(3):
                if time.time() > time_limit:
                    break
                current_voltages = {}
                for e in base_edges:
                    current_voltages[e] = random.randint(0, m - 1)
                
                temp = 1.0
                cooling = 0.95
                steps = 500
                cur_girth = 0
                _, cur_edges = build_lift(m, current_voltages)
                cur_girth = compute_girth_fast(n, cur_edges, g)
                best_g = cur_girth
                best_v = dict(current_voltages)
                
                for step in range(steps):
                    if time.time() > time_limit:
                        break
                    # Pick a random edge and change its voltage
                    e = random.choice(base_edges)
                    old_v = current_voltages[e]
                    new_v = random.randint(0, m - 1)
                    if new_v == old_v:
                        continue
                    current_voltages[e] = new_v
                    
                    _, new_edges = build_lift(m, current_voltages)
                    new_girth = compute_girth_fast(n, new_edges, g)
                    
                    delta = new_girth - cur_girth
                    if delta >= 0 or random.random() < pow(2.718281828, delta / temp):
                        cur_girth = new_girth
                        if cur_girth > best_g:
                            best_g = cur_girth
                            best_v = dict(current_voltages)
                    else:
                        current_voltages[e] = old_v
                    
                    temp *= cooling
                    if best_g >= g:
                        break
                
                if best_g > best_local_girth:
                    best_local_girth = best_g
                    best_local_voltages = best_v
                
                if best_local_girth >= g:
                    _, final_edges = build_lift(m, best_local_voltages)
                    if best_result is None or n < best_result[0]:
                        best_result = (n, final_edges, m)
                    break
        
        if best_result is not None:
            return best_result[0], best_result[1]
        return None

    if k != 3 or g != 16:
        return luw_construct(k, g)

    # First try voltage graph approach (structurally different)
    vg_result = voltage_graph_search(k, g, deadline - 10)
    if vg_result is not None:
        n, edges = vg_result
        if verify_girth(n, edges, g):
            return n, edges

    # Fallback: try LUW with various d
    best_n = float('inf')
    best_edges = None
    for d in range(7, 12):
        if time.time() > deadline:
            break
        n, edges = luw_construct(k, g, d_override=d)
        if n < best_n:
            if verify_girth(n, edges, g):
                best_n = n
                best_edges = edges

    if best_edges is not None:
        return best_n, best_edges

    return luw_construct(k, g)
