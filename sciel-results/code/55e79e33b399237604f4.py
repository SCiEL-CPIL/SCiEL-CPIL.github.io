import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 3 and g == 16:
        result = _try_voltage_lift(k, g, deadline)
        if result:
            return result
    
    return _luw_construct(k, g)

def _luw_construct(k, g):
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

def _try_voltage_lift(k, g, deadline):
    # Petersen Graph
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5)
    ]
    n_base = 10
    base_adj = [[] for _ in range(n_base)]
    for u, v in petersen_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        
    # Find simple cycles of length 5 to 9
    simple_cycles = []
    def find_cycles(start, current, path, visited):
        if len(path) >= 5 and path[0] == start:
            # Check if it's a simple cycle back to start
            # We only record if we are closing the loop
            pass
        if len(path) > 9:
            return
        for nxt in base_adj[current]:
            if nxt == start and len(path) >= 5:
                simple_cycles.append(path[:])
            elif nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                find_cycles(start, nxt, path, visited)
                path.pop()
                visited.remove(nxt)
                
    for i in range(n_base):
        visited = {i}
        find_cycles(i, i, [i], visited)
        
    # Remove duplicates (each cycle found twice? No, we start from each node, but cycle 0-1-2-3-4-0 is found starting at 0.
    # Starting at 1, we get 1-2-3-4-0-1 which is the same cycle.
    # We can normalize by taking the min vertex as start and rotating.
    unique_cycles = set()
    for cyc in simple_cycles:
        # Normalize
        min_idx = cyc.index(min(cyc))
        norm = tuple(cyc[min_idx:] + cyc[:min_idx])
        unique_cycles.add(norm)
        
    cycles_list = list(unique_cycles)
    
    # Try different moduli
    # We want N = 10 * M < 936 => M < 93
    # M should be odd to avoid 2x=0 issues
    moduli = [m for m in range(89, 49, -2) if m > 1]
    
    for M in moduli:
        if time.time() > deadline - 10:
            break
            
        # Local search
        # Initialize random voltages
        # Edges are indexed 0..14
        # We need a consistent orientation for voltage
        # Let's orient (u,v) where u<v
        edge_indices = {}
        oriented_edges = []
        for i, (u, v) in enumerate(petersen_edges):
            if u < v:
                edge_indices[(u, v)] = i
            else:
                edge_indices[(v, u)] = i
            oriented_edges.append((u, v) if u < v else (v, u))
            
        # Map cycle to edge sequence with signs
        # For a cycle [v0, v1, ..., vk-1, v0]
        # Edge (v_i, v_{i+1})
        # If v_i < v_{i+1}, voltage is +v_e
        # Else -v_e
        
        def get_cycle_sum(voltages):
            total = 0
            for i in range(len(cycles_list)):
                cyc = cycles_list[i]
                s = 0
                for j in range(len(cyc)):
                    u = cyc[j]
                    v = cyc[(j+1) % len(cyc)]
                    if u < v:
                        s += voltages[edge_indices[(u, v)]]
                    else:
                        s -= voltages[edge_indices[(v, u)]]
                if s % M == 0:
                    return i # Return index of failed cycle
            return -1
            
        # Simulated Annealing / Random Walk
        best_voltages = None
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            voltages = [random.randint(0, M-1) for _ in range(15)]
            # Ensure no single edge is 0? Not strictly necessary.
            
            # Check initial
            failed = get_cycle_sum(voltages)
            if failed == -1:
                best_voltages = voltages
                break
                
            # Local search
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                failed = get_cycle_sum(voltages)
                if failed == -1:
                    best_voltages = voltages
                    break
                # Pick a random edge in the failed cycle
                cyc = cycles_list[failed]
                edge_in_cycle = []
                for j in range(len(cyc)):
                    u = cyc[j]
                    v = cyc[(j+1) % len(cyc)]
                    if u < v:
                        edge_in_cycle.append(edge_indices[(u, v)])
                    else:
                        edge_in_cycle.append(edge_indices[(v, u)])
                e_idx = random.choice(edge_in_cycle)
                # Change voltage
                old_v = voltages[e_idx]
                new_v = (old_v + random.randint(1, M-1)) % M
                voltages[e_idx] = new_v
                # Check if improved?
                # Just accept (Random Walk)
                
        if best_voltages:
            # Construct Graph
            n_lift = n_base * M
            edges_lift = []
            for u in range(n_base):
                for v in base_adj[u]:
                    if u < v:
                        e_idx = edge_indices[(u, v)]
                        w = best_voltages[e_idx]
                    else:
                        e_idx = edge_indices[(v, u)]
                        w = -best_voltages[e_idx]
                        
                    for g_val in range(M):
                        u_lift = u * M + g_val
                        v_lift = v * M + (g_val + w) % M
                        edges_lift.append((u_lift, v_lift))
                        
            # Verify Girth
            # BFS from each vertex? Too slow for 500+ vertices?
            # 500 vertices, degree 3. BFS depth 7 is small.
            # But doing it for all vertices is 500 * (3^7) ~ 500 * 2000 = 10^6. OK.
            
            adj_lift = [[] for _ in range(n_lift)]
            for u, v in edges_lift:
                adj_lift[u].append(v)
                adj_lift[v].append(u)
                
            girth = 100
            for i in range(n_lift):
                # BFS
                dist = [-1] * n_lift
                parent = [-1] * n_lift
                q = deque()
                q.append(i)
                dist[i] = 0
                found = False
                while q and not found:
                    u = q.popleft()
                    for v in adj_lift[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            # Cycle found
                            # Length = dist[u] + dist[v] + 1
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < girth:
                                girth = cycle_len
                            if girth < g:
                                found = True
                                break
                    if found: break
                if girth < g:
                    break
            if girth >= g:
                return n_lift, edges_lift
                
    return None
