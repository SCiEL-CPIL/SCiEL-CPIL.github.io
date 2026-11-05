import time
import random
import sys
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: LUW construction from seed
    def luw_construct(k, g):
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

    if k != 3 or g != 16:
        return luw_construct(k, g)

    # Base graph: Tutte 8-cage (14 vertices, 3-regular, girth 8)
    # Vertices 0..13
    base_edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (1, 6),
        (2, 4), (2, 7), (2, 8),
        (3, 5), (3, 7), (3, 9),
        (4, 10), (4, 11), (4, 12),
        (5, 10), (5, 13), (5, 12),
        (6, 8), (6, 11), (6, 13),
        (7, 9), (7, 10), (7, 13),
        (8, 9), (8, 11), (8, 12)
    ]
    # Verify base graph properties
    adj_base = defaultdict(list)
    for u, v in base_edges:
        adj_base[u].append(v)
        adj_base[v].append(u)
        
    # Precompute cycles of length 8 to 16 in base graph
    # Since girth is 8, min cycle length is 8.
    cycles = []
    def find_cycles(start, current, path, visited, length):
        if length > 16:
            return
        if length >= 8:
            # Check if current is connected to start
            if start in adj_base[current]:
                # Canonicalize cycle to avoid duplicates
                cycle = tuple(sorted(path + [start]))
                # This is not a perfect canonicalization for cycles, but good enough for constraint checking
                # We store the path itself for voltage sum
                cycles.append(path + [start])
        for next_v in adj_base[current]:
            if next_v == start and length < 8:
                continue
            if next_v not in visited:
                visited.add(next_v)
                path.append(next_v)
                find_cycles(start, next_v, path, visited, length + 1)
                path.pop()
                visited.remove(next_v)

    for i in range(14):
        find_cycles(i, i, [i], {i}, 1)
        
    # Remove duplicate cycles (same set of edges/nodes in different order)
    unique_cycles = []
    seen = set()
    for c in cycles:
        # Canonical form: rotate so min node is first, then compare
        min_idx = c.index(min(c))
        canon = tuple(c[min_idx:] + c[:min_idx])
        if canon not in seen:
            seen.add(canon)
            unique_cycles.append(c)
            
    # Filter cycles: we only care about length <= 16
    unique_cycles = [c for c in unique_cycles if len(c) <= 16]
    
    # Sort by length descending? No, shorter cycles are more restrictive.
    
    def check_voltage_assignment(voltages):
        # voltages: dict (u,v) -> val
        for c in unique_cycles:
            l = len(c)
            if l > 16: continue
            s = 0
            for i in range(l - 1):
                u, v = c[i], c[i+1]
                # Ensure consistent direction
                if (u, v) in voltages:
                    s = (s + voltages[(u, v)])
                elif (v, u) in voltages:
                    s = (s - voltages[(v, u)])
                else:
                    return False
            s = s % k
            # Order of s in Z_k
            # We need ord(s) >= floor(16/l) + 1
            req_order = (16 // l) + 1
            if s == 0:
                return False
            # Check order
            curr = s
            for _ in range(req_order - 1):
                curr = (curr + s) % k
                if curr == 0:
                    return False
            # If we haven't hit 0 in req_order-1 steps, order >= req_order
            # Wait, order is the smallest m such that m*s = 0.
            # If (req_order-1)*s != 0, then order >= req_order.
        return True

    best_n = float('inf')
    best_edges = None
    
    # Moore bound for n(3,16) is 512.
    # Base size 14. k must be at least ceil(512/14) = 37.
    # Try k from 37 to 100.
    for k_val in range(37, 150):
        if time.time() > deadline:
            break
        # Try random assignments
        for attempt in range(1000):
            if time.time() > deadline:
                break
            voltages = {}
            for u, v in base_edges:
                voltages[(u, v)] = random.randint(0, k_val - 1)
                voltages[(v, u)] = (-voltages[(u, v)]) % k_val
            
            if check_voltage_assignment(voltages):
                # Build lift
                n_lift = 14 * k_val
                lift_edges = []
                for u, v in base_edges:
                    for g in range(k_val):
                        u_id = u * k_val + g
                        v_id = v * k_val + (g + voltages[(u, v)]) % k_val
                        lift_edges.append((u_id, v_id))
                
                # Check connectivity and find largest component
                adj_lift = defaultdict(list)
                for u, v in lift_edges:
                    adj_lift[u].append(v)
                    adj_lift[v].append(u)
                
                visited = [False] * n_lift
                max_comp_size = 0
                max_comp_vertices = []
                for i in range(n_lift):
                    if not visited[i]:
                        # BFS
                        q = deque([i])
                        visited[i] = True
                        comp = []
                        while q:
                            node = q.popleft()
                            comp.append(node)
                            for nei in adj_lift[node]:
                                if not visited[nei]:
                                    visited[nei] = True
                                    q.append(nei)
                        if len(comp) > max_comp_size:
                            max_comp_size = len(comp)
                            max_comp_vertices = comp
                
                if max_comp_size < 512:
                    continue # Too small, likely invalid or not enough vertices
                
                # Extract edges for the largest component
                comp_set = set(max_comp_vertices)
                comp_edges = [(u, v) for u, v in lift_edges if u in comp_set and v in comp_set]
                
                if len(comp_edges) == 3 * max_comp_size // 2:
                    if max_comp_size < best_n:
                        best_n = max_comp_size
                        best_edges = comp_edges
                        # Update k to current k_val? No, we want smallest n.
                        # If we found a valid graph, we can break?
                        # No, we want to minimize n.
                        # Smaller k_val gives smaller n.
                        # So we iterate k_val ascending.
                        # The first valid k_val gives the smallest n for this base.
                        # But maybe a larger k_val with a different base?
                        # We are using fixed base.
                        # So we can break out of k_val loop?
                        # No, maybe a smaller component from a larger k_val?
                        # No, component size <= 14*k_val.
                        # So 14*37 < 14*38.
                        # So the first valid k_val gives the smallest possible n for this base.
                        break
        
        if best_edges is not None:
            break

    if best_edges is not None:
        return best_n, best_edges
    else:
        return luw_construct(k, g)

# EVOLVE-BLOCK-END
