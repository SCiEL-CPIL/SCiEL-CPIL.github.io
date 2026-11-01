import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100

    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    adj = {i: [] for i in range(10)}
    for u, v in petersen_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Enumerate cycles
    cycles = []
    def find_cycles(start, current, path, visited):
        if len(path) > 15:
            return
        if len(path) >= 5 and current == start:
            min_idx = path.index(min(path))
            canon = path[min_idx:] + path[:min_idx]
            if canon not in cycles:
                cycles.append(canon)
            return
        for neighbor in adj[current]:
            if neighbor == start and len(path) >= 5:
                find_cycles(start, neighbor, path, visited)
            elif neighbor not in visited:
                visited.add(neighbor)
                find_cycles(start, neighbor, path + [neighbor], visited)
                visited.remove(neighbor)
                
    for s in range(10):
        find_cycles(s, s, [s], {s})
        
    unique_cycles = []
    seen = set()
    for c in cycles:
        min_val = min(c)
        idx = c.index(min_val)
        c1 = c[idx:] + c[:idx]
        c2 = list(reversed(c1))
        idx2 = c2.index(min_val)
        c2 = c2[idx2:] + c2[:idx2]
        key = tuple(min(c1, c2))
        if key not in seen:
            seen.add(key)
            unique_cycles.append(key)
            
    edge_map = {}
    for i, (u, v) in enumerate(petersen_edges):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i
        
    # Precompute cycle edges
    cycle_edges = []
    for c in unique_cycles:
        edges_in_cycle = []
        for i in range(len(c)):
            u = c[i]
            v = c[(i+1) % len(c)]
            edges_in_cycle.append(edge_map[(u, v)])
        cycle_edges.append(edges_in_cycle)
        
    # Map edge index to list of cycles containing it
    edge_to_cycles = {i: [] for i in range(15)}
    for idx, ce in enumerate(cycle_edges):
        for e in ce:
            edge_to_cycles[e].append(idx)

    best_graph = None
    
    for m in range(77, 200):
        if time.time() > deadline:
            break
            
        edge_indices = list(range(15))
        random.shuffle(edge_indices)
        
        voltages = [None] * 15
        cycle_sums = [0] * len(cycle_edges)
        cycle_counts = [0] * len(cycle_edges)
        
        def check_and_update(e_idx, val):
            # Update cycles containing e_idx
            for c_idx in edge_to_cycles[e_idx]:
                cycle_sums[c_idx] = (cycle_sums[c_idx] + val) % m
                cycle_counts[c_idx] += 1
                if cycle_counts[c_idx] == len(cycle_edges[c_idx]):
                    if cycle_sums[c_idx] == 0:
                        return False
            return True

        def uncheck_and_update(e_idx, val):
            for c_idx in edge_to_cycles[e_idx]:
                cycle_sums[c_idx] = (cycle_sums[c_idx] - val) % m
                cycle_counts[c_idx] -= 1

        def backtrack(depth):
            if time.time() > deadline:
                return False
            if depth == 15:
                return True
            
            e_idx = edge_indices[depth]
            vals = list(range(m))
            random.shuffle(vals)
            
            for val in vals:
                voltages[e_idx] = val
                if check_and_update(e_idx, val):
                    if backtrack(depth + 1):
                        return True
                uncheck_and_update(e_idx, val)
                voltages[e_idx] = None
                
            return False

        found = False
        for restart in range(20):
            if time.time() > deadline:
                break
            random.shuffle(edge_indices)
            for i in range(15):
                voltages[i] = None
            cycle_sums = [0] * len(cycle_edges)
            cycle_counts = [0] * len(cycle_edges)
            
            if backtrack(0):
                found = True
                break
                
        if found:
            n = 10 * m
            edges = []
            for i, (u, v) in enumerate(petersen_edges):
                for t in range(m):
                    u_lift = u * m + t
                    v_lift = v * m + (t + voltages[i]) % m
                    edges.append((u_lift, v_lift))
                    
            edge_set = set()
            valid = True
            for u, v in edges:
                if u == v:
                    valid = False
                    break
                if (u, v) in edge_set or (v, u) in edge_set:
                    valid = False
                    break
                edge_set.add((u, v))
            
            if not valid:
                continue
                
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(d != 3 for d in deg):
                continue
                
            best_graph = (n, edges)
            break
            
    if best_graph:
        return best_graph
    else:
        return _luw_fallback(k, g)

def _luw_fallback(k, g):
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
