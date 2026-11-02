import time
import random
import sys
import numpy as np

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    if k == 4 and g == 10:
        # Try to construct a voltage graph lift of the Petersen graph
        # with group Z_32 to get n=320.
        
        # Petersen graph vertices 0..9
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        edge_to_idx = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_to_idx[(u, v)] = i
            edge_to_idx[(v, u)] = i
            
        pet_adj = [[] for _ in range(10)]
        for u, v in pet_edges:
            pet_adj[u].append(v)
            pet_adj[v].append(u)
            
        # Precompute all closed walks of length 5 to 9
        closed_walks = []
        for L in range(5, 10):
            stack = [(s, s, 0, []) for s in range(10)]
            while stack:
                start, curr, length, path = stack.pop()
                if length == L:
                    if curr == start:
                        closed_walks.append(tuple(path))
                    continue
                for nb in pet_adj[curr]:
                    # Find edge index
                    # Optimization: precompute edge lookup
                    # For each vertex, list of (neighbor, edge_idx, sign_from_neighbor)
                    # Actually, let's just use the loop, it's small.
                    for ei, (eu, ev) in enumerate(pet_edges):
                        if eu == curr and ev == nb:
                            stack.append((start, nb, length + 1, path + [(ei, 1)]))
                            break
                        elif ev == curr and eu == nb:
                            stack.append((start, nb, length + 1, path + [(ei, -1)]))
                            break
        
        # Remove duplicate walks
        closed_walks = list(set(closed_walks))
        
        # Precompute constraints: list of (edge_idx, coeff)
        walk_constraints = []
        for walk in closed_walks:
            edge_sums = {}
            for ei, sign in walk:
                edge_sums[ei] = edge_sums.get(ei, 0) + sign
            constraint = [(ei, s) for ei, s in edge_sums.items() if s != 0]
            if constraint:
                walk_constraints.append(constraint)
        
        # Local Search for voltages
        n_edges = 15
        group_order = 32
        deadline = time.time() + 110
        best_voltages = None
        
        # Random restarts
        for restart in range(1000):
            if time.time() > deadline:
                break
            
            random.seed(restart)
            voltages = [random.randint(0, group_order - 1) for _ in range(n_edges)]
            # Symmetry breaking: fix v[0] = 0
            voltages[0] = 0
            
            def count_violations(v):
                count = 0
                for constraint in walk_constraints:
                    total = 0
                    for ei, s in constraint:
                        total += s * v[ei]
                    if total % group_order == 0:
                        count += 1
                return count
            
            current_violations = count_violations(voltages)
            
            # Local search loop
            max_steps = 500
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                if current_violations == 0:
                    best_voltages = voltages[:]
                    break
                
                # Pick a random violated constraint
                # We need to identify which ones are violated
                violated_indices = []
                for ci, constraint in enumerate(walk_constraints):
                    total = 0
                    for ei, s in constraint:
                        total += s * voltages[ei]
                    if total % group_order == 0:
                        violated_indices.append(ci)
                
                if not violated_indices:
                    best_voltages = voltages[:]
                    break
                    
                ci = random.choice(violated_indices)
                constraint = walk_constraints[ci]
                
                best_new_violations = current_violations
                best_change = None
                
                # Try changing one variable in the constraint
                # To speed up, we can just try all values for one random variable
                # Or try all values for all variables in the constraint
                for ei, s in constraint:
                    old_val = voltages[ei]
                    for new_val in range(group_order):
                        if new_val == old_val:
                            continue
                        voltages[ei] = new_val
                        new_violations = count_violations(voltages)
                        if new_violations < best_new_violations:
                            best_new_violations = new_violations
                            best_change = (ei, new_val)
                        if new_violations == 0:
                            break
                    voltages[ei] = old_val
                    if best_violations == 0:
                        break
                
                if best_change:
                    ei, new_val = best_change
                    voltages[ei] = new_val
                    current_violations = best_new_violations
                else:
                    # No improvement, random move
                    ei, s = random.choice(constraint)
                    old_val = voltages[ei]
                    new_val = random.randint(0, group_order - 1)
                    if new_val == old_val:
                        continue
                    voltages[ei] = new_val
                    current_violations = count_violations(voltages)
            
            if best_voltages:
                break
        
        if best_voltages:
            # Construct the lift
            n = 10 * 32
            edges = []
            for u, v in pet_edges:
                idx = edge_to_idx[(u, v)]
                w = best_voltages[idx]
                for g_val in range(32):
                    u_idx = u * 32 + g_val
                    v_idx = v * 32 + (g_val + w) % 32
                    edges.append((u_idx, v_idx))
                    
            # Verify girth
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Check simplicity and degree
            valid = True
            edge_set = set()
            for u, v in edges:
                if u == v:
                    valid = False
                    break
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
            
            if valid:
                for i in range(n):
                    if len(adj[i]) != 4:
                        valid = False
                        break
            
            if valid:
                girth_ok = True
                for start in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[start] = 0
                    queue = [start]
                    head = 0
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        if dist[u] >= g - 1:
                            continue
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            else:
                                if v != parent[u] and u != parent[v]:
                                    cycle_len = dist[u] + dist[v] + 1
                                    if cycle_len < g:
                                        girth_ok = False
                                        break
                    if not girth_ok:
                        break
                
                if girth_ok:
                    return n, edges

    # Fallback to LUW
    return luw_construct(k, g)
