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
        # Edges:
        # Outer pentagon: 0-1, 1-2, 2-3, 3-4, 4-0
        # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        # Adjacency list for Petersen
        pet_adj = [[] for _ in range(10)]
        edge_to_idx = {}
        for i, (u, v) in enumerate(pet_edges):
            pet_adj[u].append((v, i))
            pet_adj[v].append((u, i))
            edge_to_idx[(u, v)] = i
            edge_to_idx[(v, u)] = i
            
        # Generate non-backtracking closed walks of length 5 to 9
        closed_walks = [] # List of tuples (edge_idx, sign)
        
        def gen_nb_walks(start, curr, prev, length, L, path):
            if length == L:
                if curr == start:
                    closed_walks.append(tuple(path))
                return
            for v, e_idx in pet_adj[curr]:
                if v == prev:
                    continue
                # Determine sign
                u_i, v_i = pet_edges[e_idx]
                if curr == u_i:
                    sign = 1
                else:
                    sign = -1
                    
                gen_nb_walks(start, v, curr, length + 1, L, path + [(e_idx, sign)])
                
        for L in range(5, 10):
            for s in range(10):
                for v, e_idx in pet_adj[s]:
                    u_i, v_i = pet_edges[e_idx]
                    if s == u_i:
                        sign = 1
                    else:
                        sign = -1
                    gen_nb_walks(s, v, s, 1, L, [(e_idx, sign)])

        # Backtracking Solver with Pruning
        n_edges = 15
        found = False
        best_voltages = None
        deadline = time.time() + 60
        
        # Precompute max index for each constraint to enable pruning
        # constraint_max_idx[i] = max edge index in closed_walks[i]
        constraint_max_idx = []
        for walk in closed_walks:
            max_idx = 0
            for e_idx, sign in walk:
                if e_idx > max_idx:
                    max_idx = e_idx
            constraint_max_idx.append(max_idx)
            
        # Sort constraints by max_idx? No, we check them when their max_idx is assigned.
        # We can just check all constraints where max_idx == current_idx.
        
        # To be safe, we check all constraints at each step if count is small, 
        # or just the ones that became fully assigned.
        
        def solve(idx, current_voltages):
            if time.time() > deadline:
                return False
            if idx == n_edges:
                # Final check (redundant if pruning worked, but safe)
                for i, walk in enumerate(closed_walks):
                    if constraint_max_idx[i] == idx:
                        total = 0
                        for e_idx, sign in walk:
                            total += sign * current_voltages[e_idx]
                        if total % 32 == 0:
                            return False
                best_voltages[:] = current_voltages
                return True
            
            # Pruning: Check constraints that are now fully assigned (max_idx == idx)
            # Note: This assumes we assign variables 0, 1, ..., 14 in order.
            # A constraint is fully assigned when we assign the variable with the highest index in it.
            for i, walk in enumerate(closed_walks):
                if constraint_max_idx[i] == idx:
                    total = 0
                    for e_idx, sign in walk:
                        total += sign * current_voltages[e_idx]
                    if total % 32 == 0:
                        return False
            
            # Symmetry breaking
            if idx == 0:
                vals = [1]
            else:
                vals = range(32)
                
            for val in vals:
                current_voltages[idx] = val
                if solve(idx + 1, current_voltages):
                    return True
            
            current_voltages[idx] = 0
            return False

        voltages = [0] * n_edges
        best_voltages = [0] * n_edges
        if solve(0, voltages):
            found = True
            best_voltages = voltages[:]

        if found:
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
            def has_girth_at_least(n, edges, g):
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                edge_set = set()
                for u, v in edges:
                    if u == v:
                        return False
                    if (v, u) in edge_set:
                        return False
                    edge_set.add((u, v))
                
                for start in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[start] = 0
                    queue = [start]
                    head = 0
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        if dist[u] >= g // 2:
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
                                        return False
                return True
                
            if has_girth_at_least(n, edges, g):
                return n, edges

    # Fallback to LUW
    return luw_construct(k, g)
