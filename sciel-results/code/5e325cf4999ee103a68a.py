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
        
        # Map edge to index for voltage assignment
        edge_to_idx = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_to_idx[(u, v)] = i
            edge_to_idx[(v, u)] = i
            
        # Build adjacency for Petersen
        pet_adj = [[] for _ in range(10)]
        for u, v in pet_edges:
            pet_adj[u].append(v)
            pet_adj[v].append(u)
            
        # Precompute all closed walks of length 5 to 9
        # Represent each as a list of (edge_idx, sign)
        closed_walks = []
        for L in range(5, 10):
            # Iterative DFS
            stack = [(s, s, 0, []) for s in range(10)]
            while stack:
                start, curr, length, path = stack.pop()
                if length == L:
                    if curr == start:
                        closed_walks.append(tuple(path))
                    continue
                for nb in pet_adj[curr]:
                    # Find edge index
                    for ei, (eu, ev) in enumerate(pet_edges):
                        if eu == curr and ev == nb:
                            stack.append((start, nb, length + 1, path + [(ei, 1)]))
                            break
                        elif ev == curr and eu == nb:
                            stack.append((start, nb, length + 1, path + [(ei, -1)]))
                            break
        
        # Remove duplicate walks (same sequence of signed edges)
        closed_walks = list(set(closed_walks))
        
        # Precompute for each walk: list of (edge_idx, sign, coeff)
        # where coeff is the net count of that edge
        walk_constraints = []
        for walk in closed_walks:
            # Aggregate: for each edge, sum up signs
            edge_sums = {}
            for ei, sign in walk:
                edge_sums[ei] = edge_sums.get(ei, 0) + sign
            # Only keep edges with non-zero sum
            constraint = [(ei, s) for ei, s in edge_sums.items() if s != 0]
            if constraint:
                walk_constraints.append(constraint)
        
        # Backtracking solver with pruning
        n_edges = 15
        voltages = [0] * n_edges
        found = False
        deadline = time.time() + 110
        
        # Precompute which constraints involve which edges for pruning
        # For each edge index, list of constraints that use it
        edge_to_constraints = [[] for _ in range(n_edges)]
        for ci, constraint in enumerate(walk_constraints):
            for ei, s in constraint:
                edge_to_constraints[ei].append(ci)
        
        # Precompute partial sums tracking
        # We'll track for each constraint: current partial sum and which edges remain
        
        def solve(idx, current_voltages):
            if time.time() > deadline:
                return False
            if idx == n_edges:
                # Check all constraints
                for constraint in walk_constraints:
                    total = 0
                    for ei, s in constraint:
                        total += s * current_voltages[ei]
                    if total % 32 == 0:
                        return False
                voltages[:] = current_voltages
                return True
            
            for val in range(32):
                if idx == 0 and val == 0:
                    continue # Symmetry breaking
                current_voltages[idx] = val
                # Pruning: check if any constraint is already violated
                # A constraint is violated if the sum of assigned parts + max possible from unassigned == 0 mod 32
                # is impossible... actually we check if sum % 32 == 0 is forced
                # Simpler: just check if any fully-assigned constraint sums to 0
                violated = False
                for ci in edge_to_constraints[idx]:
                    constraint = walk_constraints[ci]
                    total = 0
                    all_assigned = True
                    for ei, s in constraint:
                        if ei >= n_edges:
                            all_assigned = False
                            break
                        if ei > idx:
                            all_assigned = False
                            break
                        total += s * current_voltages[ei]
                    if all_assigned and total % 32 == 0:
                        violated = True
                        break
                if not violated:
                    if solve(idx + 1, current_voltages):
                        return True
            return False

        if solve(0, voltages):
            found = True
            best_voltages = voltages[:]
        else:
            # Try different symmetry breaking
            found = False
            for w0 in range(1, 32):
                if time.time() > deadline:
                    break
                voltages[0] = w0
                def solve2(idx, current_voltages):
                    if time.time() > deadline:
                        return False
                    if idx == n_edges:
                        for constraint in walk_constraints:
                            total = 0
                            for ei, s in constraint:
                                total += s * current_voltages[ei]
                            if total % 32 == 0:
                                return False
                        voltages[:] = current_voltages
                        return True
                    for val in range(32):
                        current_voltages[idx] = val
                        if solve2(idx + 1, current_voltages):
                            return True
                    return False
                if solve2(1, voltages):
                    found = True
                    best_voltages = voltages[:]
                    break

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
                    
            # Verify girth using BFS from each vertex
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Check simplicity
            edge_set = set()
            valid = True
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
                # Check degree
                for i in range(n):
                    if len(adj[i]) != 4:
                        valid = False
                        break
            
            if valid:
                # Check girth: for each start, BFS, check non-tree edges
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
