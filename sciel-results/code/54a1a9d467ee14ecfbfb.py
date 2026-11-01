import time
import random
import sys
import math
import numpy as np

def construct(k, g):
    # Deadline guard: 140s out of 150s
    deadline = time.time() + 140.0

    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(d, q):
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

    # Specific optimization for k=4, g=10
    if k == 4 and g == 10:
        def construct_320_cage():
            # Base graph: K5,5 minus perfect matching (10 vertices, 4-regular)
            # Vertices 0-4 (A), 5-9 (B)
            base_edges = []
            edge_index = {}
            adj_base = [[] for _ in range(10)]
            for i in range(5):
                for j in range(5):
                    if i != j:
                        e_idx = len(base_edges)
                        base_edges.append((i, 5 + j))
                        edge_index[(i, 5 + j)] = e_idx
                        edge_index[(5 + j, i)] = e_idx
                        adj_base[i].append(5 + j)
                        adj_base[5 + j].append(i)
            
            # Find all simple cycles of length 4, 6, 8
            def find_cycles(length):
                cycles = []
                # Start from each vertex, DFS to find cycles of given length
                # To avoid duplicates, only consider cycles where the start vertex is the smallest index
                for start in range(10):
                    stack = [(start, [start])]
                    while stack:
                        node, path = stack.pop()
                        if len(path) == length:
                            if node == start:
                                # Check if it's a simple cycle (no repeated vertices except start)
                                if len(set(path[:-1])) == length - 1:
                                    # Normalize cycle: start with smallest vertex
                                    min_idx = min(range(length), key=lambda x: path[x])
                                    norm_path = [path[(min_idx + i) % length] for i in range(length)]
                                    # Reverse to ensure canonical form (smaller neighbor first?)
                                    # Actually, just check if this cycle is already in list
                                    # Simpler: store tuple of edge indices
                                    edge_seq = []
                                    for i in range(length):
                                        u = path[i]
                                        v = path[(i + 1) % length]
                                        e_idx = edge_index.get((u, v))
                                        if e_idx is None:
                                            edge_seq = None
                                            break
                                        edge_seq.append(e_idx)
                                    if edge_seq:
                                        # Check for duplicate (reversed order)
                                        rev_seq = edge_seq[::-1]
                                        is_dup = False
                                        for existing in cycles:
                                            if set(existing) == set(edge_seq) and len(existing) == length:
                                                # Need to check if it's the same cycle
                                                # Since we normalize by smallest vertex, we might still have orientation issues
                                                # For XOR, orientation doesn't matter.
                                                is_dup = True
                                                break
                                        if not is_dup:
                                            cycles.append(edge_seq)
                            continue
                        if len(path) > length:
                            continue
                        for nei in adj_base[node]:
                            if nei == path[0] and len(path) < length:
                                continue
                            if nei in path:
                                continue
                            stack.append((nei, path + [nei]))
                return cycles

            cycles_4 = find_cycles(4)
            cycles_6 = find_cycles(6)
            cycles_8 = find_cycles(8)
            
            all_cycles = cycles_4 + cycles_6 + cycles_8
            
            # Map edge to cycles it belongs to
            edge_to_cycles = [[] for _ in range(len(base_edges))]
            for idx, cycle in enumerate(all_cycles):
                for e in cycle:
                    edge_to_cycles[e].append(idx)
            
            # Backtracking solver
            n_edges = len(base_edges)
            voltages = [None] * n_edges
            
            # Order edges by number of constraints
            edge_order = sorted(range(n_edges), key=lambda e: -len(edge_to_cycles[e]))
            
            def backtrack(pos):
                if pos == n_edges:
                    return True
                
                e = edge_order[pos]
                for v in range(32):
                    voltages[e] = v
                    
                    valid = True
                    # Check all cycles containing e
                    for c_idx in edge_to_cycles[e]:
                        cycle = all_cycles[c_idx]
                        # Check if all edges in this cycle are assigned
                        sum_val = 0
                        complete = True
                        for ce in cycle:
                            if voltages[ce] is None:
                                complete = False
                                break
                            sum_val ^= voltages[ce]
                        if complete and sum_val == 0:
                            valid = False
                            break
                    
                    if valid and backtrack(pos + 1):
                        return True
                
                voltages[e] = None
                return False
            
            if backtrack(0):
                # Build the lift
                n_lift = 10 * 32
                lift_edges = []
                
                for u, v in base_edges:
                    idx = edge_index[(u, v)]
                    w = voltages[idx]
                    
                    for x in range(32):
                        u_lift = u * 32 + x
                        v_lift = v * 32 + ((x + w) % 32)
                        lift_edges.append((u_lift, v_lift))
                
                # Verify girth
                if has_girth_at_least(n_lift, lift_edges, 10):
                    return n_lift, lift_edges
            
            return None

        result = construct_320_cage()
        if result:
            return result

        # Fall back to LUW
        d_try = 5
        n, edges = luw_construct(d_try, 4)
        if has_girth_at_least(n, edges, 10):
            return n, edges
            
        d_try = 6
        n, edges = luw_construct(d_try, 4)
        if has_girth_at_least(n, edges, 10):
            return n, edges

    # General case
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = luw_construct(d, k)
    return n, edges
