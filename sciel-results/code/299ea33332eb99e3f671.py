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
        # Try to construct the 320-vertex cage using voltage graphs
        # Base graph: K5,5 minus perfect matching (10 vertices, 4-regular)
        # Voltage group: Z2^5 (32 elements)
        
        def construct_320_cage():
            # Base graph vertices 0-4 (A), 5-9 (B)
            # Edges (i, 5+j) for i != j
            base_edges = []
            edge_index = {}
            for i in range(5):
                for j in range(5):
                    if i != j:
                        edge_index[(i, 5 + j)] = len(base_edges)
                        base_edges.append((i, 5 + j))
            
            # Precompute 4-cycles constraints
            # A 4-cycle in K5,5 minus matching:
            # i1 - j1 - i2 - j2 - i1
            # Edges: (i1, 5+j1), (i2, 5+j1), (i2, 5+j2), (i1, 5+j2)
            # Constraints: i1 != i2, j1 != j2, i1 != j1, i2 != j1, i2 != j2, i1 != j2
            
            # For each 4-cycle, we have a constraint: w1 ^ w2 ^ w3 ^ w4 != 0
            # This is a system of constraints over GF(2)^5
            
            # Let's use a smarter approach: assign voltages to satisfy all 4-cycle constraints
            # Then check 6-cycles and 8-cycles
            
            # Number of 4-cycles in K5,5 minus matching:
            # Choose 2 from 5 A-vertices and 2 from 5 B-vertices, with matching condition
            # For each pair of A-vertices {i1, i2} and B-vertices {j1, j2},
            # we need i1 != j1, i2 != j1, i1 != j2, i2 != j2
            # This means the 2x2 matrix [[i1,j1],[i2,j2]] has no diagonal matches
            
            # Let's enumerate all 4-cycles
            four_cycles = []
            for i1 in range(5):
                for i2 in range(i1 + 1, 5):
                    for j1 in range(5):
                        if j1 == i1 or j1 == i2:
                            continue
                        for j2 in range(j1 + 1, 5):
                            if j2 == i1 or j2 == i2:
                                continue
                            # Edges: (i1, 5+j1), (i2, 5+j1), (i2, 5+j2), (i1, 5+j2)
                            e1 = edge_index.get((i1, 5 + j1))
                            e2 = edge_index.get((i2, 5 + j1))
                            e3 = edge_index.get((i2, 5 + j2))
                            e4 = edge_index.get((i1, 5 + j2))
                            if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                                four_cycles.append((e1, e2, e3, e4))
            
            # Now we need to assign 5-bit vectors to 20 edges such that
            # for every 4-cycle, the XOR of the 4 edge voltages is non-zero
            # This is equivalent to: the sum of any 4-cycle in the cycle space is non-zero
            
            # A sufficient condition: assign voltages such that the map from edges to GF(2)^5
            # is injective on the cycle space modulo the 4-cycles
            # Actually, we need: for every 4-cycle C, sum_{e in C} w_e != 0
            
            # Let's try a constructive approach:
            # Assign w_{(i,j)} = a_i ^ b_j for some vectors a_i, b_j in GF(2)^5
            # Then for a 4-cycle (i1,j1), (i2,j1), (i2,j2), (i1,j2):
            # sum = (a_i1 ^ b_j1) ^ (a_i2 ^ b_j1) ^ (a_i2 ^ b_j2) ^ (a_i1 ^ b_j2)
            #     = (a_i1 ^ a_i2 ^ a_i2 ^ a_i1) ^ (b_j1 ^ b_j1 ^ b_j2 ^ b_j2)
            #     = 0 ^ 0 = 0
            # This always gives 0! So this form doesn't work.
            
            # Try w_{(i,j)} = a_i ^ b_j ^ c_{i,j} where c_{i,j} is a "twist"
            # This is getting complex. Let's use random search with better heuristics.
            
            # Better approach: use the fact that we need a 4-regular graph of girth 10
            # The known construction uses a specific voltage assignment.
            
            # Let's try: assign random voltages, but use a greedy approach to satisfy constraints
            # Start with empty assignment, assign one edge at a time, checking constraints
            
            # Actually, let's try a different base graph or a known construction.
            
            # Known: The (4,10)-cage has 320 vertices. One construction uses
            # the complete graph K5 with a specific voltage assignment.
            
            # Let's try: base graph is K5 (5 vertices, 4-regular), voltage group Z2^5
            # K5 has 10 edges. For each edge (i,j), assign voltage w_{ij} in Z2^5.
            # The lift has 5*32 = 160 vertices. But we need 320 vertices.
            # So this gives a 160-vertex graph, not 320.
            
            # Wait, K5 is 4-regular with 5 vertices. The lift with Z2^5 has 160 vertices.
            # But we need 320 vertices. So this doesn't match.
            
            # Let me reconsider. The (4,10)-cage has 320 vertices.
            # One known construction uses a bipartite base graph with 10 vertices.
            
            # Let's try the random search more carefully.
            # Key insight: we need to avoid 4-cycles, 6-cycles, and 8-cycles.
            # 4-cycles are the most numerous and easiest to check.
            
            # Let's use a SAT-like approach: assign voltages to edges one by one,
            # backtracking when a 4-cycle constraint is violated.
            
            def solve_voltages():
                n_edges = len(base_edges)
                voltages = [None] * n_edges
                
                # Precompute which edges are in which 4-cycles
                edge_to_cycles = [[] for _ in range(n_edges)]
                for idx, (e1, e2, e3, e4) in enumerate(four_cycles):
                    for e in (e1, e2, e3, e4):
                        edge_to_cycles[e].append(idx)
                
                # Try to assign voltages using backtracking
                # Order edges by number of constraints (most constrained first)
                edge_order = sorted(range(n_edges), key=lambda e: -len(edge_to_cycles[e]))
                
                def backtrack(pos):
                    if pos == n_edges:
                        return True
                    
                    e = edge_order[pos]
                    # Try all 32 possible voltages
                    for v in range(32):
                        voltages[e] = v
                        
                        # Check all 4-cycles that include e and are now fully assigned
                        valid = True
                        for cycle_idx in edge_to_cycles[e]:
                            e1, e2, e3, e4 = four_cycles[cycle_idx]
                            # Check if all edges in this cycle are assigned
                            if (voltages[e1] is not None and voltages[e2] is not None and
                                voltages[e3] is not None and voltages[e4] is not None):
                                s = voltages[e1] ^ voltages[e2] ^ voltages[e3] ^ voltages[e4]
                                if s == 0:
                                    valid = False
                                    break
                        
                        if valid and backtrack(pos + 1):
                            return True
                    
                    voltages[e] = None
                    return False
                
                if backtrack(0):
                    return voltages
                return None
            
            voltages = solve_voltages()
            if voltages is None:
                return None
            
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
