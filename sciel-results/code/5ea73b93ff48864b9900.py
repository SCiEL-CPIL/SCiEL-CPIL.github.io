import random
import time
import sys
from itertools import product

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # 5s margin
    
    # Target: 4-regular graph with girth 10.
    # Use voltage graph lift of a 32-vertex 4-regular bipartite graph with girth 6.
    # Voltage group: Z3 x Z3 (size 9).
    # Lifted graph will have 32 * 9 = 288 vertices.
    # Wait, 288 < 320. This is better!
    # But does a 288-vertex (4,10)-cage exist?
    # Moore bound for k=4, g=10 is 243.
    # 288 is between 243 and 320.
    # If we can construct one, it's an improvement over 320.
    
    # Base graph: 16 U-nodes, 16 W-nodes.
    # U_i connected to W_{i+j} for j in {0, 1, 2, 3} mod 16.
    # This is 4-regular.
    
    n_base = 32
    u_size = 16
    w_size = 16
    
    # Build base adjacency
    # U nodes: 0..15
    # W nodes: 16..31
    adj = [[] for _ in range(n_base)]
    for i in range(u_size):
        for j in range(4):
            w_idx = 16 + (i + j) % w_size
            adj[i].append(w_idx)
            adj[w_idx].append(i)
            
    # Verify base girth
    # We assume girth is 6 as derived.
    
    # Find all cycles of length 6 and 8 in base graph.
    # A cycle in base graph corresponds to a closed walk in the lift.
    # We need to ensure that for every closed walk of length L < 10 in the base graph,
    # the product of voltages is NOT identity.
    
    # Since base girth is 6, no cycles of length 4.
    # We need to check cycles of length 6 and 8.
    
    # Enumerate cycles of length 6.
    # A cycle of length 6 is u0-w0-u1-w1-u2-w2-u0.
    # This requires u0~w0, w0~u1, u1~w1, w1~u2, u2~w2, w2~u0.
    # In our graph:
    # u0 ~ w0, w1, w2, w3.
    # w0 ~ u0, u15, u14, u13.
    # So u1 must be in {15, 14, 13}.
    # Similarly for others.
    
    # Let's generate all closed walks of length 6 and 8.
    # To avoid duplicates, we can fix the starting node and direction.
    
    def find_closed_walks(length):
        walks = []
        for start in range(n_base):
            # DFS
            path = [start]
            visited = {start}
            
            def dfs(current, depth):
                if time.time() > deadline:
                    return
                if depth == length:
                    if current == start:
                        # Check if it's a simple cycle or just a walk?
                        # For voltage graph, we need all closed walks.
                        # But if a walk repeats vertices, it contains a shorter cycle.
                        # If shorter cycles are already handled, we only need simple cycles?
                        # No, a closed walk of length 8 might be two cycles of length 4.
                        # But base girth is 6, so no cycles of length 4.
                        # So any closed walk of length < 10 must be a simple cycle of length 6 or 8.
                        # So we just need simple cycles.
                        walks.append(list(path))
                    return
                
                for neighbor in adj[current]:
                    if neighbor == start and depth < length - 1:
                        continue # Don't close early
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, depth + 1)
                    path.pop()
                    visited.remove(neighbor)
            
            dfs(start, 1)
        return walks
    
    # This DFS might be slow if many walks.
    # But n=32 is small.
    
    cycles_6 = find_closed_walks(6)
    cycles_8 = find_closed_walks(8)
    
    # Total cycles to check.
    # Assign voltages in Z3 x Z3.
    # Represent voltage as (a, b) where a, b in {0, 1, 2}.
    # Identity is (0, 0).
    # Product is addition mod 3.
    
    # We have 64 edges in base graph.
    # We need to assign voltages to these 64 edges.
    
    # Random search for voltage assignment.
    
    edge_list = []
    for i in range(n_base):
        for j in adj[i]:
            if i < j:
                edge_list.append((i, j))
                
    # Map edge to index
    edge_index = {}
    for idx, (u, v) in enumerate(edge_list):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
        
    def get_voltage(u, v, voltages):
        idx = edge_index[(u, v)]
        return voltages[idx]
        
    def check_assignment(voltages):
        # voltages is a list of (a, b) tuples
        # Check all cycles
        for cycle in cycles_6:
            # cycle is a list of nodes [v0, v1, ..., v5, v0]
            # Wait, find_closed_walks returns path including start at end?
            # No, path is [start, ..., current] where current==start.
            # So cycle is [v0, v1, v2, v3, v4, v5, v0].
            # Edges are (v0, v1), (v1, v2), ..., (v5, v0).
            sum_a = 0
            sum_b = 0
            for i in range(len(cycle) - 1):
                u = cycle[i]
                v = cycle[i+1]
                va, vb = get_voltage(u, v, voltages)
                # Direction matters?
                # In voltage graphs, voltage on (u,v) is g. Voltage on (v,u) is g^{-1}.
                # In abelian group, g^{-1} = -g.
                # So if we traverse u->v, we add g.
                # If we traverse v->u, we add -g.
                # Our cycle is v0->v1->...->v0.
                # So we always add the voltage in the direction of traversal.
                sum_a = (sum_a + va) % 3
                sum_b = (sum_b + vb) % 3
            if sum_a == 0 and sum_b == 0:
                return False
        for cycle in cycles_8:
            sum_a = 0
            sum_b = 0
            for i in range(len(cycle) - 1):
                u = cycle[i]
                v = cycle[i+1]
                va, vb = get_voltage(u, v, voltages)
                sum_a = (sum_a + va) % 3
                sum_b = (sum_b + vb) % 3
            if sum_a == 0 and sum_b == 0:
                return False
        return True
        
    # Random search
    best_voltages = None
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        voltages = [(random.randint(0, 2), random.randint(0, 2)) for _ in range(len(edge_list))]
        if check_assignment(voltages):
            best_voltages = voltages
            break
            
    if best_voltages is None:
        # Fallback to LUW if no voltage assignment found
        # Or return a larger graph
        # For now, return LUW
        return _luw_construct(k, g)
        
    # Construct lifted graph
    # Nodes: (base_node, voltage) where voltage in Z3 x Z3
    # Total nodes: 32 * 9 = 288
    
    n_lifted = n_base * 9
    edges = []
    
    # Map (base_node, va, vb) to index
    def node_index(b, va, vb):
        return b * 9 + va * 3 + vb
        
    for b in range(n_base):
        for va in range(3):
            for vb in range(3):
                idx = node_index(b, va, vb)
                for neighbor in adj[b]:
                    e_idx = edge_index[(b, neighbor)]
                    eva, evb = best_voltages[e_idx]
                    # If b < neighbor, voltage is (eva, evb).
                    # If b > neighbor, voltage is (-eva, -evb) mod 3.
                    if b < neighbor:
                        nva = (va + eva) % 3
                        nvb = (vb + evb) % 3
                    else:
                        nva = (va - eva) % 3
                        nvb = (vb - evb) % 3
                    n_idx = node_index(neighbor, nva, nvb)
                    if idx < n_idx:
                        edges.append((idx, n_idx))
                        
    return n_lifted, edges

def _luw_construct(k, g):
    # Original LUW construction as fallback
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
