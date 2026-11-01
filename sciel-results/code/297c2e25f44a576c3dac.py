import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    # For k=5, g=9, we use a lift of the Petersen graph.
    # Petersen graph: 10 vertices, 5-regular, girth 5.
    # Vertices: 0..9
    # Edges:
    # Outer 5-cycle: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Inner 5-star: (5,7), (7,9), (9,6), (6,8), (8,5)
    # Spokes: (0,5), (1,6), (2,7), (3,8), (4,9)
    
    if k != 5 or g != 9:
        # Fallback to seed logic if parameters differ, though task implies fixed k,g
        # For now, just return a large valid graph if possible or seed
        return _fallback_construct(k, g, start_time, deadline)

    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Find all simple cycles of length < g in Petersen graph
    # g=9, so we need to avoid cycles of length 3, 4, 5, 6, 7, 8.
    # Petersen has no 3, 4 cycles.
    # We need to list 5, 6, 7, 8 cycles.
    
    def find_cycles(edges, max_len):
        adj = {i: [] for i in range(10)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        cycles = set()
        
        def dfs(start, current, path, visited):
            if len(path) > 1 and current == start and len(path) >= 3 and len(path) <= max_len:
                # Canonical form for cycle
                min_idx = path.index(min(path))
                canon = tuple(path[min_idx:] + path[:min_idx])
                cycles.add(canon)
            if len(path) >= max_len:
                return
            for nb in adj[current]:
                if nb == start and len(path) >= 3:
                    # Check if this closes a cycle
                    pass
                elif nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(start, nb, path, visited)
                    path.pop()
                    visited.remove(nb)
        
        for i in range(10):
            dfs(i, i, [i], {i})
            
        return list(cycles)

    short_cycles = find_cycles(petersen_edges, g - 1)
    # Convert to list of edge indices for constraint checking
    # Map edge to index
    edge_to_idx = {}
    for i, (u, v) in enumerate(petersen_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
        
    cycle_edge_indices = []
    for cyc in short_cycles:
        edges_in_cyc = []
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i+1) % len(cyc)]
            edges_in_cyc.append(edge_to_idx[(u, v)])
        cycle_edge_indices.append(edges_in_cyc)

    # Backtracking search for voltages
    # m is the size of the group Z_m
    # We want smallest n = 10 * m
    # Moore bound 428 => m >= 43
    
    found_voltages = None
    found_m = None
    
    for m in range(43, 100): # Try up to m=100 (n=1000)
        if time.time() > deadline - 10.0:
            break
            
        # Backtracking
        voltages = [0] * 15
        valid = True
        
        def backtrack(idx):
            if time.time() > deadline - 10.0:
                return False
            if idx == 15:
                return True
            for val in range(m):
                voltages[idx] = val
                # Check constraints for cycles that include edge idx and are fully assigned
                # A cycle is "fully assigned" if all its edges are < idx? No, indices are arbitrary.
                # We need to check any cycle where all edges are assigned.
                # Since we assign in order 0..14, a cycle is fully assigned when max(edge_idx) == idx?
                # Not necessarily, if we skip some? No, we assign sequentially.
                # So if max(edge_idx in cycle) == idx, we can check it.
                
                can_proceed = True
                for cyc_edges in cycle_edge_indices:
                    if max(cyc_edges) == idx:
                        # Check if sum is 0 mod m
                        s = sum(voltages[e] for e in cyc_edges)
                        if s % m == 0:
                            can_proceed = False
                            break
                            
                if can_proceed:
                    if backtrack(idx + 1):
                        return True
                        
            return False

        if backtrack(0):
            found_voltages = voltages[:]
            found_m = m
            break

    if found_voltages is None:
        # If no solution found in range, fallback
        return _fallback_construct(k, g, start_time, deadline)

    m = found_m
    n = 10 * m
    edges = []
    
    for i, (u, v) in enumerate(petersen_edges):
        a = found_voltages[i]
        for j in range(m):
            # Edge (u, j) to (v, (j+a) % m)
            u_node = u * m + j
            v_node = v * m + (j + a) % m
            edges.append((u_node, v_node))
            
            # Edge (v, j) to (u, (j-a) % m)
            # This is the reverse direction, but in undirected graph, 
            # adding (u_node, v_node) covers the connection.
            # Wait, the lift of an undirected edge (u,v) with voltage a 
            # consists of edges (u,j)-(v, j+a) for all j.
            # Does it also include (v,j)-(u, j-a)?
            # Yes, but (v,j)-(u, j-a) is the same set of edges as (u,k)-(v, k+a) 
            # where k = j-a.
            # So we only need to add the forward direction once.
            
    return n, edges

def _fallback_construct(k, g, start_time, deadline):
    # Use the seed logic if the specific Petersen lift fails or params differ
    # This ensures we always return *something* valid-ish if the main path fails
    # But seed logic is known to be large.
    # We can just return the seed logic result.
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    if q == 4:
        mul_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    else:
        mul_table = np.outer(np.arange(q, dtype=np.int64), np.arange(q, dtype=np.int64)) % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    rule_a = np.array([rule[c][0] for c in range(2, d + 1)], dtype=np.int64)
    rule_b = np.array([rule[c][1] for c in range(2, d + 1)], dtype=np.int64)

    free_vals = np.arange(q, dtype=np.int64)

    def get_neighbours(side, x_arr):
        y_coords = np.empty((q, d + 1), dtype=np.int64)
        y_coords[:, 0] = 0
        y_coords[:, 1] = free_vals
        if side == 0:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                y_coords[:, c] = (x_arr[c] + mul_table[y_coords[:, a], x_arr[b]]) % q
        else:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                y_coords[:, c] = (x_arr[c] - mul_table[x_arr[a], y_coords[:, b]]) % q
        return y_coords

    start_key = (0, (0,) * (d + 1))
    index = {start_key: 0}
    order = [start_key]
    edges = []

    for i, vertex in enumerate(order):
        if time.time() > deadline - 5.0:
            break
        side, x_tuple = vertex
        x_arr = np.array(x_tuple, dtype=np.int64)
        y_coords = get_neighbours(side, x_arr)
        for j in range(q):
            other_key = (1 - side, tuple(y_coords[j]))
            if other_key not in index:
                index[other_key] = len(order)
                order.append(other_key)
            if side == 0:
                edges.append((index[vertex], index[other_key]))

    return len(order), edges
# EVOLVE-BLOCK-END
