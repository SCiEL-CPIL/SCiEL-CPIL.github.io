"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    deadline = time.time() + 140.0
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def _luw_graph(k, g):
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
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    if k == 4 and g == 10:
        # Strategy: Voltage Graph Lift of K_4,4
        # Base graph: K_4,4 (8 vertices, 4-regular, girth 4)
        # Lift by Z_m. Total vertices 8*m.
        # We want n < 320, so 8*m < 320 => m < 40.
        # Moore bound is 243, so 8*m >= 243 => m >= 31.
        # Try m in [31, 39].
        
        def get_base_walks(L):
            """Generate all closed walks of length L in K_4,4 starting from 0."""
            # K_4,4: Part A {0,1,2,3}, Part B {4,5,6,7}
            # Edges: (a, b) for a in A, b in B.
            walks = []
            def dfs(path, curr, length):
                if length == L:
                    if curr == 0:
                        walks.append(tuple(path))
                    return
                nexts = [v for v in range(8) if (v < 4) != (curr < 4)]
                for v in nexts:
                    dfs(path + [v], v, length + 1)
            dfs([0], 0, 0)
            return walks

        walks_4 = get_base_walks(4)
        walks_6 = get_base_walks(6)
        walks_8 = get_base_walks(8)
        
        # Pre-calculate edge indices for K_4,4
        # We need a mapping from (u, v) to an index 0..15 to store voltages
        edge_indices = {}
        idx = 0
        for a in range(4):
            for b in range(4, 8):
                edge_indices[(a, b)] = idx
                edge_indices[(b, a)] = idx
                idx += 1
        
        def check_lift(m, voltages):
            """Check if the lift has girth >= 10."""
            # voltages is a list of 16 integers mod m
            
            # Check 4-cycles
            for w in walks_4:
                s = 0
                for i in range(len(w)-1):
                    u, v = w[i], w[i+1]
                    s += voltages[edge_indices[(u, v)]]
                if s % m == 0:
                    return False
            
            # Check 6-cycles
            for w in walks_6:
                s = 0
                for i in range(len(w)-1):
                    u, v = w[i], w[i+1]
                    s += voltages[edge_indices[(u, v)]]
                if s % m == 0:
                    # Check if it's a simple cycle in the lift
                    # Walk: v0, v1, ..., v6=v0
                    # Lift: (v0, z0), (v1, z1), ..., (v0, z6)
                    # z0 = 0
                    # z_{i+1} = z_i + volt[u_i, v_i]
                    # We need all (v_i, z_i) to be distinct for i=0..5
                    visited = set()
                    z = 0
                    for i in range(6):
                        u, v = w[i], w[i+1]
                        key = (u, z)
                        if key in visited:
                            return False
                        visited.add(key)
                        z = (z + voltages[edge_indices[(u, v)]]) % m
                    # z should be 0 mod m, which we checked
                    # If we reach here, it was a simple cycle
                    return False
            
            # Check 8-cycles
            for w in walks_8:
                s = 0
                for i in range(len(w)-1):
                    u, v = w[i], w[i+1]
                    s += voltages[edge_indices[(u, v)]]
                if s % m == 0:
                    visited = set()
                    z = 0
                    for i in range(8):
                        u, v = w[i], w[i+1]
                        key = (u, z)
                        if key in visited:
                            return False
                        visited.add(key)
                        z = (z + voltages[edge_indices[(u, v)]]) % m
                    return False
            
            return True

        # Search for m
        for m in range(31, 40):
            if time.time() > deadline - 10:
                break
            n = 8 * m
            # Try random voltage assignments
            for attempt in range(2000):
                if time.time() > deadline - 5:
                    break
                # Random voltages
                # To avoid trivial 4-cycles, we can bias voltages?
                # Just random for now
                voltages = [random.randint(0, m-1) for _ in range(16)]
                
                if check_lift(m, voltages):
                    # Construct edges
                    edges = []
                    # Vertices are (v, z) -> v*m + z
                    # For each base edge (u, v) with voltage w:
                    # Edge between (u, z) and (v, z+w)
                    # We add each edge once.
                    # Since graph is undirected, we can just iterate all u, v, z.
                    # To avoid duplicates, we can enforce u < v in base?
                    # No, because z shifts.
                    # Better: iterate all vertices, add edges to neighbors with higher ID.
                    for u in range(8):
                        for v in range(8):
                            if (u, v) not in edge_indices:
                                continue
                            if u > v:
                                continue # Only one direction in base?
                                # Wait, K_4,4 is bipartite.
                                # We can just iterate all 16 base edges.
                                # For each base edge e=(u,v) with voltage w:
                                # For z in 0..m-1:
                                #   u_lift = u*m + z
                                #   v_lift = v*m + (z+w)%m
                                #   Add edge (min, max)
                                # This covers all edges exactly once?
                                # Yes, because each base edge is unique.
                            w = voltages[edge_indices[(u, v)]]
                            for z in range(m):
                                u_lift = u * m + z
                                v_lift = v * m + (z + w) % m
                                if u_lift < v_lift:
                                    edges.append((u_lift, v_lift))
                                else:
                                    edges.append((v_lift, u_lift))
                    return n, edges
            # If no success for this m, try next m
        # Fallback to LUW
        return _luw_graph(k, g)

    # Fallback to LUW
    return _luw_graph(k, g)
# EVOLVE-BLOCK-END
