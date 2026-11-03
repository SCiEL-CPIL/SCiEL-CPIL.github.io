import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # The task is specifically for k=4, g=10.
    # The seed uses Lazebnik-Ustimenko-Woldar (LUW) construction.
    # For k=4, g=10, we use q=4, d=5.
    # The resulting connected component has 320 vertices.
    
    q = k
    # Determine d such that girth >= d + 5
    # For even girth g, we typically look at d = g - 5?
    # Seed logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=10: g-5=5 (odd), so d=5.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Field operations for GF(q)
    if q == 4:
        # GF(4) is isomorphic to Z_2 x Z_2 with specific multiplication
        # The seed uses a specific table:
        # table[x][y] = x*y in GF(4)
        # 0*any=0, 1*any=any, 2*2=3, 2*3=1, 3*3=2
        # Let's hardcode the multiplication table for GF(4) as used in the seed
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return mul_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Precompute the rules for the LUW construction
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    rule = {}
    # Base cases from seed:
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # The seed uses a specific pattern for higher c
        # This pattern ensures the girth condition
        options = [
            (1, c0 - 2),
            (c0 - 1, 1),
            (c0, 1),
            (1, c0 + 1)
        ]
        rule[c] = options[r]

    # BFS to find the connected component of the zero vertex
    # Vertices are (side, tuple_of_coords)
    # side 0: point, side 1: line
    # coords: tuple of length d+1? 
    # Seed uses: (None,) + (0,)*d. Length d+1.
    # Indices 0..d. Index 0 is ignored in logic (always None/0).
    
    start_vertex = (0, (0,) * (d + 1)) # Using 0s instead of None for hash speed, index 0 unused
    
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    # Optimize neighbor generation
    # For a point (0, p), neighbors are lines (1, l)
    # For a line (1, l), neighbors are points (0, p)
    
    # We iterate through all vertices in order
    for v in order:
        if time.time() > deadline:
            break
            
        side, coords = v
        v_idx = index[v]
        
        # Generate neighbors
        # The LUW graph is bipartite.
        # If side == 0 (point), we find lines l such that l ~ p
        # If side == 1 (line), we find points p such that p ~ l
        
        # The adjacency condition in LUW(d,q):
        # Point p ~ Line l iff for all c in 2..d:
        # l[c] - p[c] = l[a_c] * p[b_c]
        
        # To find neighbors of a point p:
        # We need to find l.
        # The seed's `neighbours` function iterates `first` in range(q).
        # It sets y[1] = first.
        # Then it recursively determines y[c] for c=2..d.
        
        # Let's replicate this logic efficiently.
        
        if side == 0:
            # Point p. Find lines l.
            # l[0] is irrelevant. l[1] can be anything in 0..q-1.
            # l[c] is determined by p and l[a], l[b].
            # The seed sets y[1] = first.
            # Then for c=2..d:
            # y[c] = p[c] + l[a] * p[b]  (since side==0, add)
            # Wait, seed: y[c] = add(x[c], mul(y[a], x[b]))
            # Here x is the current vertex (point p). y is the neighbor (line l).
            # So l[c] = p[c] + l[a] * p[b].
            
            for first in range(q):
                l_coords = [0] * (d + 1)
                l_coords[1] = first
                valid = True
                for c in range(2, d + 1):
                    a, b = rule[c]
                    # l[c] = p[c] + l[a] * p[b]
                    val = add(coords[c], mul(l_coords[a], coords[b]))
                    l_coords[c] = val
                # l[0] remains 0 (or None)
                neighbor = (1, tuple(l_coords))
                if neighbor not in index:
                    index[neighbor] = len(order)
                    order.append(neighbor)
                # Add edge
                # Ensure we don't add duplicate edges.
                # Since it's bipartite and we iterate points, we add edge (v_idx, n_idx)
                # But we must ensure we don't add it twice if we also iterate lines.
                # The seed only adds edges if `vertex[0] == 0`.
                edges.append((v_idx, index[neighbor]))
                
        else:
            # Line l. Find points p.
            # p[c] = l[c] - l[a] * p[b] ?
            # Seed: y[c] = sub(x[c], mul(x[a], y[b]))
            # Here x is line l, y is point p.
            # So p[c] = l[c] - l[a] * p[b].
            # This is a recursive definition for p[c] depending on p[b].
            # The seed iterates `first` in range(q) for y[1] (which is p[1]).
            
            for first in range(q):
                p_coords = [0] * (d + 1)
                p_coords[1] = first
                valid = True
                for c in range(2, d + 1):
                    a, b = rule[c]
                    # p[c] = l[c] - l[a] * p[b]
                    # Note: p[b] must already be computed.
                    # In the rule, b is usually < c.
                    val = sub(coords[c], mul(coords[a], p_coords[b]))
                    p_coords[c] = val
                neighbor = (0, tuple(p_coords))
                if neighbor not in index:
                    index[neighbor] = len(order)
                    order.append(neighbor)
                # Do not add edge here, as we added it when processing the point.
                # The seed only adds edges when vertex[0] == 0.
                pass

    # If we hit the deadline, we might have an incomplete graph.
    # But for n=320, it should finish instantly.
    # To be safe, if the graph is not 4-regular or connected, we might need to handle it.
    # But the LUW construction guarantees regularity and connectivity of the component.
    
    n = len(order)
    return n, edges

# EVOLVE-BLOCK-END
