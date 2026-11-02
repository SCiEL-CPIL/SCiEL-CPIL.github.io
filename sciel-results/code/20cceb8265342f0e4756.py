"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 115.0  # Safety margin
    
    # Hardcoded known optimal or near-optimal cages for small parameters
    # Format: (k, g): (n, edges)
    # Note: Edges must be a list of (u, v) tuples with 0-indexed vertices.
    
    def petersen_graph():
        # 3-regular, girth 5, 10 vertices
        # Outer cycle 0-4, inner star 5-9
        edges = []
        for i in range(5):
            edges.append((i, (i+1) % 5))
            edges.append((i, 5+i))
            edges.append((5+i, 5+(i+2) % 5))
        return 10, edges

    def heawood_graph():
        # 3-regular, girth 6, 14 vertices
        # Incidence graph of Fano plane
        # 7 points, 7 lines
        edges = []
        # Points 0-6, Lines 7-13
        # Fano plane lines:
        lines = [
            (0, 1, 2), (0, 3, 4), (0, 5, 6),
            (1, 3, 5), (1, 4, 6), (2, 3, 6),
            (2, 4, 5)
        ]
        for i, line in enumerate(lines):
            for p in line:
                edges.append((p, 7+i))
        return 14, edges

    def cage_4_5():
        # 4-regular, girth 5
        # Smallest known is 19 vertices? 
        # Actually, the (4,5)-cage is the 19-vertex graph? 
        # Let's verify Moore bound: 1 + 4 + 4*3 = 17.
        # There is no (4,5) graph on 17 vertices.
        # The smallest is 19? 
        # I will use L-U-W for this to be safe, unless I can generate it.
        # L-U-W for k=4, g=5: d=3 (odd). q=4.
        # Vertices: 2 * 4^3 = 128? Too big.
        # Let's just use L-U-W for k=4, g=5 as well, or a known construction.
        # The complement of the Petersen graph is 4-regular on 10 vertices, but girth is 3.
        # I'll skip hardcoding this and rely on L-U-W or a better search if time permits.
        # For now, return None to indicate "use L-U-W"
        return None

    # Lookup table
    known_cages = {
        (3, 3): (3, [(0,1), (1,2), (0,2)]),
        (3, 4): (6, [(0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)]), # K3,3
        (3, 5): petersen_graph(),
        (3, 6): heawood_graph(),
        (4, 3): (4, [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]), # K4
        (4, 4): (8, [(i, j) for i in range(4) for j in range(4, 8)]), # K4,4
        (5, 3): (5, [(i, j) for i in range(5) for j in range(i+1, 5)]), # K5
        (5, 4): (10, [(i, j) for i in range(5) for j in range(5, 10)]), # K5,5
    }

    if (k, g) in known_cages:
        return known_cages[(k, g)]

    # If g is small and not in known cages, or k is large, use L-U-W
    # L-U-W construction:
    # q = k
    # d = g - 5 (must be odd)
    # If g is even, girth of L-U-W with d=g-5 is d+5 = g. 
    # Wait, L-U-W with d gives girth at least d+5.
    # If we want girth >= g, we need d+5 >= g => d >= g-5.
    # If g-5 is even, we can use d=g-5? 
    # The standard L-U-W paper usually specifies d is odd.
    # If d is even, the girth is at least d+6? Or does it still work?
    # Lazebnik et al. define the graph for any d. 
    # If d is odd, girth >= d+5.
    # If d is even, girth >= d+6? 
    # To be safe, if g-5 is even, use d = g-4 (which is odd).
    # Then girth >= (g-4)+5 = g+1 >= g.
    # This might use more vertices.
    # Let's check: if g=6, g-5=1 (odd). d=1. Girth >= 6.
    # If g=7, g-5=2 (even). d=3 (odd). Girth >= 8.
    # Is there a construction for even d?
    # Yes, but let's stick to odd d to be safe with the standard formula.
    
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)
    
    q = k
    
    # Precompute field tables
    if q == 4:
        # GF(4) using XOR for add/sub, table for mul
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add_table = [[i ^ j for j in range(4)] for i in range(4)]
        sub_table = add_table
    else:
        mul_table = [[(i * j) % q for j in range(q)] for i in range(q)]
        add_table = [[(i + j) % q for j in range(q)] for i in range(q)]
        sub_table = [[(i - j) % q for j in range(q)] for i in range(q)]

    # Rules for c = 2..d
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # Patterns:
        # c=5 (r=0): (1, 3) -> c0=5? No.
        # Let's re-derive the pattern from the code provided in the prompt.
        # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        # where c0 = c - (c-5)%4.
        a, b = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        rule_a[c] = a
        rule_b[c] = b

    # BFS with integer encoding
    # Vertex: (side, coeffs_int)
    # coeffs_int encodes x_1...x_d in base q.
    # x_1 is the most significant digit.
    # x_i = (coeffs_int // (q ** (d-i))) % q
    
    # Precompute powers of q for decoding
    powers = [1] * (d + 1)
    for i in range(1, d + 1):
        powers[i] = powers[i-1] * q

    start_coeff = 0
    start_vertex = (0, start_coeff)
    
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    queue = deque([0])
    
    while queue:
        if time.time() > deadline - 5:
            break
        
        vi = queue.popleft()
        side, coeffs_int = order[vi]
        
        # Decode coefficients to list for fast access in neighbor calculation
        # x[1..d]
        x = [0] * (d + 1)
        temp = coeffs_int
        for i in range(d, 0, -1):
            x[i] = temp % q
            temp //= q
            
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a = rule_a[c]
                b = rule_b[c]
                if side == 0:
                    # Point -> Line: y[c] = x[c] + y[a] * x[b]
                    y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                else:
                    # Line -> Point: y[c] = x[c] - x[a] * y[b]
                    y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
            
            # Encode y into integer
            y_int = 0
            for i in range(1, d + 1):
                y_int = y_int * q + y[i]
                
            other = (1 - side, y_int)
            
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            
            if side == 0:
                edges.append((vi, index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
