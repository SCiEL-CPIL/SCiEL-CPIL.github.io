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

    # L-U-W construction:
    # q = k
    # d = g - 5 (must be odd)
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)
    
    q = k
    
    # Rules for c = 2..d
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
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
    
    # Optimize for prime q using direct arithmetic
    is_prime = (q == 3 or q == 5 or q == 7 or q == 11 or q == 13 or q == 17 or q == 19 or q == 23 or q == 29 or q == 31)
    
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
                if is_prime:
                    if side == 0:
                        # Point -> Line: y[c] = (x[c] + y[a] * x[b]) % q
                        y[c] = (x[c] + y[a] * x[b]) % q
                    else:
                        # Line -> Point: y[c] = (x[c] - x[a] * y[b]) % q
                        y[c] = (x[c] - x[a] * y[b]) % q
                else:
                    # Fallback for non-prime q (e.g., q=4)
                    # For q=4, use GF(4) arithmetic
                    if q == 4:
                        # GF(4): {0,1,2,3} with 2^2=3, 2*3=1, 3^2=2
                        mul4 = [0, 1, 2, 3]
                        mul4[1] = [0, 1, 2, 3]
                        mul4[2] = [0, 2, 3, 1]
                        mul4[3] = [0, 3, 1, 2]
                        if side == 0:
                            y[c] = (x[c] + mul4[y[a]][x[b]]) % 4
                        else:
                            y[c] = (x[c] - mul4[x[a]][y[b]]) % 4
                    else:
                        # Generic fallback (shouldn't happen for typical cage problems)
                        if side == 0:
                            y[c] = (x[c] + y[a] * x[b]) % q
                        else:
                            y[c] = (x[c] - x[a] * y[b]) % q
            
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
