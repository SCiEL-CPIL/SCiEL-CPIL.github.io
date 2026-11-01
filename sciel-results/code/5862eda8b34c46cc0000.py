"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math

    # We construct a 3-regular graph using a voltage lift of K_{3,3} over Z_m.
    # The base graph is K_{3,3} with vertices A={0,1,2} and B={0,1,2}.
    # Edges in base: (A_i, B_j) for all i, j in 0..2.
    # Each edge gets a voltage in Z_m. The lifted graph has 6*m vertices.
    # Vertex (side, base, shift) where side in {0,1}, base in 0..2, shift in 0..m-1.
    # Edge from (0, i, s) to (1, j, (s + v[i][j]) % m) where v[i][j] is the voltage.
    # Reverse edge voltage is -v[i][j].
    
    # Girth check:
    # A cycle of length L in the lift corresponds to a non-backtracking closed walk
    # of length L in the base graph with total voltage 0 mod m.
    # We precompute all non-backtracking closed walks in K_{3,3} of length < g.
    # For each walk, we have a linear form of the voltages.
    # We need to find voltages such that no linear form is 0 mod m.
    
    # 1. Generate non-backtracking closed walks in K_{3,3}
    # Vertices: A0, A1, A2, B0, B1, B2.
    # We start at A0.
    # Walks must be even length.
    # Length 2: A0->Bj->A0. This is a backtrack. In the lift, this is just traversing the edge and back.
    # It does not form a cycle of length 2 in a simple graph.
    # So we start checking from length 4.
    
    walks = []
    
    # DFS to find non-backtracking closed walks
    # State: (current_vertex, prev_vertex, length, voltage_coeffs)
    # voltage_coeffs: dict mapping (i, j) to coefficient.
    # Actually, we can just store the sequence of edges.
    
    # Let's represent a walk by the sequence of B-vertices visited.
    # A0 -> B_j1 -> A_k1 -> B_j2 -> A_k2 -> ... -> B_jt -> A0
    # The A-vertices are determined by the B-vertices and the non-backtracking constraint.
    # A0 -> B_j1. Next A must be k1 != 0.
    # B_j1 -> A_k1. Next B must be j2 != j1.
    # A_k1 -> B_j2. Next A must be k2 != k1.
    # ...
    # B_jt -> A0. This requires A0 != k_{t-1}.
    
    # Let's enumerate by the sequence of B-vertices (j1, j2, ..., jt).
    # Constraints:
    # 1. j1, ..., jt in {0, 1, 2}.
    # 2. j_{p+1} != j_p for all p.
    # 3. The intermediate A-vertices k_p must be distinct from the previous A.
    #    A0 -> B_j1 -> A_k1. k1 != 0.
    #    B_j1 -> A_k1 -> B_j2. j2 != j1.
    #    A_k1 -> B_j2 -> A_k2. k2 != k1.
    #    ...
    #    B_jt -> A0. This step is valid if the previous A was k_{t-1} and k_{t-1} != 0.
    
    # Wait, the sequence of A-vertices is determined by the sequence of B-vertices?
    # No. From B_j, we can go to any A_k.
    # So we need to enumerate the full sequence of vertices.
    
    # Let's do a DFS on the state (current, prev, length, coeffs).
    # coeffs is a tuple of 9 integers (coeffs for v00, v01, v02, v10, v11, v12, v20, v21, v22).
    
    def idx(i, j): return i * 3 + j
    
    def dfs(current, prev, length, coeffs):
        if length >= g:
            return
        if length > 0 and current == 0: # A0
            if length >= 4: # Non-backtracking cycles start at length 4
                # Check if it's a valid non-backtracking walk
                # We enforce non-backtracking in the transition
                walks.append((length, coeffs))
            return
        
        # Determine next vertices
        # current is 0..2 (A) or 3..5 (B)
        # prev is the previous vertex
        
        if current < 3: # A
            # Next is B_j
            for j in range(3):
                next_v = 3 + j
                if next_v == prev:
                    continue
                # Update coeffs
                # Edge A_current -> B_j
                # Voltage v[current][j]
                new_coeffs = list(coeffs)
                new_coeffs[idx(current, j)] += 1
                dfs(next_v, current, length + 1, tuple(new_coeffs))
        else: # B
            # Next is A_k
            k = current - 3
            for a in range(3):
                next_v = a
                if next_v == prev:
                    continue
                # Update coeffs
                # Edge B_k -> A_a
                # Voltage -v[a][k]
                new_coeffs = list(coeffs)
                new_coeffs[idx(a, k)] -= 1
                dfs(next_v, k, length + 1, tuple(new_coeffs))
                
    # Start at A0 (0), no prev
    dfs(0, -1, 0, (0,)*9)
    
    # 2. Search for m and voltages
    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3
    
    m_min = max(1, math.ceil(moore_bound / 6))
    
    random.seed(42)
    
    # Try m from m_min upwards
    for m in range(m_min, m_min + 500):
        if 6 * m > 5000: # Cap
            break
        
        # Try random voltage assignments
        for trial in range(2000):
            voltages = [random.randint(0, m - 1) for _ in range(9)]
            
            valid = True
            for length, coeffs in walks:
                # Compute sum
                s = 0
                for c in range(9):
                    s += coeffs[c] * voltages[c]
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                # Build graph
                n = 6 * m
                edges = []
                for i in range(3):
                    for j in range(3):
                        v = voltages[idx(i, j)]
                        for s in range(m):
                            u = i * m + s
                            w = 3 * m + j * m + (s + v) % m
                            edges.append((u, w))
                return n, edges
    
    # Fallback: Return a trivial graph if no small one found (should not happen)
    # But we must return something valid.
    # Let's just return a large random graph or the previous fallback.
    # For now, return a simple cycle (not 3-regular) or something?
    # The task requires 3-regular.
    # We can use the Lazebnik-Ustimenko-Woldar fallback from the original code if needed.
    
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
# EVOLVE-BLOCK-END
