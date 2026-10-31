"""Construct a 3-regular graph of girth 16 via voltage graph / circulant lift."""
# EVOLVE-BLOCK-START
import random
from collections import deque

def construct(k, g):
    if k != 3:
        # Fallback for other k
        return _luw_construct(k, g)
    
    # Voltage graph construction for k=3
    # Base graph: 2 vertices u, v with 3 parallel edges e1, e2, e3.
    # Lift by Z_m.
    # Vertices in lift: (u, i), (v, i) for i in Z_m.
    # Edges: (u, i) ~ (v, i + v_j) for j in {1, 2, 3}.
    # Girth is determined by shortest non-backtracking closed walk in base with voltage 0 mod m.
    # Non-backtracking walks in base (2 vertices, 3 edges) have even length 2, 4, 6, ...
    # Length 2: e_i, e_j (i != j). Voltage v_i - v_j.
    # We need v_i != v_j mod m for all i != j.
    # Length 4: e_i, e_j, e_k, e_l with i!=j, j!=k, k!=l.
    # Voltage v_i - v_j + v_k - v_l.
    # We need this != 0 mod m.
    # And so on up to length g-2 (since girth >= g, no cycles < g).
    # Note: Cycles in lift correspond to closed walks in base.
    # A cycle of length L in lift corresponds to a non-backtracking closed walk of length L in base.
    # So we need no non-backtracking closed walk of length L < g with voltage 0.
    
    # Search for smallest m such that there exist v1, v2, v3 in Z_m
    # satisfying the conditions.
    
    # Lower bound for m:
    # Moore bound n >= 2 * (3 * 2^(g/2 - 1) - 1) / 1 ?
    # Standard Moore bound for k=3, girth g (even):
    # n >= 2 * (2^(g/2) - 1) ? No.
    # n >= 2 * sum_{i=0}^{g/2 - 1} 2^i = 2 * (2^(g/2) - 1) = 2^(g/2 + 1) - 2.
    # For g=16, g/2=8. n >= 2^9 - 2 = 510.
    # Since n = 2m, 2m >= 510 => m >= 255.
    
    min_m = 2 ** (g // 2)
    # Add a small buffer to ensure we find a solution quickly
    max_m = min_m * 2 + 100
    
    # Precompute non-backtracking walks in the base graph up to length g-2
    # Base graph: nodes 0 (u), 1 (v). Edges 0, 1, 2 connecting them.
    # Walk is a sequence of edges e_1, ..., e_L.
    # e_1: 0->1 or 1->0. Let's fix start at 0.
    # e_1: 0->1 (edge j1).
    # e_2: 1->0 (edge j2), j2 != j1.
    # e_3: 0->1 (edge j3), j3 != j2.
    # ...
    # e_L: 1->0 (edge jL), jL != j(L-1).
    # Closed walk: ends at 0.
    # Voltage: sum of v_j * (1 if 0->1, -1 if 1->0).
    # Since we start at 0 and end at 0, and L is even:
    # Steps 1, 3, ..., L-1 are 0->1 (coeff +1).
    # Steps 2, 4, ..., L are 1->0 (coeff -1).
    # Voltage = v_{j1} - v_{j2} + v_{j3} - ... + v_{j(L-1)} - v_{jL}.
    
    # We will generate all valid non-backtracking edge sequences of length L (even, L < g)
    # and check if any combination of v1, v2, v3 sums to 0.
    
    # To make it efficient, we can just iterate m and random v's and check.
    # Or, we can derive constraints.
    # Given the small number of edges (3), the number of walks is small.
    # Number of non-backtracking walks of length L in base:
    # L=2: 3*2 = 6 walks.
    # L=4: 6*2 = 12 walks?
    # From a walk ending at 0 with last edge j, next edge can be any k != j.
    # So 2 choices.
    # Total walks of length L: 3 * 2^(L-1).
    # For L=14, 3 * 2^13 = 24576 walks.
    # This is small. We can precompute the coefficient vectors for each walk.
    # Each walk corresponds to a vector c in {-1, 0, 1}^3.
    # We need c . v != 0 for all walks.
    
    # Let's collect all constraint vectors c for L = 2, 4, ..., g-2.
    constraints = []
    for L in range(2, g, 2):
        # Generate all non-backtracking walks of length L starting at 0
        # State: (current_node, last_edge)
        # Start: (0, -1)
        # We want to end at 0.
        
        # Recursive generation or iterative
        # Since L is small, we can use a stack or recursion.
        # But we can just enumerate all sequences j1...jL with j_i != j_{i-1}.
        # And check if it's a valid walk in the base graph.
        # In base graph (2 nodes, 3 parallel edges), any sequence with j_i != j_{i-1} is a valid non-backtracking walk.
        # It starts at 0.
        # Step 1 (0->1): edge j1.
        # Step 2 (1->0): edge j2.
        # ...
        # Step L (1->0): edge jL.
        # Ends at 0.
        # So any sequence j1...jL with j_i != j_{i-1} is valid.
        
        # We need to avoid duplicates?
        # The voltage sum depends only on the counts of each edge in odd/even positions.
        # Actually, we just need to check if c . v == 0.
        # We can store the unique constraint vectors c.
        
        seen_vectors = set()
        # Iterate over all j1
        for j1 in range(3):
            # Iterate over j2...jL
            # This is 3 * 2^(L-1) iterations.
            # For L=14, 24576.
            # Total for all L up to 14: sum_{k=1}^{7} 3*2^(2k-1) = 3 * (2^13 - 1) / (2^2 - 1) * 2?
            # Sum is approx 3 * 2^13 = 24k.
            # This is fast.
            
            # Use a list to build the sequence
            seq = [j1]
            def gen(idx, last):
                if idx == L:
                    # Calculate vector c
                    c = [0, 0, 0]
                    for i in range(L):
                        edge = seq[i]
                        if i % 2 == 0: # 0->1, coeff +1
                            c[edge] += 1
                        else: # 1->0, coeff -1
                            c[edge] -= 1
                    # Check if c is zero vector (should not be for L>=2 with non-backtracking?
                    # If c is zero, it's a trivial constraint 0=0, which is always true.
                    # But we want c.v != 0. If c is zero, this constraint is useless.
                    # However, non-backtracking walks with L>=2 cannot have c=0?
                    # Example L=2: j1=0, j2=0 is invalid. j1=0, j2=1 => c = [1, -1, 0].
                    # So c is never zero.
                    
                    # Add to constraints if not seen
                    key = tuple(c)
                    if key not in seen_vectors:
                        seen_vectors.add(key)
                        constraints.append(c)
                    return
                
                for nxt in range(3):
                    if nxt != last:
                        seq.append(nxt)
                        gen(idx + 1, nxt)
                        seq.pop()
            
            gen(1, j1)
    
    # Now we have a list of constraint vectors c.
    # We need to find v = (v1, v2, v3) in Z_m^3 such that c . v != 0 mod m for all c in constraints.
    # Also v1, v2, v3 should be distinct?
    # If v1 == v2, then constraint [1, -1, 0] gives 0.
    # So distinctness is implied by the L=2 constraints.
    
    # Search for m and v
    # Start m from min_m
    # For each m, try random v's
    # Or, we can fix v1=0, v2=1, and search for v3?
    # No, affine transformation.
    # Let's just random search.
    
    random.seed(42)
    
    for m in range(min_m, max_m + 1):
        # Try several random assignments
        for _ in range(1000):
            v = [random.randint(0, m - 1) for _ in range(3)]
            valid = True
            for c in constraints:
                s = (c[0] * v[0] + c[1] * v[1] + c[2] * v[2]) % m
                if s == 0:
                    valid = False
                    break
            if valid:
                # Construct the graph
                n = 2 * m
                edges = []
                # Vertices: 0..m-1 are u, m..2m-1 are v?
                # Let's map (u, i) to i, (v, i) to m + i.
                for i in range(m):
                    for j in range(3):
                        u_idx = i
                        v_idx = m + (i + v[j]) % m
                        edges.append((u_idx, v_idx))
                
                # Verify girth just in case (optional, but good for debugging)
                # For speed, we trust the construction.
                # But we should ensure it's simple.
                # Since v1, v2, v3 are distinct mod m (implied by L=2 constraints),
                # the neighbors of (u, i) are (v, i+v1), (v, i+v2), (v, i+v3), which are distinct.
                # So it is simple.
                
                return n, edges
        
        # If no valid v found for this m, try next m
        # We can also try to be smarter, but random should work for small m.
    
    # If we fail, fall back to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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
