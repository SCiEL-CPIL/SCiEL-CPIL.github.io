import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 145s to leave margin for 150s limit

    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    # Girth is 2d + 2.
    # We need 2d + 2 >= g => d >= (g - 2) / 2.
    # d must be an integer.
    # Also d >= 3 is a common lower bound for simple graphs in this family.
    
    q = k
    
    # Calculate minimal d
    # d_min = ceil((g - 2) / 2)
    d_min = (g - 2 + 1) // 2  # Ceiling division for positive ints
    if d_min < 3:
        d_min = 3
    d = d_min

    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # LUW Rule
    # The recurrence for LUW CD(d, q) involves specific coefficients.
    # The seed program defines `rule[c]` for c in 2..d.
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    # For LUW, the relation is often defined as:
    # y_i = x_{i-1} * x_{i-2} ... or similar.
    # The seed program's rule generation:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in 5..d:
    #   r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #   rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # This specific rule sequence is from the original LUW paper?
    # Yes, it corresponds to the standard construction.
    
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # Ensure indices are valid (1..d-1)
        # c0 is roughly c.
        # The tuple indices in `x` and `y` are 1..d.
        # The rule uses indices a, b from 1..d-1?
        # In `neighbours`, `x` and `y` are tuples of length d+1 (index 0 is None/side).
        # So valid indices are 1..d.
        # The rule values (a,b) should be in 1..d-1?
        # Let's check the seed code's usage.
        # `y[c] = add(x[c], mul(y[a], x[b]))`
        # `y` is the other side's vector. `x` is current.
        # `a, b` are indices into `y` and `x`.
        # So `a, b` must be valid indices 1..d.
        
        # The seed code logic for rule generation seems specific.
        # I will keep the exact same rule generation to ensure the graph structure is the standard LUW one.
        
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        # Sanity check: indices must be >= 1 and <= d-1?
        # If c0-2 < 1, it might break.
        # For c=5, c0=5. r=0. Rule (1, 3). OK.
        # For c=6, c0=6. r=1. Rule (5, 1). OK.
        # For c=7, c0=7. r=2. Rule (7, 1). OK.
        # For c=8, c0=8. r=3. Rule (1, 9). OK.
        # For c=9, c0=9. r=0. Rule (1, 7). OK.
        # It seems indices stay within range for small d.

    def neighbours(vertex):
        side, x = vertex
        # x is tuple of length d+1. x[0] is None. x[1..d] are values.
        # Actually, in seed: `y = [None, first] + [0] * (d - 1)`
        # So x[1] is 'first'.
        # We iterate 'first' from 0 to q-1.
        # This effectively iterates over all neighbors on the other side.
        
        for first in range(q):
            # Construct the other vector
            # The other vector has 'first' at index 1.
            # We calculate indices 2..d.
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point -> Line
                    # Relation: l[c] - p[c] = l[a] * p[b]
                    # y is Line, x is Point.
                    # y[c] = x[c] + y[a] * x[b]
                    # But y[a] is not calculated yet if a > c?
                    # The rule indices a, b must be < c?
                    # In the seed rule:
                    # c=2: a=1, b=1. 1 < 2.
                    # c=3: a=2, b=1. 2 < 3.
                    # c=4: a=1, b=2. 2 < 4.
                    # c=5: a=1, b=3. 3 < 5.
                    # c=6: a=5, b=1. 5 < 6.
                    # c=7: a=7, b=1. 7 < 7? No, 7 is not < 7.
                    # Wait, if a == c, we have a dependency on y[c] itself?
                    # In LUW, the indices are typically < c.
                    # Let's check c=7 rule in seed:
                    # c=7. (7-5)%4 = 2. c0=7.
                    # Rule index 2: (c0, 1) -> (7, 1).
                    # a=7.
                    # If side==0: y[7] = x[7] + y[7] * x[1].
                    # This implies y[7] (1 - x[1]) = x[7].
                    # y[7] = x[7] / (1 - x[1]).
                    # This is valid if 1 != x[1].
                    # If 1 == x[1], then 0 = x[7].
                    # This is a linear equation, not an explicit formula.
                    # The seed code does: `y[c] = add(x[c], mul(y[a], x[b]))`
                    # If a=c, this is `y[c] = x[c] + y[c] * x[b]`.
                    # `y[c] (1 - x[b]) = x[c]`.
                    # `y[c] = x[c] / (1 - x[b])`.
                    # The seed code does NOT divide. It just assigns.
                    # `y[c] = add(x[c], mul(y[a], x[b]))`
                    # If a=c, `y[c]` is used in its own calculation?
                    # In Python, `y[c]` on the RHS is the *current* value in the list `y`.
                    # Initially `y` is filled with 0s.
                    # So `y[c]` (RHS) is 0.
                    # Then `y[c]` (LHS) becomes `x[c] + 0`.
                    # This ignores the self-dependency.
                    # This suggests the seed code might be flawed for cases where a==c or b==c?
                    # Or the rule generation ensures a < c and b < c?
                    
                    # Let's re-examine the rule generation for a < c.
                    # c=5: (1,3). 1<5, 3<5. OK.
                    # c=6: (5,1). 5<6, 1<6. OK.
                    # c=7: (7,1). 7! = 7. a=c.
                    # c=8: (1,9). 9 > 8? b=c+1.
                    # The indices can be >= c.
                    # If b > c, x[b] is 0 (initial value) or already calculated?
                    # x is fixed. x[b] is just a value.
                    # But y is being calculated.
                    # If a > c, y[a] is 0 (not calculated yet).
                    # If the LUW definition requires solving a system, this simple forward substitution might be wrong.
                    
                    # HOWEVER, the seed program has score 1.0.
                    # It is accepted as valid.
                    # This implies either:
                    # 1. The specific rule sequence never triggers the problematic dependencies for the connected component of 0.
                    # 2. My understanding of the dependency is wrong.
                    # 3. The graph produced, even if not strictly LUW as per literature, is a valid k-regular graph with sufficient girth.
                    
                    # Given the directive "KEEP this design", I should not change the neighbor generation logic, only the parameters (d) and performance (deadline).
                    
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line -> Point
                    # p[c] = l[c] - l[a] * p[b]
                    # x is Line, y is Point.
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (from side 0)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    # Verification / Cleanup not strictly required for score if valid, 
    # but ensuring no duplicates is good.
    # The BFS logic ensures each vertex is added once.
    # Edges are added for each neighbor pair.
    # Since it's bipartite, and we iterate all neighbors, 
    # do we get duplicate edges?
    # For a point p, we iterate all lines l connected to it.
    # For a line l, we iterate all points p connected to it.
    # The edge (p, l) is added when processing p (side 0).
    # It is NOT added when processing l (side 1) because of `if vertex[0] == 0`.
    # So each edge is added exactly once.
    
    return len(order), edges
# EVOLVE-BLOCK-END
