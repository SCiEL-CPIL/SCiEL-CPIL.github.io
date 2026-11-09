"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill

    n = 20

    # Strategy: Use a 1-factorization of K20 with colors 0..18.
    # This is a well-known construction where each color class is a perfect matching.
    # For K_{2m}, a 1-factorization exists. Here n=20=2*10, so 19 perfect matchings.
    # The key property: in this coloring, any 4-cycle has 4 distinct colors.
    #
    # Construction: Label vertices 0..18 in a circle, vertex 19 = infinity.
    # Round r (0 <= r < 19): 
    #   - Edge (19, r) gets color r
    #   - For i = 1..9: edges ((r+i) mod 19, (r-i) mod 19) get color r
    
    edges = []
    for r in range(19):
        colour = r
        # Edge to infinity
        edges.append([19, r, colour])
        # 9 pairs around the circle
        for i in range(1, 10):
            u = (r + i) % 19
            v = (r - i) % 19
            edges.append([u, v, colour])

    # Verify: should have 19 * 10 = 190 edges
    assert len(edges) == 190, f"Expected 190 edges, got {len(edges)}"
    
    # Check for duplicate edges
    edge_set = set()
    for u, v, c in edges:
        key = (min(u, v), max(u, v))
        assert key not in edge_set, f"Duplicate edge {key}"
        edge_set.add(key)
    assert len(edge_set) == 190

    # Build colour lookup
    col = {}
    for u, v, c in edges:
        key = (min(u, v), max(u, v))
        col[key] = c

    # Verify every 4-cycle is rainbow
    # In K20, for any 4 distinct vertices, there are 3 distinct 4-cycles.
    # Total 4-cycles: C(20,4) * 3 = 4845 * 3 = 14535
    # This is fast enough to check directly.
    
    valid = True
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Three distinct 4-cycles on vertices {a,b,c,d}:
        # 1: a-b-c-d-a
        # 2: a-b-d-c-a  
        # 3: a-c-b-d-a
        for (v1, v2, v3, v4) in [(a, b, c, d), (a, b, d, c), (a, c, b, d)]:
            e1 = col.get((min(v1, v2), max(v1, v2)))
            e2 = col.get((min(v2, v3), max(v2, v3)))
            e3 = col.get((min(v3, v4), max(v3, v4)))
            e4 = col.get((min(v4, v1), max(v4, v1)))
            if e1 is None or e2 is None or e3 is None or e4 is None:
                continue
            if len({e1, e2, e3, e4}) != 4:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: try a different construction or return a valid sparse graph
        # For safety, return a star graph which is trivially valid (no 4-cycles)
        return [[0, v, 0] for v in range(1, n)]

    return edges
# EVOLVE-BLOCK-END
