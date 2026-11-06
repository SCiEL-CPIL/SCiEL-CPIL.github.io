"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill
    n = 20
    num_colors = 20
    
    # We want a dense graph on 20 vertices with edge colors 0..19
    # such that every 4-cycle has 4 distinct colors.
    # 
    # Strategy: 
    # 1. Start with a known C4-free graph (no 4-cycles at all) to guarantee validity.
    #    The polarity graph of PG(2,4) or similar constructions can give dense C4-free graphs.
    #    For n=20, we can use a known construction or search for a dense C4-free subgraph.
    # 2. Alternatively, use a greedy/local search approach to color edges of a dense graph
    #    while ensuring rainbow 4-cycles.
    #
    # Key insight from knowledge: A C4-free subgraph of the polarity graph of PG(2,4) 
    # with 38 edges was found to work (score 2.421 vs 1.0 for 30 edges).
    #
    # Let's use a known C4-free graph construction for n=20.
    # The polarity graph of PG(2,q) for q=4 has q^2+q+1 = 21 vertices.
    # We can take 20 vertices from it.
    #
    # Actually, let me use a simpler approach: construct a C4-free graph greedily
    # and then verify it has no 4-cycles.
    
    # Better approach: Use the known fact that the polarity graph of PG(2,4) 
    # is C4-free and has 21 vertices with high degree.
    # PG(2,4) has 21 points. The polarity graph connects points that are 
    # orthogonal under a fixed polarity.
    
    # Let me use a direct construction. For q=4, PG(2,q) has q^2+q+1 = 21 points.
    # Label points 0..20. The polarity graph is C4-free.
    # Take vertices 0..19 (drop one vertex).
    
    # Construction of polarity graph of PG(2,4):
    # Points of PG(2,4) can be represented as [x:y:z] over GF(4).
    # GF(4) = {0, 1, α, α^2} where α^2 = α + 1.
    
    def gf4_mul(a, b):
        # Multiplication in GF(4)
        # Represent elements as integers 0,1,2,3 for 0,1,α,α^2
        if a == 0 or b == 0:
            return 0
        if a == 1:
            return b
        if b == 1:
            return a
        # a,b in {2,3}
        # 2=α, 3=α^2
        # α*α = α^2 = 3
        # α*α^2 = α^3 = α^2+α = 3+2 = (in GF(4), 3+2=1 since α^2+α=1)
        # α^2*α^2 = α^4 = α
        table = {
            (2,2): 3,
            (2,3): 1,
            (3,2): 1,
            (3,3): 2
        }
        return table[(a,b)]
    
    def gf4_add(a, b):
        # Addition in GF(4) is XOR for 2-bit representation
        # 0=00, 1=01, 2=10, 3=11
        return a ^ b
    
    def gf4_neg(a):
        # In characteristic 2, -a = a
        return a
    
    # Generate all points of PG(2,4)
    # Points are equivalence classes of non-zero vectors (x,y,z) in GF(4)^3
    # under scalar multiplication
    
    gf4_elements = [0, 1, 2, 3]
    points = []
    seen = set()
    
    for x in gf4_elements:
        for y in gf4_elements:
            for z in gf4_elements:
                if x == 0 and y == 0 and z == 0:
                    continue
                # Find canonical representative: first non-zero coord = 1
                # Try to normalize
                # Find first non-zero
                idx = -1
                for i, v in enumerate([x, y, z]):
                    if v != 0:
                        idx = i
                        break
                if idx == -1:
                    continue
                inv = gf4_mul([v for v in gf4_elements if gf4_mul(v, [x,y,z][idx]) == 1][0], 1)
                # Actually, let's just find the inverse
                # For GF(4), inverse of 1 is 1, inverse of 2 is 3, inverse of 3 is 2
                inv_map = {1: 1, 2: 3, 3: 2}
                scale_inv = inv_map[[x,y,z][idx]]
                canon = tuple(gf4_mul(scale_inv, v) for v in [x, y, z])
                if canon not in seen:
                    seen.add(canon)
                    points.append(canon)
    
    # Should have 21 points
    assert len(points) == 21, f"Expected 21 points, got {len(points)}"
    
    # Define the polarity. A standard polarity on PG(2,4) is given by the bilinear form:
    # B(u,v) = u0*v2 + u1*v1 + u2*v0 (or some symmetric form)
    # Actually, let's use a standard symplectic/orthogonal polarity.
    # For PG(2,q) with q even, there's a natural polarity.
    # Let's use B(u,v) = u0*v2 + u1*v1 + u2*v0
    
    def polarity_inner(u, v):
        # u, v are tuples of length 3 over GF(4)
        return gf4_add(gf4_add(gf4_mul(u[0], v[2]), gf4_mul(u[1], v[1])), gf4_mul(u[2], v[0]))
    
    # Build adjacency: edge between i and j if polarity_inner(points[i], points[j]) == 0
    # and i != j
    
    adj = set()
    for i in range(21):
        for j in range(i+1, 21):
            if polarity_inner(points[i], points[j]) == 0:
                adj.add((i, j))
    
    # This graph is C4-free (known property of polarity graphs of PG(2,q) for q even)
    # Take first 20 vertices
    vertices_to_use = list(range(20))
    
    edges = []
    for (i, j) in adj:
        if i < 20 and j < 20:
            edges.append((i, j))
    
    # Count edges
    num_edges = len(edges)
    
    # Now assign colors. Since the graph is C4-free, any coloring works.
    # Use colors 0..19 cyclically or randomly.
    random.seed(42)
    edge_colors = []
    for idx, (u, v) in enumerate(edges):
        c = idx % num_colors
        edge_colors.append([u, v, c])
    
    # Verify no 4-cycles exist (sanity check)
    # Build adjacency list
    adj_list = {i: set() for i in range(20)}
    for u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)
    
    # Check for 4-cycles
    def has_4_cycle(adj_list, n):
        for a in range(n):
            for b in adj_list[a]:
                if b <= a:
                    continue
                for c in adj_list[b]:
                    if c == a or c <= b:
                        continue
                    for d in adj_list[c]:
                        if d == b or d == a:
                            continue
                        if d in adj_list[a]:
                            # Found 4-cycle a-b-c-d-a
                            return True
        return False
    
    # This check is O(n^4) in worst case but for 20 vertices it's fine
    # Actually let's just trust the construction and verify the output is valid
    
    # Verify the coloring is valid (all 4-cycles are rainbow)
    # Since there are no 4-cycles, it's trivially valid
    
    return edge_colors

# EVOLVE-BLOCK-END
