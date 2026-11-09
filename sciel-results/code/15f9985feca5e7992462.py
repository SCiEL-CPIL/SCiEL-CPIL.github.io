"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill

    n = 20
    num_colors = 20

    # Pre-compute all 4-cycles in K_n (n=20)
    # A 4-cycle is a set {a,b,c,d} with distinct vertices and edges (a,b),(b,c),(c,d),(d,a)
    # We enumerate all ordered quadruples (a,b,c,d) where a < b < c < d
    # and consider the 3 possible 4-cycles on these 4 vertices:
    #   1: a-b-c-d-a
    #   2: a-b-d-c-a
    #   3: a-c-b-d-a
    # For each, we need edges: (a,b),(b,c),(c,d),(d,a) etc.

    all_4cycles = []  # list of tuples of 4 edge indices

    # First, map edge (u,v) with u<v to an index
    edge_index = {}
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[(u, v)] = len(edge_list)
            edge_list.append((u, v))
    m = len(edge_list)  # 190

    def get_edge_idx(u, v):
        if u > v:
            u, v = v, u
        return edge_index.get((u, v), None)

    # Enumerate all 4-cycles
    from itertools import combinations, permutations
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # 3 distinct 4-cycles on these 4 vertices
        # Cycle 1: a-b-c-d-a => edges (a,b),(b,c),(c,d),(d,a)
        e1 = [get_edge_idx(a,b), get_edge_idx(b,c), get_edge_idx(c,d), get_edge_idx(d,a)]
        if all(e is not None for e in e1):
            all_4cycles.append(tuple(sorted(set(e1))))  # store as sorted tuple of edge indices

        # Cycle 2: a-b-d-c-a => edges (a,b),(b,d),(d,c),(c,a)
        e2 = [get_edge_idx(a,b), get_edge_idx(b,d), get_edge_idx(d,c), get_edge_idx(c,a)]
        if all(e is not None for e in e2):
            all_4cycles.append(tuple(sorted(set(e2))))

        # Cycle 3: a-c-b-d-a => edges (a,c),(c,b),(b,d),(d,a)
        e3 = [get_edge_idx(a,c), get_edge_idx(c,b), get_edge_idx(b,d), get_edge_idx(d,a)]
        if all(e is not None for e in e3):
            all_4cycles.append(tuple(sorted(set(e3))))

    # Build conflict graph: for each 4-cycle, any two edges in it form a conflict pair
    # But we need the actual constraint: in each 4-cycle, the 4 edges must have distinct colors
    # This is a global constraint, not just pairwise.

    # Strategy: Use a CSP approach.
    # Variables: edges (190 of them), each can be "present" or "absent"
    # But also color assignment for present edges.
    # This is complex. Let's use a different approach.

    # Better approach: Start with K_20 and greedily remove edges to satisfy constraints.
    # But K_20 has many 4-cycles with potential conflicts.
    #
    # Alternative: Construct a graph with a known good structure.
    #
    # Key insight: If we use a graph that is C4-free, the constraint is vacuously satisfied.
    # The maximum number of edges in a C4-free graph on 20 vertices is known to be around 40-50.
    # But we can do better by allowing some 4-cycles as long as their edges have distinct colors.
    #
    # Let's try a structured construction:
    # Use a complete bipartite graph K_{10,10} and color edges by their position.
    # K_{10,10} has 100 edges. It contains many 4-cycles.
    # In K_{10,10} with parts L={l0..l9} and R={r0..r9}, a 4-cycle is l_i - r_j - l_k - r_l - l_i
    # with i!=k and j!=l. The edges are (l_i,r_j), (r_j,l_k), (l_k,r_l), (r_l,l_i).
    # If we color edge (l_i, r_j) with color (i+j) % 20, then:
    # Edge colors: (i+j)%20, (k+j)%20, (k+l)%20, (i+l)%20
    # Are these distinct? Not necessarily.
    #
    # Let's try coloring edge (l_i, r_j) with color (i*10 + j) % 20.
    # Then: (i*10+j)%20, (k*10+j)%20, (k*10+l)%20, (i*10+l)%20
    # (i*10+j)%20 = (j + (i*10)%20)%20. Since i<10, i*10 is 0,10,20,...,90.
    # i*10 % 20 = 0 if i even, 10 if i odd.
    # So color = j if i even, j+10 if i odd.
    # For a 4-cycle with l_i, l_k (i!=k) and r_j, r_l (j!=l):
    # If i,k have same parity: colors are j, j, l, l -> not distinct!
    # So this doesn't work.
    #
    # Let's try a different approach: Use a projective plane or similar structure.
    #
    # Actually, let's use the greedy C4-free approach but try to maximize edges.
    # The current program does this with a specific seed. Let's try multiple seeds and keep the best.

    best_edges = []
    best_count = 0

    # Try multiple random orderings
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        if time.time() - t0 > DEADLINE:
            break

        adj = [0] * n
        result_edges = []

        all_edge_indices = list(range(m))
        random.seed(seed)
        random.shuffle(all_edge_indices)

        for idx in all_edge_indices:
            if time.time() - t0 > DEADLINE - 5:
                break

            u, v = edge_list[idx]

            # Check if adding (u, v) creates a C4 with non-rainbow colors
            # But we haven't assigned colors yet. Let's first build a C4-free graph.
            # Then assign colors.

            # For C4-free: check if exists w such that (v,w) in E and N(w) intersects N(u)
            is_c4 = False
            for w in range(n):
                if w == u or w == v:
                    continue
                if (adj[v] >> w) & 1:
                    if (adj[w] & adj[u]) != 0:
                        is_c4 = True
                        break

            if not is_c4:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                result_edges.append((u, v))

        count = len(result_edges)
        if count > best_count:
            best_count = count
            best_edges = list(result_edges)

    # Now we have a C4-free graph with best_count edges.
    # Assign colors: since no 4-cycles, any coloring works. Use color 0 for all.
    # But the problem says "edge-coloured with 20 colours". Let's use distinct colors to be safe.
    # Actually, the constraint is only on 4-cycles. If no 4-cycles, any coloring is valid.
    # Let's assign colors 0..19 cyclically to use all colors.

    final_edges = []
    for i, (u, v) in enumerate(best_edges):
        c = i % num_colors
        final_edges.append([u, v, c])

    return final_edges

# EVOLVE-BLOCK-END
