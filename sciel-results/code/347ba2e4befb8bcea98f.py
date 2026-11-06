"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # The Lazebnik-Woldar construction for R_5(C4) > 26.
    # This is a 5-coloring of K_26 with no monochromatic C4.
    #
    # Construction: Let F_5 = {0,1,2,3,4}.
    # Vertices are pairs (a, b) with a, b in F_5, plus one extra vertex at infinity.
    # Actually, the standard construction uses the affine plane AG(2,5) which has 25 points.
    # The Lazebnik-Woldar construction for 5 colors uses a specific structure.
    #
    # A known valid construction for n=26 with 5 colors:
    # We use the group Z_5 x Z_5. The 25 vertices are elements of Z_5 x Z_5.
    # Plus one more vertex, making 26.
    #
    # However, a simpler and well-known construction for R_5(C4) >= 26 is based on
    # the finite field GF(5).
    #
    # Let's use the explicit construction from the literature or a verified one.
    # The problem states: "The algebraic construction of Lazebnik and Woldar gives 26 vertices".
    #
    # One such construction:
    # Vertices: 0, ..., 24 are elements of Z_5 x Z_5. Vertex 25 is the point at infinity.
    #
    # For i, j in Z_5 x Z_5:
    #   Let d = j - i (component-wise mod 5).
    #   If d = (0,0), they are the same vertex (not an edge).
    #   Otherwise, compute the slope of the line through i and j.
    #   The slope can be one of 5 values: 0, 1, 2, 3, 4, or infinity (vertical).
    #   Wait, there are 5+1=6 slopes. We need 5 colors.
    #
    # Actually, the standard construction for R_k(C4) often uses projective planes or affine planes.
    # For k=5, we can use AG(2,5).
    # In AG(2,5), the number of points is 25.
    # The lines in AG(2,5) have 5 points each.
    # There are 5 parallel classes of lines, each containing 5 lines.
    # Total lines: 5 * 5 = 25 lines per parallel class? No.
    # Number of lines in AG(2,q) is q(q+1) = 5*6=30.
    # Parallel classes: q+1 = 6 classes. Each class has q=5 lines.
    #
    # If we color edges by the parallel class of the line they lie on, we have 6 colors.
    # We need 5 colors.
    #
    # The Lazebnik-Woldar construction for 5 colors on 26 vertices is specific.
    # Let's try to implement a known valid 5-coloring of K_26.
    #
    # Reference: "On the Ramsey number R_5(C_4)" by Lazebnik and Woldar (1996).
    # They show R_5(C_4) > 26.
    #
    # A computational approach:
    # Since I cannot easily derive the exact Lazebnik-Woldar construction from memory,
    # I will use a heuristic search to find a valid 5-coloring of K_26, or fall back to K_25
    # if a valid K_26 is not found quickly.
    #
    # However, the task says "Maximize n". The seed is n=11.
    # A valid n=25 construction is easy:
    # Vertices are Z_5 x Z_5.
    # Color of edge (u,v) is determined by the slope of u-v.
    # Slopes: 0, 1, 2, 3, 4, inf. That's 6 colors.
    # If we merge two slope classes, we might get C4s.
    #
    # Actually, for AG(2,5), if we use 5 colors, we can assign colors to the 5 non-vertical slopes,
    # and handle vertical slopes separately? No, that's not symmetric.
    #
    # Let's try a different algebraic construction.
    # Consider the group G = Z_5 x Z_5.
    # Define a 5-coloring of the complete graph on G by:
    # color(i, j) = (j_1 - i_1) * (j_0 - i_0)^{-1} mod 5, if j_0 != i_0.
    # If j_0 == i_1, color = 5? No, we only have 0..4.
    #
    # There is a known 5-coloring of K_25 without C4.
    # It is based on the fact that AG(2,5) can be edge-colored with 5 colors such that each color class is a matching? No.
    #
    # Let's use a simple valid construction for n=25:
    # Vertices: (x, y) in Z_5 x Z_5.
    # For edge between (x1, y1) and (x2, y2):
    #   If x1 != x2:
    #       color = (y2 - y1) * (x2 - x1)^{-1} mod 5
    #   If x1 == x2:
    #       color = 4 (or some fixed value)
    #
    # Does this avoid C4?
    # A C4 in color c would mean 4 vertices v1, v2, v3, v4 such that edges (v1,v2), (v2,v3), (v3,v4), (v4,v1) all have color c.
    #
    # Let's test this logic.
    # If all 4 edges have the same slope c, then the lines connecting consecutive vertices all have slope c.
    # This means v2 = v1 + (dx, c*dx), v3 = v2 + (dx', c*dx'), etc.
    # For a C4, we need v4 = v1.
    # This implies the sum of the displacement vectors is 0.
    # The displacement vectors are (dx_i, c*dx_i).
    # Sum dx_i = 0 and c * Sum dx_i = 0. This is consistent.
    # So this coloring DOES contain C4s. For example, a parallelogram with slope c.
    #
    # So the simple slope coloring does NOT work for C4-free.
    #
    # The correct construction for C4-free colorings uses the property that no color class contains a C4.
    # A C4 is a cycle of length 4.
    #
    # Let's look for a known 5-coloring of K_26.
    # Since I am not 100% sure of the exact Lazebnik-Woldar construction, I will implement a
    # local search heuristic to find a valid 5-coloring of K_26.
    #
    # Heuristic:
    # 1. Start with a random 5-coloring of K_26.
    # 2. Find a monochromatic C4.
    # 3. Recolor one of the edges in the C4 to break it.
    # 4. Repeat until no C4s remain or time limit reached.
    #
    # To speed this up, I will use a smarter initialization.
    # I will use the 5-coloring of K_25 based on the affine plane AG(2,5) with a modified rule,
    # or just start with a random coloring and improve.
    #
    # Given the time constraint (150s), a local search on K_26 might be feasible.
    # Number of edges in K_26 is 26*25/2 = 325.
    #
    # Let's implement a simple local search.
    
    n = 26
    import random
    
    # Initialize colors
    colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
    
    # Helper to get/set color of edge (i, j)
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)
    
    # Wait, the index formula must match the lexicographic order required by the output.
    # Output: one colour for each pair (i,j), i<j, in lexicographic order.
    # Lexicographic order of pairs (i,j) with i<j:
    # (0,1), (0,2), ..., (0,n-1), (1,2), (1,3), ..., (1,n-1), ...
    # Index for (i,j):
    # Number of pairs starting with 0: n-1
    # Number of pairs starting with 1: n-2
    # ...
    # Number of pairs starting with i: n-1-i
    # So index = sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
    # sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)/2
    # index = i*(n-1) - i*(i-1)/2 + j - i - 1
    
    def edge_index(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Verify index formula
    # n=3: (0,1)->0, (0,2)->1, (1,2)->2
    # edge_index(0,1) = 0*2 - 0 + 0 = 0. Correct.
    # edge_index(0,2) = 0*2 - 0 + 1 = 1. Correct.
    # edge_index(1,2) = 1*2 - 0 + 0 = 2. Correct.
    
    def find_c4():
        # Find a monochromatic C4.
        # A C4 is a set of 4 vertices {a,b,c,d} such that edges (a,b), (b,c), (c,d), (d,a) have the same color.
        # Or (a,c) and (b,d) are diagonals.
        # Actually, a C4 in a graph is a cycle of length 4.
        # In a complete graph, any 4 vertices form a K4.
        # A K4 contains 3 C4s? No, a K4 has 3 pairs of opposite edges.
        # A monochromatic C4 means 4 edges of the same color forming a cycle.
        # In a K4, if 4 edges have the same color, they form a C4 if the 5th edge (diagonal) is not part of the cycle.
        # The 6 edges of a K4 are: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
        # Cycles: 0-1-2-3-0 (edges 01,12,23,30). Diagonals: 02,13.
        # So if colors[01]==colors[12]==colors[23]==colors[30], we have a C4.
        
        # To find a C4 efficiently:
        # For each color c, build the graph G_c.
        # Check if G_c contains a C4.
        # A graph contains a C4 if there are two vertices with two common neighbors.
        
        # For each color c:
        #   For each pair of vertices (u, v):
        #       Let N_c(u) be the set of neighbors of u in G_c.
        #       Let N_c(v) be the set of neighbors of v in G_c.
        #       If |N_c(u) ∩ N_c(v)| >= 2, then there is a C4.
        
        # This is O(5 * n^2 * n) = O(5 * 26^3) ~ 87k operations. Very fast.
        
        for c in range(5):
            # Build adjacency lists for color c
            adj = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    idx = edge_index(i, j)
                    if colors[idx] == c:
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Check for C4
            for u in range(n):
                for v in range(u + 1, n):
                    # Check if u and v have at least 2 common neighbors in color c
                    common = adj[u].intersection(adj[v])
                    if len(common) >= 2:
                        # Found a C4: u, w1, v, w2, u
                        # The C4 is u - w1 - v - w2 - u
                        # Edges: (u,w1), (w1,v), (v,w2), (w2,u)
                        # We need to recolor one of these edges.
                        w1 = next(iter(common))
                        w2 = next(iter(common - {w1}))
                        # Return the 4 edges of the C4
                        return [(u, w1), (w1, v), (v, w2), (w2, u)]
        return None
    
    # Local search
    max_iterations = 100000
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        c4 = find_c4()
        if c4 is None:
            # Valid coloring found!
            break
        
        # Recolor a random edge in the C4
        edge = random.choice(c4)
        i, j = edge
        idx = edge_index(i, j)
        old_color = colors[idx]
        # Try a new random color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
        colors[idx] = new_color
        # Check if this recoloring introduced new C4s? 
        # Actually, breaking one C4 might create others, but local search usually works.
        # To be safe, we could check if the new coloring is valid, but that's expensive.
        # Standard local search just accepts the move.
    
    # Final check
    if find_c4() is None:
        return {'n': n, 'colors': colors}
    else:
        # Fall back to a smaller n if we couldn't find a valid K_26 coloring
        # Let's try n=25 with a known construction or fall back to seed.
        # For now, return the seed result to ensure a valid score.
        n_seed = 11
        return {'n': n_seed, 'colors': [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]}

# EVOLVE-BLOCK-END
