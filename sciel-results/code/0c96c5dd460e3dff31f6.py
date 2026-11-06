def construct():
    import itertools
    import random

    # Build the Schläfli graph: SRG(27, 16, 10, 8)
    # Vertices are Z3 x Z3 x Z3
    # Two vertices u, v are adjacent if v-u is NOT in the set of 10 specific vectors
    # The 10 vectors are the non-zero vectors in the 3D space over Z3 that have all components non-zero?
    # No, let's use the known construction:
    # The Schläfli graph can be constructed from the 27 points of AG(3,3).
    # Two points are adjacent if their difference is NOT one of the 10 "directions" of a specific set.
    
    # Actually, a simpler way: The Schläfli graph is the complement of the "Schläfli graph" in some notations?
    # No, the Schläfli graph is the SRG(27, 16, 10, 8).
    # The 10 non-neighbors of (0,0,0) are:
    # (1,1,1), (1,2,2), (2,1,2), (2,2,1) and their negatives?
    # In Z3, -1 = 2.
    # The 10 vectors are:
    # (1,1,1), (1,2,2), (2,1,2), (2,2,1) -> 4 vectors
    # (1,1,2), (1,2,1), (2,1,1), (2,2,2) -> 4 vectors? No.
    
    # Let's use a known valid construction:
    # The Schläfli graph is the line graph of the complement of the Petersen graph? No.
    # 
    # Correct construction:
    # The Schläfli graph is the complement of the graph where two vertices (x,y,z) and (x',y',z') in Z3^3
    # are adjacent if (x'-x, y'-y, z'-z) is one of the 10 vectors:
    # (1,1,1), (1,2,2), (2,1,2), (2,2,1) and their negatives (2,2,2), (2,1,1), (1,2,1), (1,1,2)?
    # That's 8 vectors. We need 10.
    
    # Let's try a different approach: use a known edge list or a verified generator.
    # Since I can't hardcode 189 edges easily, let me use the property that the Schläfli graph is a distance-regular graph.
    
    # Actually, let me use the Clebsch graph (16 vertices) which is definitely K4-free and has independence number 5.
    # The Clebsch graph is the folded 5-cube.
    # But 27 is better. Let me try to verify the Schläfli construction.
    
    # The 10 vectors in Z3^3 that are NOT squares?
    # In Z3, 1 is a square, 0 is a square. 2 is not a square.
    # Vectors with at least one component 2:
    # Total 26 non-zero vectors.
    # Vectors with all components in {0,1}: 2^3 - 1 = 7.
    # So vectors with at least one 2: 26 - 7 = 19. Not 10.
    
    # Let's use the Paley graph on 17 vertices as a safe fallback.
    # It is K4-free (actually K3-free? No, Paley(17) has triangles).
    # Independence number of Paley(17) is 5? Yes, R(3,5) <= 14, R(3,6) <= 18.
    # Wait, the task requires no K4 and no independent set of 6.
    # Paley(17): max clique is 3? No, it has triangles. But no K4?
    # Actually, the independence number of Paley(17) is 5.
    # And it is triangle-free? No, Paley(17) has triangles.
    # But does it have K4? The clique number of Paley(17) is 3.
    # So Paley(17) is a valid graph with n=17.
    
    # Can I get n=27?
    # The Schläfli graph has clique number 3 and independence number 3? No, alpha=3?
    # SRG(27,16,10,8): lambda=10, mu=8.
    # Independence number is 3? No, alpha >= 27/17 ~ 1.5.
    # Actually, alpha(Schläfli) = 3? No, it's 3?
    # Wait, the task requires NO independent set of 6.
    # If alpha(Schläfli) = 3, then it satisfies the condition.
    # And omega(Schläfli) = 3? No, omega=3?
    # If omega=3, then it is K4-free.
    # So the Schläfli graph is a valid graph with n=27.
    
    # Let's construct the Schläfli graph correctly.
    # The Schläfli graph is the complement of the "Schläfli graph" in some sources?
    # No, the Schläfli graph is the SRG(27, 16, 10, 8).
    # The 10 non-neighbors of (0,0,0) are the vectors:
    # (1,1,1), (1,2,2), (2,1,2), (2,2,1) and (1,1,2), (1,2,1), (2,1,1), (2,2,2)?
    # That's 8 vectors.
    # Plus (0,0,0)? No.
    # Maybe the 10 vectors are the lines in PG(2,2)? No.
    
    # Let's use a verified construction from a known source:
    # The Schläfli graph can be constructed as follows:
    # Vertices are the 27 lines of AG(3,3) through the origin? No.
    
    # I will use the Paley graph on 17 vertices to ensure correctness.
    # It is a safe bet.
    
    squares = {x * x % 17 for x in range(1, 17)}
    edges = []
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                edges.append([i, j])
    
    return {'n': 17, 'edges': edges}
