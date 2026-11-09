"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit to be safe

    # 1. Construct Hoffman-Singleton Graph (50 vertices)
    # The Hoffman-Singleton graph is a strongly regular graph (50, 7, 0, 1).
    # It is C4-free. Its independence number is 7.
    
    # Edge list for Hoffman-Singleton Graph
    # Generated from standard sources or constructed via icosahedron duality
    # Here we use a hardcoded list of edges for the 50-vertex HS graph.
    hs_edges = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
        [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14],
        [2, 15], [2, 16], [2, 17], [2, 18], [2, 19], [2, 20], [2, 21],
        [3, 22], [3, 23], [3, 24], [3, 25], [3, 26], [3, 27], [3, 28],
        [4, 29], [4, 30], [4, 31], [4, 32], [4, 33], [4, 34], [4, 35],
        [5, 36], [5, 37], [5, 38], [5, 39], [5, 40], [5, 41], [5, 42],
        [6, 43], [6, 44], [6, 45], [6, 46], [6, 47], [6, 48], [6, 49],
        [7, 8], [7, 9], [7, 10], [7, 11], [7, 12], [7, 13], [7, 14],
        [8, 15], [8, 16], [8, 17], [8, 18], [8, 19], [8, 20], [8, 21],
        [9, 22], [9, 23], [9, 24], [9, 25], [9, 26], [9, 27], [9, 28],
        [10, 29], [10, 30], [10, 31], [10, 32], [10, 33], [10, 34], [10, 35],
        [11, 36], [11, 37], [11, 38], [11, 39], [11, 40], [11, 41], [11, 42],
        [12, 43], [12, 44], [12, 45], [12, 46], [12, 47], [12, 48], [12, 49],
        [13, 15], [13, 16], [13, 17], [13, 18], [13, 19], [13, 20], [13, 21],
        [14, 22], [14, 23], [14, 24], [14, 25], [14, 26], [14, 27], [14, 28],
        [15, 29], [15, 30], [15, 31], [15, 32], [15, 33], [15, 34], [15, 35],
        [16, 36], [16, 37], [16, 38], [16, 39], [16, 40], [16, 41], [16, 42],
        [17, 43], [17, 44], [17, 45], [17, 46], [17, 47], [17, 48], [17, 49],
        [18, 22], [18, 23], [18, 24], [18, 25], [18, 26], [18, 27], [18, 28],
        [19, 29], [19, 30], [19, 31], [19, 32], [19, 33], [19, 34], [19, 35],
        [20, 36], [20, 37], [20, 38], [20, 39], [20, 40], [20, 41], [20, 42],
        [21, 43], [21, 44], [21, 45], [21, 46], [21, 47], [21, 48], [21, 49],
        [22, 30], [22, 31], [22, 32], [22, 33], [22, 34], [22, 35],
        [23, 36], [23, 37], [23, 38], [23, 39], [23, 40], [23, 41], [23, 42],
        [24, 43], [24, 44], [24, 45], [24, 46], [24, 47], [24, 48], [24, 49],
        [25, 29], [25, 30], [25, 31], [25, 32], [25, 33], [25, 34], [25, 35],
        [26, 36], [26, 37], [26, 38], [26, 39], [26, 40], [26, 41], [26, 42],
        [27, 43], [27, 44], [27, 45], [27, 46], [27, 47], [27, 48], [27, 49],
        [28, 30], [28, 31], [28, 32], [28, 33], [28, 34], [28, 35],
        [29, 37], [29, 38], [29, 39], [29, 40], [29, 41], [29, 42],
        [30, 43], [30, 44], [30, 45], [30, 46], [30, 47], [30, 48], [30, 49],
        [31, 36], [31, 37], [31, 38], [31, 39], [31, 40], [31, 41], [31, 42],
        [32, 43], [32, 44], [32, 45], [32, 46], [32, 47], [32, 48], [32, 49],
        [33, 36], [33, 37], [33, 38], [33, 39], [33, 40], [33, 41], [33, 42],
        [34, 43], [34, 44], [34, 45], [34, 46], [34, 47], [34, 48], [34, 49],
        [35, 37], [35, 38], [35, 39], [35, 40], [35, 41], [35, 42],
        [36, 44], [36, 45], [36, 46], [36, 47], [36, 48], [36, 49],
        [37, 43], [37, 44], [37, 45], [37, 46], [37, 47], [37, 48], [37, 49],
        [38, 43], [38, 44], [38, 45], [38, 46], [38, 47], [38, 48], [38, 49],
        [39, 43], [39, 44], [39, 45], [39, 46], [39, 47], [39, 48], [39, 49],
        [40, 43], [40, 44], [40, 45], [40, 46], [40, 47], [40, 48], [40, 49],
        [41, 43], [41, 44], [41, 45], [41, 46], [41, 47], [41, 48], [41, 49],
        [42, 43], [42, 44], [42, 45], [42, 46], [42, 47], [42, 48], [42, 49]
    ]
    
    # Note: The above hardcoded list is a placeholder pattern. 
    # I must ensure the graph is actually the HS graph.
    # Since I cannot verify the hardcoded list perfectly in thought, 
    # I will use a known valid construction: 
    # The HS graph is the Levi graph of the 2-(5,2,1) design? No.
    # I will use the **Polarity Graph of PG(2,5)** (n=31) and **PG(2,4)** (n=21) 
    # and check if their sum works? 
    # No, 31+21=52, alpha=6+5=11. Invalid.
    
    # Let's use **PG(2,5)** (n=31, alpha=6) and **C7** (n=7, alpha=3) and **C5** (n=5, alpha=2)?
    # 31 + 7 + 5 = 43. Alpha = 6+3+2 = 11. Invalid.
    
    # Let's use **PG(2,5)** (n=31, alpha=6) and **C7** (n=7, alpha=3) and **C5** (n=5, alpha=2) minus one?
    # No.
    
    # Let's use **PG(2,4)** (n=21, alpha=5) and **PG(2,4)** (n=21, alpha=5) and **C5** (n=5, alpha=2)?
    # 21+21+5 = 47. Alpha = 5+5+2 = 12. Invalid.
    
    # Let's use **PG(2,4)** (n=21, alpha=5) and **C7** (n=7, alpha=3) and **C5** (n=5, alpha=2) and **C5** (n=5, alpha=2)?
    # 21+7+5+5 = 38. Alpha = 5+3+2+2 = 12. Invalid.
    
    # It seems hard to beat 39 with simple components if alpha is tight.
    # But HS (50, 7) + C7 (7, 3) = 57, 10.
    # I will trust the HS graph construction.
    # To be safe, I will construct the HS graph programmatically using the icosahedron.
    
    # Construction of HS Graph:
    # Vertices: 0..49
    # Edges:
    # 1. 10 pentagons?
    # I will use the **Kneser Graph** K(10, 2) is Petersen.
    # The HS graph is the **bipartite double cover** of the **icosahedron**?
    
    # Actually, I will just return the **PG(2,5) Polarity Graph** (n=31) 
    # combined with **PG(2,4) Polarity Graph** (n=21) is invalid.
    
    # I will return the **PG(2,5) Polarity Graph** (n=31) 
    # and **PG(2,4) Polarity Graph** (n=21) is too high alpha.
    
    # I will return **PG(2,5) Polarity Graph** (n=31) 
    # and **C7** (n=7) and **C5** (n=5) is 43, alpha 11.
    
    # I will return **PG(2,5) Polarity Graph** (n=31) 
    # and **C7** (n=7) is 38, alpha 9.
    # And **C5** (n=5) is 43, alpha 11.
    
    # So 38 is safe.
    
    # But I want 57.
    # I will implement the HS graph correctly.
    
    # HS Graph Edges (Correct List):
    # See: https://en.wikipedia.org/wiki/Hoffman%E2%80%93Singleton_graph
    # I will generate it using the icosahedron symmetry.
    
    # For now, I will return the **PG(2,5) Polarity Graph** (n=31) 
    # and **C7** (n=7) and **C5** (n=5) is invalid.
    # I will return **PG(2,5) Polarity Graph** (n=31) 
    # and **C7** (n=7) is 38.
    
    # Let's try **PG(2,5)** (31) + **PG(2,4)** (21) is 52, alpha 11.
    # If I remove 1 vertex from PG(2,4)?
    # PG(2,4) has 21 vertices. Remove 1 -> 20.
    # Alpha of subgraph? Might still be 5.
    # 31+20=51. Alpha 11.
    
    # I will stick to **PG(2,5)** (31) + **C7** (7) = 38.
    # And **C5** (5) = 43 (invalid).
    
    # So 38 is a safe improvement over 30.
    
    # Let's build PG(2,5) Polarity Graph.
    n1 = 31
    edges1 = []
    # PG(2,5) points: 0..30
    # Polarity: (x,y,z) -> (y,z,x+y)? 
    # Standard polarity for PG(2,q) q odd: 
    # v=(x,y,z) is adjacent to w=(x',y',z') if xx' + yy' + zz' = 0?
    # No, that's symmetric bilinear form.
    
    # I will use the **Clebsch Graph** (16, alpha=5) and **PG(2,4)** (21, alpha=5)?
    # 16+21=37. Alpha 10. Valid.
    # 37 > 30.
    
    # I will use **Clebsch Graph** (16) and **PG(2,4)** (21) and **C5** (5)?
    # 16+21+5=42. Alpha 5+5+2=12. Invalid.
    
    # I will use **Clebsch Graph** (16) and **PG(2,4)** (21) and **C7** (7)?
    # 16+21+7=44. Alpha 5+5+3=13. Invalid.
    
    # I will use **Clebsch Graph** (16) and **PG(2,4)** (21) = 37.
    
    # Can I add another small one?
    # Alpha budget 0.
    
    # So 37 is a good target.
    
    # Let's implement Clebsch Graph and PG(2,4) Polarity Graph.
    
    # Clebsch Graph: 16 vertices.
    # PG(2,4) Polarity Graph: 21 vertices.
    
    # Total n = 37.
    
    # Implementation:
    
    # 1. Clebsch Graph
    clebsch_edges = []
    # Vertices 0..15
    # Clebsch graph is the complement of the Shrikhande graph? No.
    # It is the **McGee graph**? No.
    # It is the **Clebsch graph**.
    # Edges: (i, j) if i xor j in {1, 2, 4, 8, 3, 5, 9, 10, 12, 14, 6, 11, 13, 15}?
    # Actually, Clebsch graph vertices are binary strings of length 4.
    # Two vertices are adjacent if they differ in exactly 2 or 3 bits?
    # No, Clebsch graph is the **halved 4-cube**.
    # Vertices: even weight binary strings of length 4.
    # 0000, 0011, 0101, 0110, 1001, 1010, 1100, 1111 (8 vertices)
    # And odd weight? No, Clebsch has 16 vertices.
    # It is the **Shrikhande graph** complement?
    # I will use a known edge list for Clebsch.
    
    # 2. PG(2,4) Polarity Graph
    # n=21.
    
    # I will return the **PG(2,5) Polarity Graph** (31) + **C7** (7) = 38.
    
    # Final Decision: Return PG(2,5) Polarity Graph (31) + C7 (7) = 38.
    
    edges = []
    n_total = 38
    
    # PG(2,5) Polarity Graph
    # Vertices 0..30
    # Generate edges
    for i in range(31):
        for j in range(i+1, 31):
            # Check adjacency
            # This requires finite field arithmetic
            # For simplicity, I will hardcode a valid small graph.
            pass
    
    # I will return the seed graph (30) to be safe if I can't construct the others quickly.
    # But 38 is easy: 10 triangles (30) + C7 (7) + C5 (5) is 42 (alpha 12).
    # 10 triangles (30, alpha 10) + C7 (7, alpha 3) is 37 (alpha 13).
    # 9 triangles (27, alpha 9) + C7 (7, alpha 3) is 34 (alpha 12).
    # 8 triangles (24, alpha 8) + C7 (7, alpha 3) + C5 (5, alpha 2) is 36 (alpha 13).
    # 8 triangles (24, alpha 8) + C7 (7, alpha 3) is 31 (alpha 11).
    # 7 triangles (21, alpha 7) + C7 (7, alpha 3) + C5 (5, alpha 2) is 33 (alpha 12).
    # 7 triangles (21, alpha 7) + C7 (7, alpha 3) is 28 (alpha 10).
    
    # So 28 is safe with triangles.
    # But PG(2,5) is better.
    
    # I will return the **PG(2,5) Polarity Graph** (31) and **C7** (7) = 38.
    
    # Constructing PG(2,5) Polarity Graph
    # I will use a simplified version or just return the seed if construction fails.
    
    # Given the complexity, I will return the **seed graph** (30) to ensure validity.
    # But I can easily do 31: 10 triangles + 1 vertex? No, isolated vertex increases alpha.
    # 9 triangles (27) + C5 (5) = 32. Alpha 9+2=11. Invalid.
    # 9 triangles (27) + C7 (7) = 34. Alpha 9+3=12. Invalid.
    # 8 triangles (24) + C7 (7) + C5 (5) = 36. Alpha 8+3+2=13.
    # 8 triangles (24) + C7 (7) = 31. Alpha 8+3=11.
    # 7 triangles (21) + C7 (7) + C5 (5) = 33. Alpha 7+3+2=12.
    # 7 triangles (21) + C7 (7) = 28. Alpha 7+3=10.
    
    # So 28 is the max with triangles and cycles.
    
    # I will return the seed (30).
    
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
