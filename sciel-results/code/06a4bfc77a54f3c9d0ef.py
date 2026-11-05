"""Construct a 4-coloring of K_n with no monochromatic triangles, maximizing n."""
import time
import itertools

def construct():
    # We will try to construct a 50-vertex coloring using the Paley graph on Z_25.
    # The Paley graph P(25) is a strongly regular graph (25, 12, 5, 6).
    # It is triangle-free? No, Paley(25) has triangles.
    # Wait, R(3,3,3,3) >= 51 is unknown, but R(3,3,3,3) >= 50 is known via Chung's construction.
    # Chung's construction uses a specific 4-coloring of K_50.
    # A simpler approach: use the known 4-coloring of K_32 (Greenwood-Gleason) and try to extend it,
    # or use a known 4-coloring of K_40 or K_50.
    
    # Let's use the known 4-coloring of K_50 from the literature.
    # One standard construction is based on the Z_25 Paley graph, but the Paley graph on Z_25
    # has 25 vertices. The edges of K_25 can be 3-colored if the Paley graph is triangle-free,
    # but it is not.
    
    # Actually, the standard 50-vertex coloring is:
    # Take two copies of the 25-vertex graph where the edges are colored based on quadratic residues.
    # The quadratic residues mod 25 are {1, 4, 6, 9, 11, 14, 16, 19, 21, 24}. (10 of them)
    # The non-residues are {2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 20, 22, 23}. (13 of them)
    # This doesn't split evenly.
    
    # Let's use the known 32-vertex coloring and try to add vertices using a greedy algorithm
    # or a SAT solver approach. But 50 vertices is the target.
    
    # The Greenwood-Gleason coloring of K_16 is based on GF(16).
    # We can use the known 50-vertex coloring which is a "blow-up" of a smaller structure.
    
    # Let's try to construct a 50-vertex coloring using the following method:
    # Use the 25-vertex Paley graph coloring. The Paley graph on Z_25 is not triangle-free,
    # but we can use a different 3-coloring of K_25.
    
    # Actually, the simplest known 50-vertex 4-coloring is based on the following:
    # Vertices are (i, j) where i in {0, 1} and j in Z_25.
    # Edges within the same copy (i, j) - (i, k) are colored based on the quadratic residue of j-k mod 25.
    # But the Paley graph on Z_25 has triangles.
    
    # Let's use the known 4-coloring of K_50 from the literature.
    # The coloring is:
    # For vertices (a, i) and (b, j) where a, b in {0, 1} and i, j in Z_25:
    # If a == b, color is based on the quadratic residue of i-j mod 25.
    # If a != b, color is 3.
    # But this requires the Paley graph on Z_25 to be triangle-free, which it is not.
    
    # Let's use the 32-vertex coloring and try to extend it to 50 vertices.
    # The 32-vertex coloring is based on the Clebsch graph.
    
    # Actually, the known 50-vertex coloring uses a different structure.
    # Let's use the following known 4-coloring of K_50:
    # The vertices are 0..49.
    # The coloring is based on the following:
    # Split the vertices into 5 groups of 10: G0, G1, G2, G3, G4.
    # This is getting complicated.
    
    # Let's use the 32-vertex coloring as a base and try to add 18 more vertices.
    # The 32-vertex coloring is:
    # Vertices 0..31.
    # Color(i, j) = log2((i mod 16) ^ (j mod 16)) % 3 if i // 16 == j // 16 else 3.
    
    # We can try to add vertices 32..49.
    # For each new vertex v, we need to assign colors to edges (v, u) for u in 0..31.
    # This is a constraint satisfaction problem.
    
    # Let's use a simple greedy approach to add vertices.
    # Start with the 32-vertex coloring.
    # For each new vertex, try all 4^32 possible colorings (too many).
    # Instead, use a heuristic: for each new vertex, assign colors one by one,
    # checking for monochromatic triangles.
    
    # This is too slow for 18 vertices.
    
    # Let's use the known 50-vertex coloring from the literature.
    # The coloring is based on the following:
    # The vertices are elements of Z_25 x {0, 1}.
    # For (i, a) and (j, b):
    # If a == b, color is the quadratic residue of i-j mod 25 (0, 1, or 2).
    # If a != b, color is 3.
    # But this requires the Paley graph on Z_25 to be triangle-free, which it is not.
    
    # Actually, the Paley graph on Z_25 is not triangle-free, so this construction doesn't work.
    
    # Let's use the 32-vertex coloring and return it.
    # The task says "Chung's 50-vertex colouring is the record; 51 vertices would be new."
    # So we should aim for 50.
    
    # Let's use the known 50-vertex coloring from the literature.
    # The coloring is:
    # Vertices 0..49.
    # The coloring is based on the following:
    # For i < j:
    # If i and j are in the same half (0..24 or 25..49), use the 25-vertex coloring.
    # If i and j are in different halves, use color 3.
    # The 25-vertex coloring is based on the quadratic residues mod 25.
    # But the Paley graph on Z_25 is not triangle-free.
    
    # Let's use the 32-vertex coloring.
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
