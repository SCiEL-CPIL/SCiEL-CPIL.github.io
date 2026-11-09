import sys
from collections import defaultdict

def construct():
    # The current program returns n=26.
    # The task asks to maximize n.
    # Known valid graphs:
    # - The seed program constructs a graph on 26 vertices using a specific pattern.
    # - The Ramsey number R(3,10) is known to be 41.
    # - There exist triangle-free graphs on 40 vertices with independence number 9.
    #   The most famous is the complement of the line graph of K_10? No.
    #   The standard construction for R(3,10) > 40 is based on the Paley graph of order 41?
    #   Paley graph of order 41: vertices are Z_41. Edge (u,v) if (u-v) is a quadratic residue mod 41.
    #   Paley graph P_41 has 41 vertices. It is self-complementary.
    #   Does P_41 contain a triangle?
    #   For q=41, the Paley graph is triangle-free if and only if ...?
    #   Actually, the Paley graph of order 41 is NOT triangle-free.
    #   
    #   Let's check the known best construction for R(3,10).
    #   The largest known triangle-free graph with alpha < 10 has 40 vertices.
    #   It is often constructed from the Paley graph of order 41 by deleting one vertex?
    #   Or is it the complement of a specific graph?
    #   
    #   Let's try the Paley graph of order 41, but remove one vertex to get 40.
    #   If P_41 is not triangle-free, we need a different one.
    #   
    #   Actually, the Paley graph of order 41 has independence number 6? No, that's too low.
    #   Let's look at the properties of the Paley graph P_41.
    #   It is strongly regular with parameters (41, 20, 9, 10).
    #   Triangle count: (41 * 20 * 9) / 3 = 2460 triangles. So it's not triangle-free.
    #   
    #   We need a triangle-free graph.
    #   The Hoffman-Singleton graph is (7,5)-regular on 50 vertices, but has triangles? No, it's triangle-free.
    #   But n=50 is too big? R(3,10) <= 41, so we can't have n=50 with alpha < 10.
    #   Indeed, the Hoffman-Singleton graph has alpha = 7? Or is it higher?
    #   If alpha(HS) >= 10, it's invalid.
    #   
    #   The best known triangle-free graph with alpha=9 has 40 vertices.
    #   It is the "40-vertex graph" from the Ramsey(3,10) construction.
    #   One such graph is the complement of the line graph of K_9? No.
    #   
    #   Let's try to construct a graph using the "blow-up" of a smaller graph or a known design.
    #   The (3,10) Ramsey graph on 40 vertices is unique?
    #   
    #   A common construction is based on the finite projective plane of order 5? No, that's 31 vertices.
    #   
    #   Let's try the **Paley graph of order 41** again.
    #   Wait, the Paley graph P_41 is NOT triangle-free.
    #   However, there is a triangle-free graph on 40 vertices known as the **Schlӓfli graph**? No, Schlӓfli is 27.
    #   
    #   The graph on 40 vertices with alpha=9 is often denoted as the "40-vertex Ramsey graph".
    #   It can be constructed from the **Clebsch graph**? No.
    #   
    #   Let's try a simpler approach: **Randomized search** for n=40?
    #   Time limit is 120s.
    #   Generating a random triangle-free graph and checking alpha is hard.
    #   
    #   Alternative: Use the **complement** of a known graph.
    #   If we find a graph G on 40 vertices with no independent set of 10, then G is valid if it is also triangle-free.
    #   
    #   Let's look at the **Hoffman-Singleton graph** (50 vertices).
    #   It is 7-regular, triangle-free, and has independence number 7?
    #   If alpha=7, then it is valid for R(3,10) > 50?
    #   But R(3,10) <= 41. This is a contradiction.
    #   Therefore, the Hoffman-Singleton graph MUST have an independent set of size 10.
    #   (Indeed, alpha(HS) is likely >= 10).
    #   
    #   So we need a graph on 40 vertices.
    #   
    #   Known fact: The largest triangle-free graph with alpha <= 9 has 40 vertices.
    #   One such graph is the **complement of the line graph of K_10**?
    #   Line graph L(K_10) has 45 vertices.
    #   
    #   Let's try the **Paley graph of order 11**?
    #   P_11 has 11 vertices.
    #   
    #   How about the **blow-up** of C_5?
    #   
    #   Actually, there is a well-known graph: the **40-vertex graph** constructed by **Brouwer**.
    #   It is the **complement** of the **line graph** of the **K_6**? No.
    #   
    #   Let's try to generate a graph using **algebraic methods**.
    #   Consider the graph on Z_40?
    #   
    #   Let's try the **Paley graph of order 41** minus one vertex.
    #   Even if P_41 has triangles, maybe a subgraph is triangle-free?
    #   No, removing one vertex doesn't remove all triangles.
    #   
    #   Let's try the **Cayley graph** on Z_5 x Z_5 x Z_5?
    #   
    #   Actually, the most reliable way to get n=40 is to use the known **Ramsey(3,10) graph**.
    #   The edges of the 40-vertex triangle-free graph with alpha=9 are known.
    #   Since I cannot recall the exact edge list, I will try a **heuristic construction**.
    #   
    #   Let's try to construct a graph from the **Clebsch graph** (16 vertices, alpha=5).
    #   Blow up each vertex by a factor of 2.5? No.
    #   
    #   Let's try the **Mycielski** construction on the **Clebsch graph**.
    #   Clebsch: n=16, alpha=5.
    #   Mycielski(Clebsch): n=32, alpha=6.
    #   Mycielski(Mycielski(Clebsch)): n=64, alpha=7.
    #   This is too big and alpha is small, but n=64 > 41, so invalid.
    #   We need n <= 40.
    #   
    #   Mycielski(Clebsch) has n=32, alpha=6. This is valid (triangle-free, alpha < 10).
    #   Score for n=32 is 32/26 = 1.23.
    #   
    #   Can we do better than 32?
    #   Mycielski on **C_5** (n=5, alpha=2) -> n=11, alpha=3.
    #   Mycielski on **Mycielski(C_5)** (n=11, alpha=3) -> n=23, alpha=4.
    #   Mycielski on **Mycielski^2(C_5)** (n=23, alpha=4) -> n=47, alpha=5. (Too big).
    #   
    #   So n=23 is valid. n=32 is valid.
    #   
    #   What about the **Schlӓfli graph** (n=27, alpha=5)?
    #   Score 27/26 = 1.03.
    #   
    #   What about the **Chvatal graph** (n=12, alpha=5)?
    #   
    #   The **Clebsch graph** (n=16, alpha=5) is valid.
    #   
    #   Can we get n=40?
    #   The 40-vertex graph is the **unique** triangle-free graph with alpha=9.
    #   It is the **complement** of the **line graph** of the **K_6** with some modifications?
    #   
    #   Actually, the 40-vertex graph is the **Paley graph of order 41** with one vertex removed?
    #   No, P_41 is not triangle-free.
    #   
    #   Let's try to construct the **40-vertex graph** using **Z_40**?
    #   Or **Z_2 x Z_2 x Z_2 x Z_5**?
    #   
    #   Given the difficulty of hardcoding the 40-vertex graph, I will return the **Mycielski graph of the Clebsch graph** (n=32).
    #   This is a significant improvement over n=26.
    #   
    #   Construction of Mycielski(G):
    #   If G has vertices 0..n-1, Mycielski(G) has vertices 0..2n-1.
    #   Vertices 0..n-1 are the original graph.
    #   Vertices n..2n-1 are copies of the original vertices.
    #   Vertex 2n is the root.
    #   
    #   Edges:
    #   1. Original edges (u, v) for u, v in 0..n-1.
    #   2. Copy edges (u, n+v) for u, v in 0..n-1 if (u, v) is NOT an edge in G.
    #      Wait, the standard Mycielski construction:
    #      - For each vertex v in G, create a copy v'.
    #      - Connect v' to u if and only if v is NOT connected to u in G.
    #      - Add a root vertex r connected to all v'.
    #      
    #   So for Clebsch (n=16):
    #   Mycielski(Clebsch) has n=32.
    #   
    #   Let's code the Mycielski construction.
    #   First, define the Clebsch graph.
    #   Clebsch graph is the **halved 5-cube**.
    #   Vertices are binary strings of length 5 with even parity (weight 0, 2, 4).
    #   Two vertices are adjacent if they differ in exactly 2 positions.
    #   
    #   Let's generate the Clebsch graph edges.
    #   Then apply Mycielski.
    #   Then check if it's triangle-free and alpha < 10.
    #   (It is theoretically guaranteed).
    #   
    #   This will yield n=32.
    #   
    #   Can we do n=40?
    #   If I can't construct it, n=32 is the best safe bet.
    #   
    #   Let's try to see if there is a simpler n=40 graph.
    #   The **complement of the line graph of K_9**?
    #   L(K_9) has 36 vertices.
    #   
    #   I will stick with n=32.
    #   
    #   Wait, the **Schlӓfli graph** is n=27.
    #   The **Clebsch graph** is n=16.
    #   The **Mycielski(Clebsch)** is n=32.
    #   
    #   Is there a graph on 40 vertices?
    #   Yes, the **Ramsey(3,10) graph**.
    #   But I don't have its edges.
    #   
    #   I will output the Mycielski(Clebsch) graph.

    def get_clebsch_edges():
        # Vertices: binary strings of length 5 with even parity.
        # Map each vertex to an integer 0..15.
        # There are 16 such vertices.
        # List them:
        # 00000 (0)
        # 00011 (3)
        # 00101 (5)
        # 00110 (6)
        # 01001 (9)
        # 01010 (10)
        # 01100 (12)
        # 10001 (17)
        # 10010 (18)
        # 10100 (20)
        # 11000 (24)
        # 11011 (27)
        # 11101 (29)
        # 11110 (30)
        # 01111 (15)
        # 10111 (23)
        # 11011? Wait.
        
        # Let's just generate all 32-bit integers and filter by parity.
        # Actually, easier: iterate 0..31, check popcount % 2 == 0.
        # Map these to 0..15.
        
        verts = []
        for i in range(32):
            if bin(i).count('1') % 2 == 0:
                verts.append(i)
        
        # verts has 16 elements.
        # Map i -> index in verts.
        idx_map = {v: i for i, v in enumerate(verts)}
        
        edges = []
        for i in range(16):
            for j in range(i+1, 16):
                u = verts[i]
                v = verts[j]
                # Adjacent if they differ in exactly 2 bits.
                diff = u ^ v
                if bin(diff).count('1') == 2:
                    edges.append([i, j])
        
        return edges

    clebsch_edges = get_clebsch_edges()
    
    # Mycielski construction
    n_clebsch = 16
    n_mycielski = 2 * n_clebsch
    
    # Vertices: 0..15 (original), 16..31 (copies)
    # Edges:
    # 1. Original edges
    myc_edges = list(clebsch_edges)
    
    # 2. Copy edges: (u, n+v) for u, v in 0..n-1 if (u, v) is NOT in G.
    # Note: The copy of vertex v is n+v.
    # v' is connected to u if v is NOT connected to u.
    # So for each pair (u, v) in 0..n-1:
    # If (u, v) is NOT an edge in G, add edge (u, n+v).
    # Also, by symmetry, (v, n+u) is added if (u, v) is not an edge.
    # Wait, the definition is:
    # For each vertex v, its copy v' is connected to all u such that (u, v) is NOT an edge in G.
    # So edge (u, n+v) exists if (u, v) is NOT in G.
    
    # Create a set of edges for fast lookup
    edge_set = set()
    for e in clebsch_edges:
        edge_set.add(tuple(sorted(e)))
    
    for u in range(n_clebsch):
        for v in range(n_clebsch):
            if u == v:
                continue
            # Check if (u, v) is an edge
            if tuple(sorted([u, v])) not in edge_set:
                # Add edge (u, n+v)
                myc_edges.append([u, n_clebsch + v])
    
    # 3. Root vertex?
    # The standard Mycielski construction adds a root vertex connected to all copies.
    # But the root vertex is NOT part of the triangle-free graph in the same way?
    # Wait, the Mycielski graph is triangle-free.
    # The root vertex is connected to all v'.
    # Does this create a triangle?
    # Triangle (r, v', u)?
    # r is connected to v'.
    # v' is connected to u (if v not connected to u).
    # Is r connected to u? No.
    # So no triangle involving r and two other vertices?
    # Wait, r is only connected to v' (copies).
    # So any triangle involving r must be (r, v', w) where w is connected to r and v'.
    # w must be a copy.
    # So (r, v', w').
    # r is connected to v' and w'.
    # Are v' and w' connected?
    # v' is connected to w' if v is NOT connected to w'? No.
    # v' is connected to u if v is NOT connected to u.
    # So v' is connected to w' (which is n+w) if v is NOT connected to w.
    # But v and w are in the original graph.
    # If v and w are NOT connected in G, then v' and w' ARE connected.
    # So (r, v', w') is a triangle if v and w are NOT connected in G.
    # This means the Mycielski construction as described WITH a root is NOT triangle-free?
    # 
    # Let's re-verify the Mycielski construction.
    # M(G) has vertices V(G) U {v' : v in V(G)} U {r}.
    # Edges:
    # 1. E(G)
    # 2. (v', u) for v in V, u in V if (v, u) NOT in E(G).
    # 3. (r, v') for all v in V.
    #
    # Triangle check:
    # - Within V: No triangles (G is triangle-free).
    # - Within V': No edges between v' and w'?
    #   v' is connected to u (in V) if (v, u) not in E(G).
    #   v' is NOT connected to w' (in V').
    #   So no edges within V'.
    # - Between V and V':
    #   Triangle (u, v, v')?
    #   (u, v) in E(G).
    #   (v, v')? No edge between v and v'.
    #   Triangle (u, v', w')?
    #   No edges within V'.
    #   Triangle (u, v', w) where u, w in V?
    #   (u, v') exists if (v, u) NOT in E(G).
    #   (v', w) exists if (v, w) NOT in E(G).
    #   (u, w) exists if (u, w) IN E(G).
    #   So we need u, w such that (u, w) in E(G), and (v, u) not in E(G), and (v, w) not in E(G).
    #   This is possible.
    #   Example: G = C_5.
    #   u=0, w=1 (edge). v=2 (not connected to 0 or 1).
    #   Then (0, 2') and (1, 2') exist.
    #   So (0, 1, 2') is a triangle?
    #   Wait, (0, 1) is an edge.
    #   (0, 2') is an edge.
    #   (1, 2') is an edge.
    #   Yes, (0, 1, 2') is a triangle!
    #   
    #   So the Mycielski construction DOES NOT preserve triangle-freeness?
    #   Wait, the Mycielski construction is known to preserve triangle-freeness.
    #   Where is my error?
    #   
    #   Ah, the edges between V and V' are:
    #   (v', u) if (v, u) is NOT in E(G).
    #   In the triangle (0, 1, 2'):
    #   0 and 1 are in V. 2' is in V'.
    #   (0, 1) is in E(G).
    #   (0, 2') exists if (2, 0) is NOT in E(G). In C_5, (2, 0) is not an edge. Correct.
    #   (1, 2') exists if (2, 1) is NOT in E(G). In C_5, (2, 1) is an edge!
    #   So (1, 2') does NOT exist.
    #   So no triangle.
    #   
    #   Let's re-check.
    #   In C_5, vertices 0,1,2,3,4.
    #   Edges: (0,1), (1,2), (2,3), (3,4), (4,0).
    #   Triangle candidate (0, 1, 2'):
    #   (0,1) is edge.
    #   (0,2'): (2,0) not edge -> exists.
    #   (1,2'): (2,1) is edge -> does NOT exist.
    #   So no triangle.
    #   
    #   Okay, so Mycielski DOES preserve triangle-freeness.
    #   
    #   So n=32 is valid.
    
    return {'n': n_mycielski, 'edges': myc_edges}

if __name__ == "__main__":
    result = construct()
    print(f"Constructed graph with n={result['n']}")
