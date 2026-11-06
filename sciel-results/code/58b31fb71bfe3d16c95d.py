"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    """
    Twin-clique construction: two K_20 cliques on vertices 0..19 and 20..39,
    connected by a perfect matching (i, 20+i) for i in 0..19.
    
    Edge count: C(20,2) * 2 + 20 = 190 + 190 + 20 = 400.
    Score: 400 - 40 = 360.
    
    Cycle construction:
    - For k = 3..20: use k-1 vertices from clique A (0..19) and 1 vertex from clique B.
      Specifically: vertices [0, 1, ..., k-2] from A, and vertex 20 from B.
      Edges: (0,1),(1,2),...,(k-3,k-2) in A; (k-2,20) matching; (20,0) matching.
      Wait, that's only if k-2 >= 2. For k=3: [0,1] from A, [20] from B.
      Edges: (0,1) in A, (1,20) matching, (20,0) matching. ✓
      
    - For k = 21..39: use 2 vertices from A and k-2 vertices from B.
      Vertices: [0, 1] from A, and [21, 22, ..., 20+k-3, 20] from B.
      Edges: (0,1) in A; (1,21) matching; (21,22),...,(20+k-4,20+k-3) in B;
             (20+k-3, 20) in B; (20,0) matching.
      For k=21: B vertices = [21, 20] (since 20+21-3=38, but wait let me recheck).
      Actually for k=21: k-2=19 B vertices. Range 21 to 20+21-3=38, plus 20.
      So B vertices: 21,22,...,38,20. That's 18+1=19. ✓
      
    - For k = 40: Hamiltonian cycle using all vertices.
      We need a cycle that visits all 40 vertices.
      Use: 0, 1, 2, ..., 19, 20, 21, ..., 39, 0.
      Edges: (0,1),...,(18,19) in A; (19,20) matching; (20,21),...,(38,39) in B; (39,0)? 
      But (39,0) is not an edge. 39 is in B, 0 is in A. Matching is (i,20+i), so (19,39) is matching.
      So (39,0) doesn't exist.
      
      Alternative Hamiltonian cycle: 0, 20, 1, 21, 2, 22, ..., 19, 39, 0.
      Edges: (0,20) matching; (20,1) matching (since 1+20=21? No, matching is (i,20+i), so (1,21)).
      Hmm, (20,1): 20 is in B, 1 is in A. Matching pairs are (i, 20+i). So (1,21) is matching, not (1,20).
      So (20,1) is not an edge unless 1=20+i for some i, i.e., i=-19. No.
      
      Let me reconsider. The matching is (i, 20+i) for i=0..19.
      So edges between A and B are: (0,20), (1,21), (2,22), ..., (19,39).
      
      For a Hamiltonian cycle, I need to visit all 40 vertices.
      Try: 0, 1, 2, ..., 19, 20, 21, ..., 39, 0.
      Edges: (0,1)✓, (1,2)✓, ..., (18,19)✓, (19,20)✓ (matching), (20,21)✓, ..., (38,39)✓, (39,0)?
      (39,0): 39 is in B, 0 is in A. Is (0,39) an edge? Matching is (i,20+i), so (19,39) is matching. (0,39) is not.
      So this doesn't work.
      
      Try: 0, 20, 21, 22, ..., 39, 19, 18, ..., 1, 0.
      Edges: (0,20)✓, (20,21)✓, ..., (38,39)✓, (39,19)✓ (matching), (19,18)✓, ..., (2,1)✓, (1,0)✓.
      This works! Length 40. ✓
    """
    edges = set()
    
    # Clique A: vertices 0..19
    for i in range(20):
        for j in range(i+1, 20):
            edges.add((i, j))
    
    # Clique B: vertices 20..39
    for i in range(20, 40):
        for j in range(i+1, 40):
            edges.add((i, j))
    
    # Perfect matching: (i, 20+i) for i in 0..19
    for i in range(20):
        edges.add((i, 20+i))
    
    edges_list = sorted(edges)
    
    cycles = []
    for k in range(3, N+1):
        if k <= 20:
            # k-1 vertices from A: [0, 1, ..., k-2], and 1 vertex from B: [20]
            a_verts = list(range(k-1))  # 0 to k-2
            b_verts = [20]
            cycles.append(a_verts + b_verts)
        elif k < 40:
            # 2 vertices from A: [0, 1], and k-2 vertices from B
            a_verts = [0, 1]
            # B vertices: 21, 22, ..., 20+k-3, 20
            # Count: (20+k-3 - 21 + 1) + 1 = (k-4) + 1 = k-3... wait
            # Let me count: range(21, 20+k-2) gives 21 to 20+k-3, which is (20+k-3)-21+1 = k-3 vertices.
            # Plus 20, total k-2. ✓
            b_verts = list(range(21, 20+k-2)) + [20]
            cycles.append(a_verts + b_verts)
        else:
            # k = 40: Hamiltonian cycle
            # 0, 20, 21, 22, ..., 39, 19, 18, ..., 1, 0
            cyc = [0] + list(range(20, 40)) + list(range(19, 0, -1))
            cycles.append(cyc)
    
    return {'edges': edges_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
