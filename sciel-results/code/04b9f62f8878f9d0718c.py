"""
Pentagon-free graphs without independent 9-sets (Ramsey number R(C5,K9))

Strategy:
- The complement of a C5-free graph is also C5-free (C5 is self-complementary).
- A graph G with no C5 and alpha(G) < 9 has complement H with no C5 and no K9.
- We can construct H as a union of cliques (each clique is C5-free) such that
  no 9 vertices form a clique in H (i.e., no K9 in H).
  
- The best known lower bound construction for R(C5,K9) > 32 uses 8 disjoint K4s
  in the complement H. This gives n=32.
  
- For n=33: we need H on 33 vertices, C5-free, no K9. 
  A union of cliques: if we use cliques of size at most 8, we need at least ceil(33/8)=5 cliques.
  But we also need C5-free. A union of cliques is C5-free iff each clique has size <= 4? No!
  Actually a union of cliques (disjoint) is C5-free if and only if no clique has size >= 5? 
  Wait: a K5 contains a C5. A K6 contains a C5. So any clique of size >= 5 contains a C5.
  So each clique must have size at most 4.
  
  With clique size at most 4, to cover 33 vertices we need at least ceil(33/4) = 9 cliques.
  But then alpha(H) >= 9? No wait - alpha(H) is the independence number of H.
  In H (union of disjoint cliques of size <= 4), an independent set in H picks at most one vertex from each clique.
  So alpha(H) = number of cliques.
  We need alpha(H) < 9, i.e., number of cliques <= 8.
  With 8 cliques of size at most 4, max vertices = 8*4 = 32.
  
  So n=33 is impossible with a union-of-cliques construction for H.
  
  But could there be a non-union-of-cliques H that is C5-free and K9-free on 33 vertices?
  The task says no valid graph has more than 35 vertices, and best known is 32.
  
  So the maximum n we can achieve is 32, using 8 disjoint K4s in H (complement).
  This means G is the complement of 8 disjoint K4s.
  
  Let's construct G directly:
  - G has 32 vertices partitioned into 8 groups of 4.
  - Within each group of 4: no edges in G (since they form a K4 in H, which means independent set in G).
    Wait, let me re-think.
  
  H = complement of G.
  H = 8 disjoint K4s.
  So in H: vertices in the same group of 4 are all connected (clique), vertices in different groups are not connected.
  In G = complement of H: vertices in the same group of 4 have NO edges, vertices in different groups ARE connected.
  
  So G is a complete 8-partite graph with parts of size 4: K_{4,4,4,4,4,4,4,4}.
  
  Check: 
  - C5-free? A C5 in a complete multipartite graph would need 5 vertices from 5 different parts (since no edges within a part). 
    In K_{4,4,...,4} with 8 parts, a 5-cycle would need 5 vertices from 5 distinct parts, all connected. 
    Actually, in a complete multipartite graph, any cycle of length >= 3 exists as long as you have enough parts.
    A C5 needs 5 vertices in 5 distinct parts. Since we have 8 parts, we can pick 5 vertices from 5 different parts, and they form a K5 which contains a C5!
    
    Wait, that's a problem. Let me reconsider.
    
    Actually, in a complete multipartite graph K_{n1,n2,...,nk}, a cycle C_m exists if and only if m <= k (number of parts) and each part has at least 1 vertex... no that's not right either.
    
    Let me think more carefully. In K_{4,4,4,4,4,4,4,4} (8 parts of 4):
    - Pick vertices v1, v2, v3, v4, v5 from 5 different parts.
    - All pairs are connected (different parts).
    - So v1-v2-v3-v4-v5-v1 is a 5-cycle!
    
    So K_{4,4,4,4,4,4,4,4} CONTAINS a C5. This is NOT C5-free!
    
    I made an error. Let me reconsider.
    
    The complement H = 8 disjoint K4s is C5-free (each component is K4, which has no C5).
    But G = complement of H = K_{4,4,4,4,4,4,4,4} which DOES contain C5s.
    
    So this construction does NOT work for G being C5-free!
    
    Let me reconsider the problem. We need G to be C5-free AND alpha(G) < 9.
    
    The seed program uses 7 disjoint K4s for G itself. Each K4 is C5-free, and the disjoint union is C5-free. Alpha = 7 < 9. n=28.
    
    For n=32, we need a C5-free graph on 32 vertices with alpha < 9.
    
    The known construction for R(C5,K9) > 32... let me think about what it could be.
    
    Actually, I recall that the extremal construction often involves Turan-type graphs or specific known examples. 
    
    Let me reconsider: maybe we should NOT use the complement approach for the construction, but rather directly build a C5-free graph.
    
    A C5-free graph can be:
    1. A union of cliques of size at most 4 (each K4 is C5-free, disjoint union is C5-free)
       - Alpha = number of cliques (pick one from each)
       - For alpha < 9: at most 8 cliques
       - Max n = 8 * 4 = 32
       
    So 8 disjoint K4s in G itself! Not in the complement.
    
    Let me verify:
    - G = 8 disjoint K4s
    - C5-free: Yes, each K4 has no C5, and disjoint union has no C5 (a C5 would need to span multiple components, impossible in a disjoint union)
    - Alpha(G): In a disjoint union of cliques, an independent set picks at most one vertex from each clique. So alpha = 8 < 9. ✓
    - n = 32 ✓
    
    This is the correct construction! The seed used 7 K4s (n=28), we can use 8 K4s (n=32).
    
    Now, can we do better than 32? 
    - If we use cliques of size at most 4 (to stay C5-free), max 8 cliques (for alpha < 9), so max n = 32.
    - Could we use a non-clique C5-free component? E.g., a K4 minus an edge? That's still C5-free and has 4 vertices. But it doesn't help increase n since we're limited by the number of components (alpha).
    - What about a component that is C5-free but has more than 4 vertices? 
      - K5 contains C5. K5 minus an edge: does it contain C5? K5-e has 5 vertices. A C5 needs 5 vertices in a cycle. In K5-e, the two non-adjacent vertices... Let's see: K5 minus edge (1,2). Can we find a C5? 1-3-2-4-5-1: edges (1,3)✓, (3,2)✓, (2,4)✓, (4,5)✓, (5,1)✓. Yes! So K5-e contains C5.
      - What about other 5-vertex C5-free graphs? The max edges in a C5-free graph on 5 vertices... Actually, any graph on 5 vertices with 5+ edges contains a C5? No. C5 itself has 5 edges. A graph with 5 vertices and 5 edges that is not C5: e.g., K5 minus 5 edges... 
      
      Actually, the key insight: for a connected C5-free graph on v vertices, what's the max v if we want to keep alpha small?
      
      The simplest approach: stick with disjoint K4s. n=32 is the answer.
"""

def construct():
    # 8 disjoint copies of K4: C5-free, alpha = 8 < 9, n = 32
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
