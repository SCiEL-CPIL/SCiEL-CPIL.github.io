def _has_q3(adj):
    """Fast Q3 check using bipartite structure."""
    # Q3 is bipartite: even-parity vertices {0,3,5,6} and odd-parity {1,2,4,7}
    # In any Q3 embedding, the 8 vertices split into two 4-sets A and B
    # such that the bipartite graph between A and B is exactly Q3.
    #
    # For each 8-subset S, split into A (4 vertices) and B (4 vertices).
    # Check if the bipartite subgraph G[A,B] contains Q3.
    #
    # But we need to try all ways to split S into two 4-sets: C(8,4)/2 = 35 ways.
    # And for each split, check if the 4x4 bipartite adjacency matrix
    # contains Q3 as a subgraph.
    #
    # Q3 as a bipartite graph: it's a 3-regular bipartite graph on 4+4 vertices.
    # There are only a few such graphs. Actually, the only 3-regular bipartite
    # graph on 4+4 vertices is Q3 itself (up to isomorphism).
    #
    # So: G contains Q3 iff there exists an 8-subset S and a partition S=A∪B
    # with |A|=|B|=4 such that the bipartite graph G[A,B] is 3-regular.
    #
    # Wait, is that true? A 3-regular bipartite graph on 4+4 vertices has
    # 4*3 = 12 edges. And Q3 is the only such graph up to isomorphism.
    # Actually, there are multiple 3-regular bipartite graphs on 4+4 vertices.
    # Let me check: the adjacency matrix is 4x4 with row and column sums = 3.
    # Such matrices correspond to 3-regular bipartite graphs.
    # The number is related to the number of 4x4 (0,1)-matrices with row/col sum 3.
    # These are the complements of permutation matrices, so there are 4! = 24 such matrices.
    # But up to isomorphism (permuting rows and columns), how many?
    # Q3 has adjacency matrix (after suitable ordering):
    # 0 1 1 1
    # 1 0 1 1
    # 1 1 0 1
    # 1 1 1 0
    # Wait, that's K4,4 minus a perfect matching. Is that Q3?
    # Q3: vertices 000,001,010,011,100,101,110,111
    # Even: 000,011,101,110 -> 0,3,5,6
    # Odd: 001,010,100,111 -> 1,2,4,7
    # Edges from 0: 1,2,4
    # Edges from 3: 2,1,7
    # Edges from 5: 4,7,1
    # Edges from 6: 7,4,2
    # So the bipartite adjacency (even rows, odd cols, order even:0,3,5,6 odd:1,2,4,7):
    # 0: 1,2,4 -> cols 0,1,2
    # 3: 1,2,7 -> cols 0,1,3
    # 5: 1,4,7 -> cols 0,2,3
    # 6: 2,4,7 -> cols 1,2,3
    # Matrix:
    # 1 1 1 0
    # 1 1 0 1
    # 1 0 1 1
    # 0 1 1 1
    # This is K4,4 minus the matching (0,3),(1,2),(2,1),(3,0).
    # So it IS K4,4 minus a perfect matching. And all 3-regular bipartite graphs on 4+4
    # are isomorphic to this (since they're all K4,4 minus a perfect matching, and all
    # perfect matchings are equivalent under row/col permutations).
    #
    # So: G contains Q3 iff there exist disjoint 4-sets A,B such that
    # the bipartite graph G[A,B] is K4,4 minus a perfect matching.
    # Equivalently: every vertex in A has exactly 3 neighbors in B, and
    # every vertex in B has exactly 3 neighbors in A.
    #
    # This is much faster to check!
    
    from itertools import combinations
    
    for S in combinations(range(16), 8):
        # Try all ways to split S into A (4) and B (4)
        A_candidates = list(combinations(S, 4))
        for A in A_candidates:
            B = tuple(x for x in S if x not in A)
            # Check if G[A,B] is 3-regular bipartite
            ok = True
            for a in A:
                deg = 0
                for b in B:
                    if adj[a] & (1 << b):
                        deg += 1
                if deg != 3:
                    ok = False
                    break
            if not ok:
                continue
            for b in B:
                deg = 0
                for a in A:
                    if adj[a] & (1 << b):
                        deg += 1
                if deg != 3:
                    ok = False
                    break
            if ok:
                return True
    return False
