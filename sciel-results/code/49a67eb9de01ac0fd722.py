import time
import random

_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0  # safety margin before 150s

def _deadline_check():
    return time.time() > _DEADLINE

def _has_k4(adj, n):
    """Check if graph with adjacency matrix adj has a K4."""
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                continue
            for k in range(j+1, n):
                if not adj[i][k] or not adj[j][k]:
                    continue
                for l in range(k+1, n):
                    if adj[i][l] and adj[j][l] and adj[k][l]:
                        return True
    return False

def _has_independent_6(adj, n):
    """Check if graph has an independent set of size 6.
    Independent set of 6 in G = clique of size 6 in complement."""
    # Build complement adjacency
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                # i,j non-adjacent in G, so adjacent in complement
                pass
    # Use Bron-Kerbosch style for clique of size 6 in complement
    # Simpler: check all 6-subsets
    from itertools import combinations
    for subset in combinations(range(n), 6):
        is_independent = True
        for idx in range(6):
            for jdx in range(idx+1, 6):
                u, v = subset[idx], subset[jdx]
                if adj[u][v]:
                    is_independent = False
                    break
            if not is_independent:
                break
        if is_independent:
            return True
    return False

def _validate(adj, n):
    """Check: no K4 and no independent set of 6."""
    if _has_k4(adj, n):
        return False
    if _has_independent_6(adj, n):
        return False
    return True

def _construct_schlaflig_complement():
    """Construct the complement of the Schläfli graph (35 vertices).
    
    The Schläfli graph is an SRG(35, 16, 6, 8) - actually let me reconsider.
    
    The Schläfli graph has parameters (35, 16, 6, 8).
    Its complement has parameters (35, 18, 10, 9).
    
    Wait - the known extremal graph for R(4,6) is a 35-vertex graph that is K4-free
    with independence number 5. Let me use a known construction.
    
    The (3,5,6)-graph or the graph derived from the projective plane of order 5.
    
    Actually, let me use a well-known 35-vertex construction. The graph can be
    constructed from the 5x7 grid with specific connections.
    
    Let me use the construction based on the complement of the (7,3,1) projective plane
    lines, or use a direct known edge list.
    
    I'll use a construction: take vertices as pairs (i,j) where i in {0..4}, j in {0..6}.
    Connect (i,j) to (i',j') if j'-j mod 7 is in some set depending on i-i'.
    """
    n = 35
    # Use the known (35, 18, 9, 10) SRG complement construction
    # Actually let me just use a well-known 35-vertex Ramsey graph.
    # 
    # The graph is constructed from the 5x7 array. Vertex (r,c) for r=0..4, c=0..6.
    # Two vertices (r,c) and (r',c') are adjacent iff (r'-r mod 5, c'-c mod 7) is in a specific set.
    #
    # This is the "5x7 torus" construction. Let me use known parameters.
    
    # Alternative: use a direct known construction.
    # The 35-vertex graph can be described as follows:
    # Partition vertices into 5 groups of 7: V_i = {7*i + j : j in 0..6}
    # Within each group, form a cycle C7 (each vertex adjacent to 2 neighbors mod 7)
    # Between groups V_i and V_j, connect based on (i-j mod 5) and a shift.
    
    # Let me use a simpler known approach: the graph is the complement of a (7,7,7,7) design
    # 
    # Actually, let me just use a known valid 35-vertex graph.
    # The one from Exoo's work. I'll use the construction:
    # Vertices 0..34. Edges defined by a specific pattern.
    
    # Known construction: the (3,5,6)-cage related graph.
    # Let me use the following well-known 35-vertex Ramsey(4,6) graph:
    # It's the graph where vertices are Z_5 x Z_7, and (a,b) ~ (a',b') iff
    # b' - b is in S_{a'-a} where S_0 = {1,2,4,5,6} (not a cycle, a 5-set)
    # Actually this needs care.
    
    # Let me use a verified construction. The graph from the paper "On the Ramsey number R(4,6)"
    # can be constructed as follows:
    
    # Use the incidence graph of the projective plane PG(2,5) restricted... 
    # No, that gives 31 vertices.
    
    # Let me use the 5x7 construction properly:
    # V = Z_5 x Z_7
    # (i,j) ~ (i',j') iff j'-j mod 7 in D_{i'-i mod 5}
    # where D_k for k=0..4 are 5-subsets of Z_7.
    # For the graph to be regular of degree 18: each vertex has 5 neighbors in same row (D_0)
    # and 5 neighbors in each of 4 other rows (D_1, D_2, D_3, D_4).
    # Total degree = 5*4 + 5 = 25? No, that's too many.
    # Wait: 5 neighbors in same row + 5 in each of 4 other rows = 5 + 20 = 25. Too high.
    # We need degree 18. So maybe 3 in same row and 3 in each other row: 3+12=15. Not 18.
    # Or 4 in same row + 4 in each other row = 4+16=20. Close but not 18.
    # Hmm, 18 = 5*3 + 3? No. 18/5 is not integer. So it's not a circulant on Z_5 x Z_7.
    
    # Let me just hardcode a known valid 35-vertex graph.
    # I'll use the one that's been verified in the literature.
    
    # Actually, let me try a different approach: use the complement of the (35,16,6,8) SRG.
    # The Schläfli graph S(35) is an SRG(35, 16, 6, 8).
    # Its complement is SRG(35, 18, 10, 9).
    # The complement is K4-free? Let's check: a K4 in complement = independent set of 4 in original.
    # In Schläfli graph, can we find 4 mutually non-adjacent vertices? 
    # Independence number of Schläfli graph: alpha(S) = 5? If so, complement has no independent 6...
    # Wait, we need: complement is K4-free AND complement has no independent 6.
    # Complement K4-free means: no 4 vertices all non-adjacent in original, i.e., alpha(original) <= 3.
    # Complement has no independent 6 means: no 6 vertices all non-adjacent in complement,
    # i.e., no 6 vertices all adjacent in original, i.e., omega(original) <= 5.
    # Schläfli graph: alpha = 5, omega = 5? Let me check.
    # Actually for SRG(35,16,6,8): 
    #   eigenvalues: 16, 4 (mult 7), -3 (mult 27)? Let me verify.
    #   16 + 7*4 + 27*(-3) = 16+28-81 = -37. Not zero. Wrong.
    #   Actually: r=16, s=-3? 16+7*4+27*(-3) ≠ 0.
    #   For SRG(35,16,6,8): (v-1-k) = 18, k(k-2λ-mu) = 16*12=192, (v-1-k)(k-mu)=18*8=144. 192≠144. 
    #   Hmm, let me recheck. SRG(v,k,λ,μ): (v-1-k)μ = k(k-2λ-μ)? No.
    #   The equation is: (v-1-k)μ = k(k-2λ-μ).
    #   (35-1-16)*8 = 18*8 = 144. k(k-2λ-μ) = 16*(16-12-8) = 16*(-4) = -64. Not equal.
    #   So (35,16,6,8) is NOT a valid SRG parameter set!
    
    # Let me look up the actual Schläfli graph parameters.
    # The Schläfli graph is SRG(35, 16, 6, 8)? Let me verify with the equation.
    # Actually the correct check: for non-neighbors x,y, common neighbors = μ.
    # For neighbors x,y, common neighbors = λ.
    # The parameter condition: (v-1-k)μ = k(k-2λ-μ)
    # 18*8 = 16*(16-12-8) → 144 = 16*(-4) → 144 = -64. FALSE.
    # So (35,16,6,8) is not valid.
    
    # The Schläfli graph is actually SRG(35, 16, 6, 8)? Let me look at this differently.
    # Maybe I have the parameters wrong. The Schläfli graph:
    # - 35 vertices, degree 16
    # - Each pair of adjacent vertices has 6 common neighbors
    # - Each pair of non-adjacent vertices has 8 common neighbors
    # Let me re-verify: (v-1-k)μ = k(k-2λ-μ)
    # v=35, k=16, λ=6, μ=8
    # LHS: (35-1-16)*8 = 18*8 = 144
    # RHS: 16*(16-12-8) = 16*(-4) = -64
    # These don't match, so these can't be the correct parameters.
    
    # Let me reconsider. Perhaps the Schläfli graph has different parameters.
    # Looking it up mentally: the Schläfli graph has 35 vertices, is regular of degree 16.
    # Hmm, maybe λ=4, μ=6?
    # (18)*6 = 16*(16-8-6) = 16*2 = 32. 108≠32. No.
    # λ=2, μ=4? 18*4=72, 16*(16-4-4)=16*8=128. No.
    # 
    # I think I'm confusing graphs. Let me just use a known valid 35-vertex construction
    # from a different source.
    
    # The 35-vertex Ramsey(4,6) graph can be constructed from the 5x7 grid as follows:
    # Use the "Hoffman-Singleton type" or just use a known edge list.
    
    # Let me use a constructive approach: 
    # The graph on 35 vertices where V = {0,...,34}, and we define edges via a
    # specific pattern that's known to work.
    
    # Known construction from Radziszowski's survey:
    # The (3,5,6)-graph: take the 35 vertices as all 2-subsets of a 7-element set? No, that's 21.
    # 
    # Let me try: vertices = Z_5 x Z_7, edges: (i,a)~(j,b) iff (i-j, a-b) in a specific set.
    # For degree 18: we need |S| * 5 / 35 * ... this is getting complicated.
    
    # Simplest: use a known valid edge list for the 35-vertex graph.
    # I'll generate it using a known algebraic construction.
    
    # The graph is the "5x7 torus" with specific connection rules.
    # After research, the correct construction uses:
    # D_0 = {1, 2, 4, 5, 6} (5 elements) - but this gives degree issues.
    
    # Let me just use a brute-force verified construction:
    # I'll define the graph using the following known 35-vertex Ramsey(4,6) graph
    # from the literature (the one found by Exoo and others).
    
    # For now, let me use a simpler but valid smaller graph and also try to
    # construct the 35-vertex one.
    
    # Actually, let me use the Paley graph on 49 vertices? No, that might have K4.
    # Paley(49): 49 vertices, degree 24. Does it contain K4? Likely yes.
    
    # Let me use a known valid 31-vertex graph: the complement of the (31,15,6,8) SRG?
    # No, that's the complement of the Petersen-type graph on 31... 
    
    # The (3,5,6)-cage is a graph, not directly what we need.
    
    # OK let me just use a well-known valid construction. The 35-vertex graph
    # from the paper by Exoo (2017) can be described as follows:
    # 
    # Take 5 copies of a 7-cycle. Within each copy, connect vertices at distance 1 and 2 on the cycle
    # (so each vertex has 4 neighbors within its copy, making it a C7^2 or the square of C7).
    # Between copies i and j, connect vertex a in copy i to vertex b in copy j iff
    # (b-a) mod 7 is in a specific set depending on (j-i) mod 5.
    #
    # For degree: 4 (within) + 4*? (between) = 18, so 4 between each pair? No, 4+4x=18, x=3.5. Not integer.
    # 3 within + 3*4=12 between = 15. Not 18.
    # 5 within (C7^2 has degree 4, not 5)... 
    #
    # I think the cleanest approach is to use a verified edge list. Let me construct
    # the graph using a known algebraic method.
    
    # The 35-vertex graph: vertices are (i, j) with i in Z_5, j in Z_7.
    # (i,j) ~ (i',j') iff (i'-i, j'-j) in S where S is a subset of Z_5 x Z_7 with specific properties.
    # For the graph to be undirected and regular: S = -S (mod (5,7)), 0 not in S, |S|=18.
    #
    # Known S for the Ramsey(4,6) graph:
    # S = {(0,1),(0,2),(0,4),(0,5),(0,6),  (1,1),(1,3),(1,5),(2,2),(2,4),(2,6),
    #      (3,1),(3,3),(3,5),(4,2),(4,4),(4,6),  ... }
    # This needs |S| = 18 and S = -S.
    
    # Let me try: S consists of all (a,b) where a in {1,2,3,4} and b in {1,2,3,4,5,6} 
    # satisfying some condition, plus (0, b) for b in {1,2,4,5,6}.
    # That's 5 + 4*? We need 18 total. 5 from a=0, so 13 from a in {1,2,3,4}.
    # 13/4 is not integer. Hmm.
    
    # Let me try: 4 from a=0 (b in {1,2,4,5}), and 14 from a in {1,2,3,4} (3 or 4 each).
    # 4 + 4*3.5... still not clean.
    
    # I'll use a different approach: construct the graph and verify.
    # Let me use the following known construction:
    
    # The graph is the "Higman-Sims" type or use the 5x7 with:
    # Same row: connect if difference in {1,2,3,4,5,6} minus one... 
    
    # You know what, let me just use a known valid smaller graph (like 31 or 29 vertices)
    # and also try to get 35. For the 35-vertex graph, I'll use a specific known edge list.
    
    # After more thought, the correct 35-vertex Ramsey(4,6) graph is constructed as follows:
    # It's the complement of the (7,3,1)-BIBD design graph, or equivalently:
    # 
    # Vertices: all points of PG(2, 5) would be 31, not 35.
    # 
    # Let me use the following: the 35-vertex graph from the "5x7" construction where
    # the difference set is carefully chosen. I'll generate it and verify.
    
    # Construction: V = Z_5 x Z_7
    # Adjacency: (i,j) ~ (i',j') iff j'-j mod 7 in A_{i'-i mod 5}
    # where A_k is a subset of Z_7 for k = 0,1,2,3,4.
    # For regularity: |A_0| = |A_1| = |A_2| = |A_3| = |A_4| = d (degree = 5d)
    # 5d = 18 → d = 3.6. Not integer!
    
    # So it's NOT a circulant of this form. The graph must have a different structure.
    
    # Let me try: the graph is not vertex-transitive. It has 5 groups of 7, but the
    # connections between groups vary in a non-circulant way.
    
    # OK, I'll use a completely different approach: use a known valid 35-vertex graph
    # edge list. I'll encode it directly.
    
    # The 35-vertex Ramsey(4,6) graph edge list (0-indexed):
    # I'll use the construction from the paper "New lower bounds for Ramsey numbers" by Exoo.
    # 
    # For now, let me use a verified smaller graph and add the 35-vertex one if I can
    # construct it correctly.
    
    # Let me use the Paley graph on 25 vertices? Paley(25): 25 vertices, degree 12.
    # Does it have K4? The Paley graph on p^2 where p is prime: if p ≡ 3 mod 4, then...
    # 25 = 5^2. 5 ≡ 1 mod 4, so Paley(25) is defined. Degree = (25-1)/2 = 12.
    # Independence number of Paley(25): at most sqrt(25) = 5? Actually by spectral methods,
    # alpha ≤ n * (1 + sqrt(1+4n))/... For Paley(q) with q≡1 mod 4: alpha ≤ (1+sqrt(q))/2 * ... 
    # Actually alpha(Paley(25)) ≤ 5 (by the bound alpha ≤ n(1+sqrt(1+4(n-1)))/(4(n-1))... 
    # Let me just check: Paley(25) has no K4? If the smallest clique is 4, then it has K4.
    # For Paley(25), the clique number is 5 (I believe), so it HAS K4. Not valid.
    
    # What about the complement of Paley(25)? Same issue (K4 in one = independent 4 in other).
    
    # Let me use a 23-vertex graph or construct one.
    
    # Actually, for a strong score, I need as many vertices as possible.
    # Let me try to construct the 35-vertex graph using a known method.
    
    # The 35-vertex graph can be constructed from the (7,7) Latin square or from
    # the 5x7 grid with the following rule (from the original Ramsey(4,6) paper):
    # 
    # V = {0,...,34}. Write v = 7i + j where i in {0,...,4}, j in {0,...,6}.
    # Edges within a row i: (7i+j) ~ (7i+j') iff |j-j'| in {1, 2, 5, 6} mod 7 (i.e., not 0,3,4)
    #   Wait, that gives degree 4 within row.
    # Edges between rows i and i' (i≠i'): (7i+j) ~ (7i'+j') iff (j'-j) mod 7 in B_{i'-i}
    #   where B_k depends on the difference.
    # For total degree 18: 4 (within) + sum of |B_k| for k=1,2,3,4 = 18, so sum = 14.
    # If each |B_k| = 3 or 4: 3+3+4+4=14. 
    #
    # Let me try: B_1 = {0,1,3}, B_2 = {0,2,5}, B_3 = {1,3,4,6}, B_4 = {0,2,4,6}
    # Check: |B_1|=3, |B_2|=3, |B_3|=4, |B_4|=4. Sum=14. Total degree=4+14=18. Good.
    # But I need to verify no K4 and no independent 6.
    
    # This is getting too uncertain. Let me use a different strategy:
    # Use a known valid graph and verify it programmatically.
    
    # For now, I'll use the following approach:
    # 1. Try to construct the 35-vertex graph using a known edge list.
    # 2. Verify it.
    # 3. If it fails, fall back to a smaller verified graph.
    
    # Let me use a known 35-vertex graph. The one from the "Small Ramsey Numbers" survey:
    # I'll use the following construction based on the 5x7 grid:
    
    n = 35
    adj = [[0]*n for _ in range(n)]
    
    # Row-internal edges: connect vertices in same row if their column difference is in {1,2,3,4,5,6}
    # minus {0,3,4} → {1,2,5,6}. So 4 neighbors per vertex within row.
    for i in range(5):
        for j in range(7):
            for dj in [1, 2, 5, 6]:
                j2 = (j + dj) % 7
                u = 7*i + j
                v = 7*i + j2
                if u < v:
                    adj[u][v] = 1
                    adj[v][u] = 1
    
    # Between-row edges: for row difference d (mod 5), use a specific set of column shifts.
    # B_1 = {0, 1, 3} (3 connections)
    # B_2 = {0, 2, 5} (3 connections)  
    # B_3 = {1, 3, 4, 6} (4 connections)
    # B_4 = {0, 2, 4, 6} (4 connections)
    B = {
        1: {0, 1, 3},
        2: {0, 2, 5},
        3: {1, 3, 4, 6},
        4: {0, 2, 4, 6}
    }
    
    for i in range(5):
        for i2 in range(5):
            if i == i2:
                continue
            d = (i2 - i) % 5
            if d == 0:
                continue
            # Use B_d or B_{5-d}
            shift_set = B.get(d, B.get(5-d, set()))
            for j in range(7):
                for shift in shift_set:
                    j2 = (j + shift) % 7
                    u = 7*i + j
                    v = 7*i2 + j2
                    if u < v:
                        adj[u][v] = 1
                        adj[v][u] = 1
    
    # Verify
    if _validate(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # If 35 fails, try smaller graphs
    # Use Paley(17) as fallback
    return None

def construct():
    """Construct a K4-free graph with no independent set of 6, maximizing n."""
    
    # Try the 35-vertex construction first
    result = _construct_schlaflig_complement()
    if result is not None:
        return result
    
    # Fallback: try various constructions
    # Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [[0]*17 for _ in range(17)]
    for i in range(17):
        for j in range(i+1, 17):
            if (j - i) % 17 in squares:
                adj[i][j] = 1
                adj[j][i] = 1
    
    if _validate(adj, 17):
        edges = []
        for i in range(17):
            for j in range(i+1, 17):
                if adj[i][j]:
                    edges.append([i, j])
        return {'n': 17, 'edges': edges}
    
    # If all else fails, return a trivial valid graph
    return {'n': 1, 'edges': []}
