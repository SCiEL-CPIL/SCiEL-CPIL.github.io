"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_N = 16
_E = _N * (_N - 1) // 2  # 120 edges
_DEADLINE_MARGIN = 20  # seconds to spare before the 150s kill

# Precompute all edges
_ALL_EDGES = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
_EDGE_INDEX = {}
for i, (u, v) in enumerate(_ALL_EDGES):
    _EDGE_INDEX[(u, v)] = i
    _EDGE_INDEX[(v, u)] = i

# Precompute Q3 structure
_Q3_EDGES = []
for i in range(8):
    for j in range(i + 1, 8):
        if (i ^ j) in (1, 2, 4):
            _Q3_EDGES.append((i, j))

# Build conflict graph: two potential edges conflict if they can be part of the same Q3
# Two edges e1=(a,b) and e2=(c,d) conflict if there exists a Q3 embedding where
# both e1 and e2 appear as edges of the Q3.
# 
# For the conflict graph, we need: for each pair of edges, do they jointly appear
# in some Q3 subgraph? This is equivalent to: can we embed Q3 such that both
# graph-edges correspond to Q3-edges?
#
# A simpler approach: build the conflict graph by checking all Q3 embeddings.
# For each Q3 embedding (a bijection from {0..7} to a subset of {0..15}), 
# mark all pairs of Q3-edges as conflicting.
#
# But there are too many embeddings. Instead, use a smarter approach:
# Two edges (a,b) and (c,d) conflict iff they can coexist in a Q3.
# This happens iff there's a way to assign them to Q3 edges such that
# the rest of the Q3 can be completed.
#
# For our purposes, we'll build the conflict graph by:
# 1. For each pair of edges that share a vertex, check if they can be in a Q3
# 2. For disjoint edges, check similarly
#
# Actually, let's use a different approach: build the conflict graph by
# enumerating all ways two edges can appear in a Q3.
#
# In Q3, any two edges either:
# - Share a vertex (adjacent edges in Q3)
# - Are disjoint (opposite edges in a 4-cycle, or in different 4-cycles)
#
# Let's build the conflict graph more directly.
# Two graph edges e1 and e2 conflict if there exists a Q3 embedding containing both.
# We'll compute this by checking all possible Q3 edge-pair assignments.

def _build_conflict_graph():
    """Build conflict graph where node i represents edge _ALL_EDGES[i].
    Two nodes are connected if the two edges cannot both be in a Q3-free graph
    (i.e., they conflict - adding both would create a Q3 with some other edges).
    
    Actually, for maximum independent set formulation:
    - Nodes = potential edges (120 nodes)
    - Two nodes are connected if the two edges CANNOT both be selected in a Q3-free graph
    - We want maximum independent set = maximum set of edges with no conflicts
    
    Two edges conflict if they MUST appear together in some Q3... no that's wrong.
    Two edges conflict if selecting BOTH of them forces a Q3 to exist.
    But that's not quite right either - it depends on other edges too.
    
    The correct formulation: we want the largest set S of edges such that the
    graph (V, S) contains no Q3. This is NOT simply an independent set problem
    because Q3-free is not a pairwise property.
    
    However, we can use a relaxation: if two edges e1, e2 are such that ANY
    Q3 containing both e1 and e2 can be "completed" (i.e., there always exists
    a completion), then they conflict. But this is hard to determine.
    
    Alternative: use the conflict graph where e1 conflicts with e2 if they
    are edges in the SAME Q3 in some embedding. Then any independent set in
    this conflict graph is Q3-free (because it contains no two edges from the
    same Q3 embedding). This is a valid but potentially loose constraint.
    
    Let's build this: e1 and e2 conflict if there exists a Q3 embedding where
    both e1 and e2 are edges of the Q3.
    """
    n = _E
    # Use bitsets for the conflict graph
    conflicts = [0] * n  # conflicts[i] is a bitmask of nodes conflicting with i
    
    # For each pair of Q3 edges, determine which graph-edge pairs conflict
    # Q3 has 12 edges. For each pair of Q3 edges, we need to find all
    # graph-edge pairs that can realize them.
    
    # Q3 edges as (i,j) pairs
    q3_e = _Q3_EDGES  # 12 edges
    
    # For each pair of Q3 edges (e_a, e_b), find all graph realizations
    # A realization maps Q3 vertices to graph vertices
    # e_a = (a1, a2) in Q3, e_b = (b1, b2) in Q3
    # We need: graph edge (f(a1), f(a2)) and graph edge (f(b1), f(b2))
    # where f is an injective map from {0..7} to {0..15}
    
    # This is complex. Let's use a different approach:
    # For each pair of graph edges, check if they can be in the same Q3.
    
    # Two edges (u1,v1) and (u2,v2) can be in the same Q3 iff:
    # Case 1: They share a vertex. Say u1 = u2. Then we need to find
    #         a Q3 where two edges share vertex u1, and the other endpoints
    #         map to v1 and v2. In Q3, vertex 000 has edges to 001, 010, 100.
    #         So v1 and v2 must be at "distance 1" in the Q3 from the shared vertex.
    #         The shared vertex in Q3 can be any of the 8 vertices.
    #         From a Q3 vertex, its 3 neighbors are at specific positions.
    #         Two edges sharing a vertex in Q3: the two other endpoints must
    #         be at Hamming distance 2 from each other (they differ in 2 bits
    #         if the shared vertex is the LCA).
    #         Actually in Q3, two neighbors of the same vertex are at distance 2.
    #         So v1 and v2 must be at distance 2 in the graph... no, that's
    #         a property of the embedding, not the graph.
    #
    # Let me think differently. The conflict graph for the MIS approach:
    # e1 and e2 are in conflict if they CANNOT both be in a Q3-free graph.
    # The simplest sufficient condition: e1 and e2 are edges in some Q3 embedding
    # AND that Q3 embedding is "forced" - but that's not right.
    #
    # Let's use the standard approach: build the conflict graph where e1~e2
    # if there's a Q3 embedding where both are edges. Then MIS in this graph
    # gives a Q3-free graph (no two edges from any Q3 are both selected).
    # This is a valid upper bound on the answer.
    
    # For efficiency, let's enumerate Q3 embeddings partially.
    # A Q3 embedding is determined by where we place vertices 0 and 1 (adjacent),
    # then the rest is constrained.
    
    # For each ordered pair of distinct graph vertices (a, b) as Q3 vertices 0 and 1:
    # Q3 vertex 0 is at 'a', Q3 vertex 1 is at 'b'
    # Q3 vertex 2 (adjacent to 0): must be a neighbor of 'a' in the graph, != b
    # Q3 vertex 4 (adjacent to 0): must be a neighbor of 'a', != b, != vertex2
    # Q3 vertex 3 (adjacent to 1 and 2): must be common neighbor of b and vertex2
    # Q3 vertex 5 (adjacent to 1 and 4): must be common neighbor of b and vertex4
    # Q3 vertex 6 (adjacent to 2 and 4): must be common neighbor of vertex2 and vertex4
    # Q3 vertex 7 (adjacent to 3, 5, 6): must be common neighbor of 3, 5, and 6
    
    # For the conflict graph, we just need to know which pairs of edges appear
    # in some Q3 embedding. Let's enumerate all Q3 embeddings and mark conflicts.
    
    # But there could be many embeddings. Let's be smart:
    # For each pair of edges (e1, e2), check if they can coexist in a Q3.
    # This is O(E^2) pairs * O(embedding check) which might be too slow.
    
    # Alternative: enumerate all Q3 embeddings by fixing edge (0,1) and
    # backtracking. For each valid embedding, mark all 12 edges as mutually
    # conflicting.
    
    # Let's build adjacency first (complete graph for enumeration)
    adj = [0] * _N
    for i in range(_N):
        for j in range(i + 1, _N):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    
    # Enumerate Q3 embeddings
    # Fix Q3 vertex 0 at graph vertex a, Q3 vertex 1 at graph vertex b (a != b)
    # Then backtrack on vertices 2, 4, 3, 5, 6, 7
    
    _FIXED_EDGE_ORDER = [2, 4, 3, 5, 6, 7]
    _FIXED_CONSTRAINTS = {
        2: ([], 0, []),   # must be adjacent to Q3 vertex 0
        4: ([], 0, []),   # must be adjacent to Q3 vertex 0
        3: ([2], 1, []),  # must be adjacent to Q3 vertices 2 and 1
        5: ([4], 1, []),  # must be adjacent to Q3 vertices 4 and 1
        6: ([2, 4], -1, []),  # must be adjacent to Q3 vertices 2 and 4
        7: ([3, 5], -1, []),  # must be adjacent to Q3 vertices 3 and 5 (and 6)
    }
    # Additional constraints for 7: must also be adjacent to 6
    # And for 3: adjacent to 1 and 2
    # For 5: adjacent to 1 and 4
    
    # Let's just do full enumeration with backtracking
    def enumerate_q3_embeddings():
        """Yield all Q3 embeddings as list of 8 graph vertex indices."""
        for a in range(_N):
            for b in range(_N):
                if a == b:
                    continue
                # Q3 vertex 0 -> a, Q3 vertex 1 -> b
                # Must be adjacent in graph
                if not (adj[a] >> b) & 1:
                    continue
                
                assigned = [-1] * 8
                assigned[0] = a
                assigned[1] = b
                used = (1 << a) | (1 << b)
                
                def bt(idx):
                    if idx == 6:
                        # Check all Q3 edges among vertices 2-7
                        for i in range(2, 8):
                            for j in range(i + 1, 8):
                                if (i ^ j) in (1, 2, 4):
                                    if not (adj[assigned[i]] >> assigned[j]) & 1:
                                        return
                        yield assigned[:]
                        return
                    qv = _FIXED_EDGE_ORDER[idx]
                    # Determine constraints
                    if qv == 2:
                        # Adjacent to Q3 vertex 0 (graph vertex a)
                        cands = adj[a] & ~used
                    elif qv == 4:
                        # Adjacent to Q3 vertex 0 (graph vertex a)
                        cands = adj[a] & ~used
                    elif qv == 3:
                        # Adjacent to Q3 vertices 1 (b) and 2
                        cands = adj[b] & adj[assigned[2]] & ~used
                    elif qv == 5:
                        # Adjacent to Q3 vertices 1 (b) and 4
                        cands = adj[b] & adj[assigned[4]] & ~used
                    elif qv == 6:
                        # Adjacent to Q3 vertices 2 and 4
                        cands = adj[assigned[2]] & adj[assigned[4]] & ~used
                    elif qv == 7:
                        # Adjacent to Q3 vertices 3, 5, and 6
                        cands = adj[assigned[3]] & adj[assigned[5]] & adj[assigned[6]] & ~used
                    else:
                        cands = 0
                    
                    c = cands
                    while c:
                        lsb = c & (-c)
                        gv = lsb.bit_length() - 1
                        c ^= lsb
                        assigned[qv] = gv
                        used_old = used
                        used |= (1 << gv)
                        yield from bt(idx + 1)
                        used = used_old
                        assigned[qv] = -1
                
                yield from bt(0)
    
    # For each Q3 embedding, mark all 12 edges as mutually conflicting
    # This could be slow if there are many embeddings. Let's limit it.
    # Actually, for the complete graph on 16 vertices, there are many Q3 embeddings.
    # Let's count: choose 8 vertices from 16, then count Q3 embeddings on them.
    # C(16,8) * 8! / 8 (automorphisms) = 12870 * 8! / 8 = 12870 * 80640 = ~1 billion.
    # Too many! We need a smarter approach.
    
    # Better approach: for each pair of graph edges, directly check if they
    # can be in the same Q3. Two edges (a,b) and (c,d) can be in a Q3 iff
    # there exists a Q3 embedding containing both.
    
    # Case analysis:
    # 1. Edges share a vertex: (a,b) and (a,c) [b != c]
    #    In Q3, two edges sharing vertex v: the other endpoints are at distance 2.
    #    The shared vertex in Q3 has degree 3. The two other endpoints must be
    #    two of its three neighbors. So we need: in Q3, vertices b' and c' are
    #    both neighbors of a', and b' != c'. Then we need to complete the Q3.
    #    For this to work, we need a common structure. Let's check:
    #    Place a at Q3 vertex 0. Then b and c are at Q3 vertices from {1,2,4}.
    #    Without loss, b at 1, c at 2. Then we need:
    #      Q3 vertex 3 (adj to 1,2): common neighbor of b,c in graph
    #      Q3 vertex 4 (adj to 0): neighbor of a, != b,c
    #      Q3 vertex 5 (adj to 1,4): common neighbor of b, vertex4
    #      Q3 vertex 6 (adj to 2,4): common neighbor of c, vertex4
    #      Q3 vertex 7 (adj to 3,5,6): common neighbor of 3,5,6
    #    So the check is: exists vertex4 (neighbor of a, not b,c) such that
    #    exists v3 (common neighbor of b,c), v5 (common neighbor of b,vertex4),
    #    v6 (common neighbor of c,vertex4), v7 (common neighbor of v3,v5,v6),
    #    all distinct and not in {a,b,c}.
    
    # 2. Edges are disjoint: (a,b) and (c,d) [all distinct]
    #    In Q3, two disjoint edges: they can be opposite in a 4-cycle, or in
    #    different parts of the cube.
    #    If opposite in a 4-cycle: a,b,c,d form a 4-cycle in Q3.
    #    In Q3, a 4-cycle is like 000-001-011-010-000. The opposite edges are
    #    (000,001) and (011,010), or (000,010) and (001,011).
    #    So a and c are at distance 2, b and d are at distance 2 (or similar).
    #    Actually in the 4-cycle 0-1-3-2-0: edges (0,1) and (2,3) are opposite.
    #    Distances: d(0,2)=2, d(1,3)=2.
    #    So for disjoint edges (a,b) and (c,d) to be opposite in a 4-cycle:
    #    we need d(a,c)=2 and d(b,d)=2, or d(a,d)=2 and d(b,c)=2.
    #    But "distance" here is in the graph, not in Q3. The embedding just
    #    needs to exist.
    
    # This is getting complex. Let's use a practical approach:
    # Build the conflict graph by checking, for each pair of edges, whether
    # they can coexist in a Q3, using a fast backtracking check.
    
    # For each pair (i, j) of edges, check if there's a Q3 containing both.
    # We'll use the fixed-edge backtracking but adapted for two fixed edges.
    
    # Actually, let's take a step back. The MIS formulation with this conflict
    # graph may not be tight enough. Let me instead use a hybrid approach:
    # 1. Build a "weak" conflict graph (edges that are in the same Q3 in some
    #    embedding) for a quick upper bound
    # 2. Use branch-and-bound with the actual Q3 check for pruning
    
    # For the weak conflict graph, let's enumerate Q3 embeddings but limit
    # the work. We'll sample embeddings.
    
    # Actually, let's just build the full conflict graph by checking each pair.
    # There are C(120,2) = 7140 pairs. For each, we check if they can be in
    # a Q3. The check involves backtracking over ~6 variables with small
    # branching factor. Should be fast enough.
    
    def _can_be_in_q3(e1, e2, adj):
        """Check if edges e1=(a,b) and e2=(c,d) can both be in some Q3 embedding."""
        a, b = e1
        c, d = e2
        
        # Case 1: edges share a vertex
        if a == c or a == d or b == c or b == d:
            # Find the shared vertex
            if a == c:
                shared = a
                other1 = b
                other2 = d
            elif a == d:
                shared = a
                other1 = b
                other2 = c
            elif b == c:
                shared = b
                other1 = a
                other2 = d
            else:  # b == d
                shared = b
                other1 = a
                other2 = c
            
            # In Q3, place shared at vertex 0.
            # other1 at vertex 1, other2 at vertex 2 (WLOG, since we can swap)
            # Actually, other1 and other2 are both neighbors of shared in Q3.
            # In Q3, vertex 0's neighbors are 1, 2, 4.
            # So other1 and other2 map to two of {1, 2, 4}.
            # WLOG: other1 -> 1, other2 -> 2.
            # Then we need:
            #   v3: common neighbor of other1(1) and other2(2) in graph
            #   v4: neighbor of shared(0), not other1, other2
            #   v5: common neighbor of other1(1) and v4
            #   v6: common neighbor of other2(2) and v4
            #   v7: common neighbor of v3, v5, v6
            # All distinct, not in {shared, other1, other2}
            
            used_base = (1 << shared) | (1 << other1) | (1 << other2)
            
            # v3: common neighbor of other1 and other2
            cands_v3 = adj[other1] & adj[other2] & ~used_base
            if cands_v3 == 0:
                return False
            
            # v4: neighbor of shared, not other1, other2, not v3
            cands_v4_base = adj[shared] & ~used_base
            
            c3 = cands_v3
            while c3:
                lsb3 = c3 & (-c3)
                v3 = lsb3.bit_length() - 1
                c3 ^= lsb3
                
                c4 = cands_v4_base & ~(1 << v3)
                while c4:
                    lsb4 = c4 & (-c4)
                    v4 = lsb4.bit_length() - 1
                    c4 ^= lsb4
                    
                    used2 = used_base | (1 << v3) | (1 << v4)
                    
                    # v5: common neighbor of other1 and v4
                    cands_v5 = adj[other1] & adj[v4] & ~used2
                    if cands_v5 == 0:
                        continue
                    
                    c5 = cands_v5
                    while c5:
                        lsb5 = c5 & (-c5)
                        v5 = lsb5.bit_length() - 1
                        c5 ^= lsb5
                        
                        used3 = used2 | (1 << v5)
                        
                        # v6: common neighbor of other2 and v4
                        cands_v6 = adj[other2] & adj[v4] & ~used3
                        if cands_v6 == 0:
                            continue
                        
                        c6 = cands_v6
                        while c6:
                            lsb6 = c6 & (-c6)
                            v6 = lsb6.bit_length() - 1
                            c6 ^= lsb6
                            
                            used4 = used3 | (1 << v6)
                            
                            # v7: common neighbor of v3, v5, v6
                            cands_v7 = adj[v3] & adj[v5] & adj[v6] & ~used4
                            if cands_v7 != 0:
                                return True
                        
                        # Also check: v5 and v6 could be same? No, they must be distinct.
                        # Actually v5 is neighbor of other1 and v4, v6 is neighbor of other2 and v4.
                        # They could potentially be the same vertex? No, in Q3, 5 != 6.
                        # But in the graph, they could map to the same vertex? No, embedding is injective.
                    
                    # Try the other assignment: other1 -> 2, other2 -> 1
                    # This is symmetric, so if the above didn't find anything,
                    # swapping won't help since the structure is the same.
                    # Actually it's different because v3 is common neighbor of 1 and 2,
                    # which is symmetric. And v5 is neighbor of 1 and v4, v6 is neighbor of 2 and v4.
                    # Swapping 1<->2 swaps v5<->v6, which is symmetric. So no need.
            
            return False
        
        # Case 2: edges are disjoint (a,b) and (c,d), all four distinct
        # In Q3, two disjoint edges can be:
        # (a) Opposite edges in a 4-cycle
        # (b) In "opposite" 4-cycles (e.g., edge (0,1) and edge (4,5) in Q3)
        # (c) One in a 4-cycle, one elsewhere
        
        # For (a): (a,b) and (c,d) are opposite in a 4-cycle.
        # In Q3, 4-cycle: 0-1-3-2-0. Opposite edges: (0,1) and (2,3).
        # So a->0, b->1, c->2, d->3 (or permutations).
        # Need: a-c adjacent (dist 2 in Q3), b-d adjacent (dist 2 in Q3).
        # In the graph: need a common neighbor of a and c (that's Q3 vertex... wait)
        # Actually in the 4-cycle 0-1-3-2-0:
        #   Edges: (0,1), (1,3), (3,2), (2,0)
        #   Opposite edges: (0,1) and (3,2), or (1,3) and (2,0)
        #   For (0,1) and (2,3): need vertex 3 common neighbor of 1 and 2,
        #   and vertex 0... wait, 0 is already used.
        #   The 4-cycle uses vertices 0,1,2,3. Edges: (0,1),(1,3),(3,2),(2,0).
        #   Opposite pairs: {(0,1),(2,3)} and {(1,3),(0,2)}.
        #   So for edges (a,b)~(0,1) and (c,d)~(2,3): need common neighbor of b and c
        #   (that's Q3 vertex 3), and common neighbor of a and d (that's Q3 vertex... 
        #   wait, in the 4-cycle, a=0, b=1, c=2, d=3. Common neighbor of b=1 and c=2
        #   in Q3: vertex 3. But d=3 is already used. Hmm.
        #   Actually in Q3, N(1)={0,3,5}, N(2)={0,3,6}. Common: {0,3}. 
        #   0 is used (a), 3 is d. So the common neighbor is d itself? No.
        #   Wait, I'm confusing myself. Let me redo.
        #   
        #   In Q3, the 4-cycle containing edges (0,1) and (2,3):
        #   Is there a 4-cycle with these as opposite edges?
        #   4-cycles in Q3: for each pair of dimensions, there's a 4-cycle.
        #   Dimensions {0,1}: cycle 000-001-011-010-000, i.e., 0-1-3-2-0.
        #     Edges: (0,1),(1,3),(3,2),(2,0). Opposite: (0,1)&(2,3), (1,3)&(0,2).
        #   So (0,1) and (2,3) ARE opposite in this 4-cycle.
        #   The other two vertices in the cycle are 3 and... wait, the cycle is 0,1,3,2.
        #   So vertices used: 0,1,2,3. Edges (0,1) and (2,3) are opposite.
        #   For this to work in the graph: need edge (1,3) and edge (0,2) in graph.
        #   i.e., b must be adjacent to some vertex (Q3 v3), and a must be adjacent to 
        #   some vertex (Q3 v2=c). Wait, c IS Q3 vertex 2.
        #   So we need: b adjacent to Q3_v3, and a adjacent to c (Q3_v2).
        #   And Q3_v3 is a new vertex (not a,b,c,d).
        #   Also need: Q3_v3 adjacent to c (edge (3,2) in Q3).
        #   So: Q3_v3 must be common neighbor of b and c in the graph.
        #   And a must be adjacent to c (already given as edge (c,d)? No, (a,b) and (c,d) are the edges.
        #   a is adjacent to b, c is adjacent to d. We need a adjacent to c? No!
        #   In the 4-cycle 0-1-3-2-0: edges are (0,1),(1,3),(3,2),(2,0).
        #   (0,1) = (a,b), (2,3) = (c,d) means: 0->a, 1->b, 2->c, 3->d.
        #   Then we need edges (1,3) = (b,d) and (2,0) = (c,a) in the graph.
        #   So we need: b adjacent to d, AND c adjacent to a.
        
        if (adj[a] >> c) & 1 and (adj[b] >> d) & 1:
            return True  # 4-cycle: a-b-d-c-a with edges (a,b) and (c,d) opposite
        if (adj[a] >> d) & 1 and (adj[b] >> c) & 1:
            return True  # 4-cycle: a-b-c-d-a with edges (a,b) and (c,d) opposite
        
        # (b) Edges in "opposite" positions, e.g., (0,1) and (4,5) in Q3.
        # (0,1) is in the 000-001 edge, (4,5) is 100-101.
        # These are in different 4-cycles. The Q3 vertices: 0,1,4,5.
        # We need to complete: 2,3,6,7.
        # 2: neighbor of 0, not 1,4. In Q3: 2 is neighbor of 0.
        # 3: neighbor of 1 and 2.
        # 6: neighbor of 2 and 4.
        # 7: neighbor of 3,5,6. Also neighbor of 4 and 5? No: 7=111, 4=100, 5=101.
        #   7 is adjacent to 4? 111 vs 100: differ in 2 bits. No.
        #   7 is adjacent to 5? 111 vs 101: differ in 1 bit. Yes.
        #   So 7 is adjacent to 3, 5, and 6.
        
        # Place: 0->a, 1->b, 4->c, 5->d.
        # Need:
        #   v2: neighbor of a (Q3 v0), not b,c,d
        #   v3: common neighbor of b (Q3 v1) and v2
        #   v6: common neighbor of v2 and c (Q3 v4)
        #   v7: common neighbor of v3, d (Q3 v5), and v6
        
        used_base = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        cands_v2 = adj[a] & ~used_base
        if cands_v2 == 0:
            return False
        
        c2 = cands_v2
        while c2:
            lsb2 = c2 & (-c2)
            v2 = lsb2.bit_length() - 1
            c2 ^= lsb2
            used2 = used_base | (1 << v2)
            
            cands_v3 = adj[b] & adj[v2] & ~used2
            if cands_v3 == 0:
                continue
            c3 = cands_v3
            while c3:
                lsb3 = c3 & (-c3)
                v3 = lsb3.bit_length() - 1
                c3 ^= lsb3
                used3 = used2 | (1 << v3)
                
                cands_v6 = adj[v2] & adj[c] & ~used3
                if cands_v6 == 0:
                    continue
                c6 = cands_v6
                while c6:
                    lsb6 = c6 & (-c6)
                    v6 = lsb6.bit_length() - 1
                    c6 ^= lsb6
                    used4 = used3 | (1 << v6)
                    
                    cands_v7 = adj[v3] & adj[d] & adj[v6] & ~used4
                    if cands_v7 != 0:
                        return True
        
        # Also try: 0->a, 1->b, 4->d, 5->c (swap c and d)
        used_base2 = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        cands_v2b = adj[a] & ~used_base2
        if cands_v2b == 0:
            return False
        
        c2b = cands_v2b
        while c2b:
            lsb2b = c2b & (-c2b)
            v2 = lsb2b.bit_length() - 1
            c2b ^= lsb2b
            used2b = used_base2 | (1 << v2)
            
            cands_v3b = adj[b] & adj[v2] & ~used2b
            if cands_v3b == 0:
                continue
            c3b = cands_v3b
            while c3b:
                lsb3b = c3b & (-c3b)
                v3 = lsb3b.bit_length() - 1
                c3b ^= lsb3b
                used3b = used2b | (1 << v3)
                
                cands_v6b = adj[v2] & adj[d] & ~used3b  # v4 is now d
                if cands_v6b == 0:
                    continue
                c6b = cands_v6b
                while c6b:
                    lsb6b = c6b & (-c6b)
                    v6 = lsb6b.bit_length() - 1
                    c6b ^= lsb6b
                    used4b = used3b | (1 << v6)
                    
                    cands_v7b = adj[v3] & adj[c] & adj[v6] & ~used4b  # v5 is now c
                    if cands_v7b != 0:
                        return True
        
        # (c) Other configurations: e.g., (0,1) and (4,6) in Q3.
        # 4=100, 6=110. Edge (4,6) is in Q3.
        # 0=000, 1=001. Edge (0,1).
        # These share no 4-cycle. Need to complete.
        # 0->a, 1->b, 4->c, 6->d.
        # 2: neighbor of 0 (a), not 1,4,6
        # 3: neighbor of 1 (b) and 2
        # 5: neighbor of 1 (b) and 4 (c)
        # 7: neighbor of 3, 5, and 6 (d)
        
        used_base3 = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        cands_v2c = adj[a] & ~used_base3
        if cands_v2c == 0:
            return False
        
        c2c = cands_v2c
        while c2c:
            lsb2c = c2c & (-c2c)
            v2 = lsb2c.bit_length() - 1
            c2c ^= lsb2c
            used2c = used_base3 | (1 << v2)
            
            cands_v3c = adj[b] & adj[v2] & ~used2c
            if cands_v3c == 0:
                continue
            c3c = cands_v3c
            while c3c:
                lsb3c = c3c & (-c3c)
                v3 = lsb3c.bit_length() - 1
                c3c ^= lsb3c
                used3c = used2c | (1 << v3)
                
                cands_v5c = adj[b] & adj[c] & ~used3c
                if cands_v5c == 0:
                    continue
                c5c = cands_v5c
                while c5c:
                    lsb5c = c5c & (-c5c)
                    v5 = lsb5c.bit_length() - 1
                    c5c ^= lsb5c
                    used4c = used3c | (1 << v5)
                    
                    cands_v7c = adj[v3] & adj[v5] & adj[d] & ~used4c
                    if cands_v7c != 0:
                        return True
        
        # There are more configurations... this is getting very complex.
        # For a practical solution, let's use a simpler but valid approach:
        # Just check the most common cases. The branch-and-bound will handle
        # the rest with the actual Q3 check.
        
        return False
    
    # Build conflict graph
    for i in range(_E):
        for j in range(i + 1, _E):
            if _can_be_in_q3(_ALL_EDGES[i], _ALL_EDGES[j], adj):
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    return conflicts


# Build the conflict graph
print("Building conflict graph...")
_start_build = time.time()
_CONFLICTS = _build_conflict_graph()
print(f"Conflict graph built in {time.time() - _start_build:.2f}s")


def _mis_branch_and_bound(conflicts, deadline, initial_solution=None):
    """
    Maximum independent set using branch-and-bound.
    conflicts[i] is a bitmask of nodes conflicting with i.
    Returns the best independent set found as a bitmask.
    """
    n = _E
    best_set = 0
    best_size = 0
    
    if initial_solution is not None:
        best_set = initial_solution
        best_size = bin(initial_solution).count('1')
    
    # Upper bound: greedy coloring of the remaining graph
    def greedy_color_bound(remaining):
        """Color the induced subgraph on 'remaining' vertices.
        The number of colors is an upper bound on the independent set size."""
        colors = 0
        uncolored = remaining
        while uncolored:
            colors += 1
            # Find a maximal independent set in the uncolored subgraph
            # (i.e., a color class)
            color_class = 0
            avail = uncolored
            while avail:
                # Pick vertex with minimum degree in avail
                # For speed, just pick the first one
                lsb = avail & (-avail)
                v = lsb.bit_length() - 1
                avail ^= lsb
                color_class |= lsb
                # Remove neighbors of v from avail
                avail &= ~conflicts[v]
            uncolored &= ~color_class
        return colors
    
    def bb(remaining, current_set, current_size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if remaining == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        
        # Upper bound
        ub = current_size + greedy_color_bound(remaining)
        if ub <= best_size:
            return
        
        # Pick a vertex to branch on
        # Choose vertex with highest degree in remaining (most constraining)
        lsb = remaining & (-remaining)
        v = lsb.bit_length() - 1
        
        # Option 1: include v
        # New remaining: remove v and all its neighbors
        new_remaining = remaining & ~conflicts[v] & ~(1 << v)
        bb(new_remaining, current_set | (1 << v), current_size + 1)
        
        # Option 2: exclude v
        bb(remaining & ~(1 << v), current_set, current_size)
    
    bb((1 << n) - 1, 0, 0)
    return best_set


def _greedy_is(conflicts):
    """Greedy independent set: repeatedly pick vertex with minimum degree."""
    n = _E
    remaining = (1 << n) - 1
    result = 0
    
    while remaining:
        # Pick vertex with minimum degree in remaining
        best_v = -1
        best_deg = n
        c = remaining
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            c ^= lsb
            deg = bin(conflicts[v] & remaining).count('1')
            if deg < best_deg:
                best_deg = deg
                best_v = v
            if deg == 0:
                break
        
        if best_v == -1:
            break
        
        result |= (1 << best_v)
        remaining &= ~conflicts[best_v]
        remaining &= ~(1 << best_v)
    
    return result


def _is_q3_free(edge_mask):
    """Check if the graph with edges specified by edge_mask is Q3-free."""
    adj = [0] * _N
    for i in range(_E):
        if (edge_mask >> i) & 1:
            u, v = _ALL_EDGES[i]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Check for Q3 by trying all embeddings
    # For speed, check each edge as potential Q3 edge (0,1)
    for i in range(_E):
        if not ((edge_mask >> i) & 1):
            continue
        u, v = _ALL_EDGES[i]
        if _has_q3_with_edge_fast(adj, u, v):
            return False
    return True


def _has_q3_with_edge_fast(adj_bits, u, v):
    """Fast Q3 detection for edge (u,v)."""
    order = [2, 4, 3, 5, 6, 7]
    assigned = [-1] * 8
    assigned[0] = u
    assigned[1] = v
    used_mask = (1 << u) | (1 << v)
    
    def backtrack(idx):
        nonlocal used_mask
        if idx == len(order):
            for i in range(2, 8):
                for j in range(i + 1, 8):
                    if (i ^ j) in (1, 2, 4):
                        if not (adj_bits[assigned[i]] >> assigned[j]) & 1:
                            return False
            return True
        
        qv = order[idx]
        candidates = (~used_mask) & ((1 << _N) - 1)
        
        if qv == 2:
            candidates &= adj_bits[u]
        elif qv == 4:
            candidates &= adj_bits[u]
        elif qv == 3:
            candidates &= adj_bits[v] & adj_bits[assigned[2]]
        elif qv == 5:
            candidates &= adj_bits[v] & adj_bits[assigned[4]]
        elif qv == 6:
            candidates &= adj_bits[assigned[2]] & adj_bits[assigned[4]]
        elif qv == 7:
            candidates &= adj_bits[assigned[3]] & adj_bits[assigned[5]] & adj_bits[assigned[6]]
        
        if candidates == 0:
            return False
        
        c = candidates
        while c:
            lsb = c & (-c)
            gv = lsb.bit_length() - 1
            c ^= lsb
            
            assigned[qv] = gv
            used_mask |= (1 << gv)
            if backtrack(idx + 1):
                return True
            used_mask &= ~(1 << gv)
            assigned[qv] = -1
        
        return False
    
    return backtrack(0)


def _greedy_construction(adj, edge_order, deadline):
    """Greedy edge addition with Q3 check."""
    edges = []
    adj = adj[:]
    
    for u, v in edge_order:
        if time.time() > deadline:
            break
        
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        if not _has_q3_with_edge_fast(adj, u, v):
            edges.append([u, v])
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    return edges, adj


def _local_search(adj, edges, all_edges, deadline):
    """Local search: try to add missing edges."""
    edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
    missing = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(missing)
        for u, v in missing[:]:
            if time.time() > deadline:
                break
            if (u, v) in edge_set:
                continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not _has_q3_with_edge_fast(adj, u, v):
                edges.append([u, v])
                edge_set.add((u, v))
                if (u, v) in missing:
                    missing.remove((u, v))
                improved = True
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    return edges, adj


def _edges_to_mask(edges):
    """Convert list of edges to a bitmask over edge indices."""
    mask = 0
    for u, v in edges:
        idx = _EDGE_INDEX[(u, v)]
        mask |= (1 << idx)
    return mask


def _mask_to_edges(mask):
    """Convert edge bitmask to list of [u, v]."""
    edges = []
    for i in range(_E):
        if (mask >> i) & 1:
            edges.append(list(_ALL_EDGES[i]))
    return edges


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_MARGIN
    
    best_edges = []
    best_count = 0
    best_adj = [0] * _N
    
    # Phase 1: Greedy constructions to get initial solutions
    num_trials = 30
    for trial in range(num_trials):
        if time.time() > deadline - 30:
            break
        
        if trial == 0:
            edge_order = _ALL_EDGES[:]
        elif trial == 1:
            edge_order = _ALL_EDGES[::-1]
        else:
            random.seed(trial * 7919 + 104729)
            edge_order = _ALL_EDGES[:]
            random.shuffle(edge_order)
        
        adj = [0] * _N
        edges, adj = _greedy_construction(adj, edge_order, deadline - 25)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            best_adj = adj[:]
    
    # Phase 2: Local search
    if time.time() < deadline - 20:
        adj = best_adj[:]
        edges = best_edges[:]
        edges, adj = _local_search(adj, edges, _ALL_EDGES, deadline - 15)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            best_adj = adj
    
    # Phase 3: Branch-and-bound on conflict graph
    if time.time() < deadline - 5:
        # Get initial solution as bitmask
        init_mask = _edges_to_mask(best_edges)
        
        # Run B&B with time limit
        bb_deadline = deadline - 2
        best_mask = _mis_branch_and_bound(_CONFLICTS, bb_deadline, init_mask)
        
        # Verify the solution is Q3-free
        if _is_q3_free(best_mask):
            bb_edges = _mask_to_edges(best_mask)
            if len(bb_edges) > best_count:
                best_count = len(bb_edges)
                best_edges = bb_edges
        else:
            # Fallback: use greedy IS
            greedy_mask = _greedy_is(_CONFLICTS)
            if _is_q3_free(greedy_mask):
                greedy_edges = _mask_to_edges(greedy_mask)
                if len(greedy_edges) > best_count:
                    best_count = len(greedy_edges)
                    best_edges = greedy_edges
    
    # Phase 4: Final local search
    if time.time() < deadline - 1:
        adj = [0] * _N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        edges, adj = _local_search(adj, best_edges, _ALL_EDGES, deadline - 0.5)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    return best_edges
# EVOLVE-BLOCK-END
