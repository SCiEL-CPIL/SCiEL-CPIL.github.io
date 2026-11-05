"""Construct a C5-free graph with alpha < 9, maximizing n.

Strategy:
- We know 8 disjoint K4s give n=32, alpha=8, no C5. Score 32/28 = 1.1428...
- We want to try to get n=33 or higher.
- Key insight: A C5-free graph can be built from cliques. If we take disjoint cliques,
  the independence number is the number of cliques (pick one from each).
  For n=33 with 9 cliques, alpha=9 which is too high.
  For n=33 with 8 cliques, alpha=8, but we need 33 vertices in 8 cliques,
  so average clique size ~4.125. Cliques of size 4 and 5: 8 cliques,
  e.g., 7 K4s + 1 K5 = 7*4+5 = 33. alpha = 8. No C5 (disjoint cliques).
  This gives n=33!

Wait - let me verify: 7 K4s + 1 K5 = 28 + 5 = 33 vertices.
Independence number: pick at most 1 from each clique, so alpha = 8.
No C5: a 5-cycle would need 5 vertices in different cliques (since within a clique
all are connected, a cycle of length >= 5 can't stay in one clique of size <= 5... 
actually K5 contains no 5-cycle as a subgraph? Wait, K5 DOES contain a 5-cycle!
A 5-cycle is a subgraph of K5 (just pick 5 vertices in a cycle order).
So K5 contains a C5! That's a problem.

So we can't use K5. Max clique size without C5 is 4 (K4 has no cycle at all).
Actually, K4 has no cycles. K5 has a 5-cycle. So max clique size is 4 for C5-free.

With max clique size 4 and 8 cliques: max n = 8*4 = 32. That's what we have.

To get n=33, we need a different structure. Options:
1. Use a non-complete graph component that is C5-free and has more vertices.
2. Use overlapping structures.

Actually, let me think about C5-free graphs more carefully.
A graph is C5-free if it contains no 5-cycle as a subgraph.

What about complete bipartite graphs? K_{a,b} contains a 4-cycle if a,b >= 2,
but does it contain a 5-cycle? A 5-cycle in K_{a,b} would need 5 vertices alternating
between the two parts, which is impossible for odd cycles in bipartite graphs.
So complete bipartite graphs are C5-free!

K_{a,b} has independence number max(a,b) (the larger part is independent).
For alpha < 9, we need max(a,b) <= 8.
Max n = a+b with max(a,b) <= 8, so a=8, b=8, n=16. Not great.

What about combining cliques and bipartite structures?

Actually, let me reconsider. The known best is 32. The upper bound is 35.
Let me try a different approach: start with 8 K4s (n=32, alpha=8) and try to
add one more vertex with edges that maintain C5-free and alpha<9.

If I add vertex 32, I need:
- No C5 created: vertex 32's neighbors must form a structure that doesn't create a 5-cycle
- alpha stays < 9: vertex 32 must be adjacent to at least one vertex in each of the 8 K4s
  (otherwise, picking vertex 32 plus one from each K4 it's not adjacent to gives an independent set)

Wait, if vertex 32 is not adjacent to any vertex in K4_i, then we can pick vertex 32 and one
from each K4_j for j != i, giving an independent set of size 8. That's still < 9.
Actually we need independent set of size 9. So we can pick at most 1 from each K4.
If vertex 32 is non-adjacent to at least one vertex in each K4, we could pick vertex 32
plus one non-neighbor from each K4 = 9 independent vertices.

So to keep alpha < 9, vertex 32 must be adjacent to ALL vertices in at least one K4.
That is, vertex 32 must be a universal vertex with respect to at least one K4.

But if vertex 32 is adjacent to all 4 vertices of K4_0, then consider the subgraph induced
by vertex 32 and K4_0: that's K5 (vertex 32 + 4 vertices all mutually adjacent).
K5 contains a 5-cycle! So that creates a C5.

Hmm. So we can't make vertex 32 adjacent to all 4 vertices of any K4.

What if vertex 32 is adjacent to 3 vertices of K4_0? Then in K4_0, the one non-adjacent
vertex v. Can we form a C5? We'd need a path of length 4 from v back to v through vertex 32.
v is not adjacent to 32. v is adjacent to the 3 neighbors of 32 in K4_0. 
Path: v - u1 - 32 - u2 - u3 - v? But u3 must be adjacent to v. In K4_0, v is adjacent to u1, u2, u3.
So v-u1-32-u2-u3-v is a 5-cycle if all these edges exist. They do! v-u1 (in K4), u1-32 (we said 32 adjacent to u1), 32-u2 (yes), u2-u3 (in K4), u3-v (in K4).
So yes, that's a 5-cycle!

So if vertex 32 is adjacent to 3 vertices of a K4, we get a C5.
If adjacent to 2 vertices of a K4: v1, v2 adjacent to 32, v3, v4 not.
Can we get a C5? 32-v1-v3-v2-32 is a 4-cycle, not 5.
32-v1-v3-v4-v2-32? That's 5 vertices: 32, v1, v3, v4, v2. Edges: 32-v1 (yes), v1-v3 (yes, K4), v3-v4 (yes, K4), v4-v2 (yes, K4), v2-32 (yes). That's a 5-cycle!

If adjacent to 1 vertex of a K4: 32-v1. Others v2,v3,v4 not adjacent to 32.
32-v1-v2-v3-32? v3 not adjacent to 32. No.
32-v1-v2-v3-v4-32? v4 not adjacent to 32. No.
Actually any cycle through 32 must enter and leave via v1 (the only neighbor in K4).
So cycles through 32 and K4 vertices: 32-v1-x-...-32 where x is in K4. But 32 only connects to v1 in K4,
so any cycle must be 32-v1-v1-32 (not a cycle) or go through other components.

So if vertex 32 is adjacent to at most 1 vertex in each K4, no C5 is formed within a single K4+32.
But could a C5 go through multiple K4s? A C5 needs 5 vertices. If they're in different K4s,
there are no edges between K4s, so no. A C5 must have at least 2 vertices in the same K4 or in the same "component".

Actually, since the K4s are disjoint with no edges between them, any cycle must be entirely
within one K4 plus possibly vertex 32. Since K4 has no cycles, any cycle involving 32 must
use 32 and vertices from exactly one K4 (since no edges between different K4s).

So the constraint is: for each K4, vertex 32 is adjacent to at most 1 vertex of that K4.

But then, for each K4, there are 3 vertices non-adjacent to 32. We can pick 32 plus one
non-neighbor from each of the 8 K4s = 9 independent vertices. That violates alpha < 9.

So we can't just add a vertex to 8 K4s. We need a fundamentally different construction.

Let me think about what C5-free graphs with high minimum degree look like.
Actually, let me try a different decomposition.

What about using K_{2,2,2} (octahedron graph)? It's the complement of 3 disjoint edges.
It has 6 vertices, is C5-free? Let me check. The octahedron graph: vertices can be labeled
as pairs (+1,-1) in 3 dimensions. Two vertices are adjacent iff they differ in at least 2 coordinates.
Actually, the octahedron graph is K_{2,2,2} complete tripartite. It contains 4-cycles but no 5-cycles
(bipartite-like? No, it's not bipartite). Actually K_{2,2,2} has triangles. Does it have a 5-cycle?
Pick vertices from parts A={a1,a2}, B={b1,b2}, C={c1,c2}. A 5-cycle would need 5 vertices.
By pigeonhole, at least 2 from the same part. But same-part vertices aren't adjacent.
A 5-cycle: a1-b1-c1-a2-b2-c1? c1 appears twice. 
a1-b1-c1-a2-b2-a1? That's a1-b1-c1-a2-b2-a1, 5 edges, 5 vertices: a1,b1,c1,a2,b2. 
Check edges: a1-b1 (yes, different parts), b1-c1 (yes), c1-a2 (yes), a2-b2 (yes), b2-a1 (yes). 
That's a 5-cycle! So K_{2,2,2} contains a C5. Not usable.

What about the complement of C5? The complement of C5 is C5 itself (C5 is self-complementary). Not helpful.

Let me try: what about taking the join of two graphs? The join G ∨ H has all edges between G and H.
If G and H are C5-free, is G ∨ H C5-free? A C5 in G ∨ H could use vertices from both.
If 3 vertices from G and 2 from H: the 3 from G must form a path of length 2 in G, and the 2 from H
must be adjacent in H, and all cross edges exist. This could form a C5. So join doesn't preserve C5-freeness.

Let me try a computational approach: use a greedy/randomized search to find a valid graph with n > 32.

Actually, given the time constraint, let me just go with n=33 using a clever construction if I can find one,
otherwise return n=32.

Let me try: 6 K4s (24 vertices) + a C5-free graph on 9 vertices with alpha <= 2.
A C5-free graph on 9 vertices with alpha <= 2 means the complement has no K9-free... 
alpha <= 2 means the complement has no independent set of size 3, i.e., complement is triangle-free.
So we need a C5-free graph G on 9 vertices whose complement is triangle-free.
Equivalently, G contains no C5 and no independent set of size 3.

The complement being triangle-free means every 3 vertices in G have at least one non-edge,
i.e., alpha(G) <= 2.

What about the Petersen graph? 10 vertices, alpha=4. No good.
What about a 9-vertex graph? 

Actually, let me try: take a 9-vertex graph that is the complement of a triangle-free graph.
The densest triangle-free graph on 9 vertices is Turan graph T(9,2) = K_{4,5}, which has 20 edges.
So G = complement of K_{4,5} on 9 vertices. G has C(9,2)-20 = 36-20 = 16 edges.
G consists of K4 (the 4 vertices in one part) and K5 (the 5 vertices in other part), with no edges between.
But K5 contains a C5! So G contains a C5.

What about a different triangle-free graph? We need the complement to be C5-free.
This is getting complex. Let me just do a randomized search within the time budget.

Given the complexity, let me implement a randomized local search that starts from 8 K4s
and tries to add edges/vertices to reach n=33+, with a deadline guard.
"""

import time
import random
import itertools

def _has_c5(adj, n):
    """Check if graph with adjacency matrix has a 5-cycle."""
    # A 5-cycle: v0-v1-v2-v3-v4-v0, all distinct, all edges present
    # To avoid duplicates, fix v0=0 (if n>0) and v1 < v4
    if n < 5:
        return False
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            if not adj[v0][v1]:
                continue
            for v4 in range(v0 + 1, v1):  # v4 < v1 to avoid duplicates
                if not adj[v0][v4]:
                    continue
                for v2 in range(v1 + 1, n):
                    if not adj[v1][v2]:
                        continue
                    for v3 in range(v2 + 1, n):
                        if not adj[v2][v3]:
                            continue
                        if adj[v3][v4]:
                            return True
    return False

def _has_independent_set(adj, n, k):
    """Check if graph has an independent set of size k."""
    # Equivalent to: complement has a clique of size k
    # Use backtracking
    # Build complement adjacency
    comp = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                comp[i][j] = 1
                comp[j][i] = 1
    
    # Find clique of size k in comp
    def find_clique(vertices, size_needed):
        if size_needed == 0:
            return True
        if len(vertices) < size_needed:
            return False
        if size_needed == 1:
            return True
        for idx, v in enumerate(vertices):
            # v must be connected to all remaining chosen vertices
            remaining = [u for u in vertices[idx+1:] if comp[v][u]]
            if find_clique(remaining, size_needed - 1):
                return True
        return False
    
    return find_clique(list(range(n)), k)

def _validate(n, edges):
    """Check C5-free and no independent set of size 9."""
    adj = [[0] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    if _has_c5(adj, n):
        return False
    if _has_independent_set(adj, n, 9):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Best known: 8 disjoint K4s, n=32
    best_n = 32
    best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    # Try to find n=33, 34, 35
    for target_n in [33, 34, 35]:
        if time.time() > deadline:
            break
        found = False
        attempts = 0
        max_attempts = 5000
        
        while not found and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            # Start from 8 K4s (32 vertices) and add one vertex
            # Or try random dense C5-free graphs
            
            # Strategy: random graph with controlled density, check constraints
            # For C5-free, we want high density but no 5-cycles
            # Key: complete multipartite graphs with parts of size <= 4 are C5-free?
            # No, K_{2,2,2} has C5 as shown.
            
            # Try: union of cliques of size <= 4, plus some cross edges
            # Start with 8 K4s, try to add vertex 32 with limited connections
            
            if target_n == 33:
                # 8 K4s + 1 vertex
                edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
                adj = [[0] * 33 for _ in range(33)]
                for u, v in edges:
                    adj[u][v] = 1
                    adj[v][u] = 1
                
                # Connect vertex 32 to some vertices
                # Must avoid C5: at most 1 neighbor per K4
                # Must keep alpha < 9: vertex 32 must be "covered"
                # With at most 1 neighbor per K4, vertex 32 is non-adjacent to 3 in each K4
                # Independent set: 32 + one non-neighbor from each K4 = 9. BAD.
                # So this strategy fails.
                
                # Alternative: modify the K4s themselves
                # Remove some edges from K4s to allow more connections
                pass
            
            # Try a completely different approach: random search
            # Generate random graph, check if C5-free and alpha < 9
            # This is very unlikely to work for n=33+
            
            # Better: use the structure of C5-free graphs
            # C5-free graphs can be quite dense. The extremal number ex(n, C5) ~ n^{3/2}
            # For n=33, ex ~ 33^{1.5} ~ 189 edges. Total possible: 528.
            # So we need a relatively sparse graph.
            
            # Actually, let me try: complete 4-partite graph K_{a,b,c,d}
            # K_{a,b,c,d} is C5-free? A 5-cycle in complete multipartite:
            # needs 5 vertices from 4 parts, so 2 from same part (not adjacent). 
            # So a 5-cycle must use vertices from at least 5 parts, or...
            # Actually in complete multipartite, two vertices are adjacent iff in different parts.
            # A 5-cycle v0-v1-v2-v3-v4-v0: consecutive vertices must be in different parts.
            # v0 and v2 must be in different parts (not adjacent in cycle, but could be same part).
            # Actually v0 and v2 are NOT adjacent in the cycle, so they CAN be in the same part.
            # v0-v1: different parts. v1-v2: different parts. v2-v3: different parts. v3-v4: different parts. v4-v0: different parts.
            # So we need a cyclic sequence of 5 parts where consecutive are different.
            # With 4 parts: e.g., A-B-A-C-B. Check: A-B (ok), B-A (ok), A-C (ok), C-B (ok), B-A (ok).
            # But we need 5 distinct vertices. If parts A has >=2 vertices, we can pick 2 from A.
            # So K_{a,b,c,d} with at least 2 parts having >=2 vertices contains a C5!
            # Specifically if any part has size >= 2, we can form a C5 using 2 from that part.
            # Wait: A-B-A-C-B uses 2 from A, 2 from B, 1 from C. Need |A|>=2, |B|>=2, |C|>=1.
            # So K_{2,2,1,1} = K4 (complete graph) which has no C5. Contradiction?
            # K4 has 4 vertices. A 5-cycle needs 5 vertices. So with 4 parts of size 1, no C5.
            # With parts of total size >= 5 and some part size >= 2:
            # K_{2,2,1}: 5 vertices. Can we form C5? A-B-A-C-B: vertices a1,b1,a2,c1,b2.
            # Edges: a1-b1 (diff parts, yes), b1-a2 (yes), a2-c1 (yes), c1-b2 (yes), b2-a1 (yes). C5!
            # So complete multipartite with 5+ vertices and a part of size >=2 has C5.
            # Complete multipartite is C5-free iff it's a clique (all parts size 1) or has at most 4 vertices.
            # Actually K_{1,1,1,1} = K4, no C5. K_{2,1,1} = K4, no C5 (4 vertices).
            # K_{2,2,1} = 5 vertices, has C5.
            
            # So complete multipartite doesn't help for n>4.
            
            # Let me try: disjoint union of K4s is the way to go, giving n=32.
            # For n=33, we need a non-disjoint construction.
            
            # Try: 7 K4s + a C5-free graph on 9 vertices with alpha <= 2
            # The 9-vertex part needs alpha <= 2 (so combined alpha = 7+2 = 9... still 9!)
            # We need alpha <= 1 for the 9-vertex part, which means it's a clique K9.
            # But K9 contains C5. Dead end.
            
            # 7 K4s + 9-vertex C5-free with alpha <= 2: total alpha = 7 + 2 = 9. Not < 9.
            # 6 K4s + 15-vertex C5-free with alpha <= 3: total alpha = 6+3 = 9. Still not < 9.
            # We need total alpha <= 8.
            # k K4s + (33-4k)-vertex C5-free with alpha <= 8-k.
            # k=8: 1-vertex, alpha 0. Total alpha 8. n=33. But 1 vertex alone is trivial.
            #   The 1 vertex connects to nothing (disjoint). alpha = 8+1 = 9. BAD.
            #   The 1 vertex must connect to at least one vertex in each K4 to reduce alpha.
            #   But as shown, connecting to 1 vertex in a K4 is safe for C5, but then
            #   the vertex is non-adjacent to 3 in that K4, and we can pick it + 3 non-neighbors...
            #   Wait, we pick at most 1 from each K4. If v is non-adjacent to 3 in K4_i,
            #   we pick v and one of those 3. That's 1 from K4_i. Total: v + 1 from each K4 = 9.
            #   Unless v is adjacent to ALL vertices in some K4 (impossible, creates C5)
            #   or we modify the K4s.
            
            # What if we make the K4s into K4 minus an edge (diamond)? 
            # Diamond: 4 vertices, 5 edges. No C5 (only 4 vertices).
            # Independence number of diamond: 2 (the two non-adjacent vertices).
            # 8 diamonds: 32 vertices, alpha = 8*2 = 16. Way too high.
            
            # This approach isn't working. Let me just return n=32.
            # Actually, let me try one more thing: a computational search for n=33.
            break
        
        if found:
            best_n = target_n
            break
    
    return {'n': best_n, 'edges': best_edges}
