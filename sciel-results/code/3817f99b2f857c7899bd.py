"""Program for maximizing edges in a Q3-free graph on 16 vertices."""
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a Q3-free graph on 16 vertices with as many edges as possible.
    
    Strategy:
    1. Start with K_{4,4,4,4} (complete 4-partite with parts of size 4).
       This is Q3-free because Q3 is not bipartite? Wait, Q3 is bipartite.
       Actually, Q3 has 8 vertices, 12 edges, is bipartite, 3-regular.
       K_{4,4,4,4} has 4*4*3 = 48 edges.
       Is K_{4,4,4,4} Q3-free? A subgraph isomorphic to Q3 must be induced?
       No, "subgraph isomorphic to" means there exists an injective map from V(Q3) to V(G)
       such that edges of Q3 map to edges of G. It doesn't need to be induced.
       
       Let's check: Can Q3 be embedded in K_{4,4,4,4}?
       Q3 is bipartite with partitions of size 4 and 4.
       In K_{4,4,4,4}, any two vertices in the same part are non-adjacent.
       Any two vertices in different parts are adjacent.
       
       For Q3 to be embedded, we need to map the 8 vertices of Q3 to 8 distinct vertices of G
       such that all 12 edges of Q3 are present in G.
       The vertices of Q3 form a bipartite graph with parts A and B, each of size 4.
       In Q3, there are no edges within A or within B.
       In K_{4,4,4,4}, if we map two vertices from A to the same part in G, they are non-adjacent (good, since they're non-adjacent in Q3).
       If we map two vertices from A to different parts in G, they are adjacent (bad, since they're non-adjacent in Q3).
       So, for the embedding to work, all 4 vertices of A must be mapped to the same part in G, and all 4 vertices of B must be mapped to the same part in G.
       Since each part in G has size 4, this is possible: map A to part 0, B to part 1.
       Then all edges between A and B in Q3 correspond to edges between part 0 and part 1 in G, which are all present.
       So K_{4,4,4,4} DOES contain Q3 as a subgraph!
       
       Therefore, K_{4,4,4,4} is NOT Q3-free.
       
    Let's rethink.
    
    Alternative strategy:
    The current seed uses K_{3,13} which has 3*13 = 39 edges.
    K_{3,13} is bipartite with parts 3 and 13.
    Q3 is bipartite with parts 4 and 4.
    For Q3 to be a subgraph of K_{3,13}, we would need to map the 4 vertices of one part of Q3 to the 3 vertices of one side of K_{3,13}, which is impossible by pigeonhole principle (4 > 3).
    So K_{3,13} is indeed Q3-free.
    
    Can we do better than 39 edges?
    
    Let's try a different approach.
    We want to maximize edges while avoiding Q3.
    
    Let's use a random greedy approach with local search.
    Start with a random graph or a known Q3-free graph, then try to add edges.
    
    Actually, let's think about the structure of Q3.
    Q3 is a 3-regular bipartite graph on 8 vertices.
    
    Let's try to construct a graph with more edges.
    
    Consider the complement. We want to minimize non-edges while keeping the graph Q3-free.
    
    Let's try a different partition.
    What if we use a complete graph K_16 and remove edges to destroy all Q3 subgraphs?
    K_16 has 120 edges. We need to remove enough edges to break every Q3.
    
    This is a hitting set problem.
    
    Alternatively, let's try a constructive approach.
    
    Let's try a 2-partite graph K_{m,n} with m+n=16.
    To be Q3-free, we need min(m,n) < 4, because Q3 has parts of size 4.
    So max min(m,n) = 3, giving K_{3,13} with 39 edges.
    
    What if we add some edges within the parts?
    Start with K_{3,13}. Add edges within the part of size 13.
    If we add an edge (u,v) within the part of size 13, does it create a Q3?
    A Q3 would need 4 vertices from the part of size 3 and 4 from the part of size 13.
    Since the part of size 3 has only 3 vertices, we can't have 4 vertices from it.
    Wait, Q3 is bipartite with parts of size 4 and 4.
    In K_{3,13}, the bipartition is 3 and 13.
    For Q3 to be embedded, one part of Q3 (size 4) must map to a subset of one side of K_{3,13}, and the other part (size 4) to the other side.
    Since one side has size 3, we can't map 4 vertices to it.
    So K_{3,13} is Q3-free.
    
    If we add edges within the part of size 13, say between vertices u and v in the 13-part.
    Now, could there be a Q3?
    The Q3 would still need 4 vertices from one bipartition side and 4 from the other.
    But the graph is no longer bipartite.
    However, Q3 is bipartite.
    Any subgraph isomorphic to Q3 must be bipartite.
    So the 8 vertices of the Q3 must be 2-colorable.
    
    Let's say the 8 vertices are {v1,...,v8} with edges forming Q3.
    In the original K_{3,13}, these 8 vertices would have to be split between the 3-part and the 13-part.
    Since the 3-part has only 3 vertices, at least 5 vertices of the Q3 must be in the 13-part.
    In Q3, the maximum independent set has size 4 (one part of the bipartition).
    So among the 5 vertices in the 13-part, there must be at least 2 that are adjacent in Q3.
    But in K_{3,13}, vertices in the 13-part are not adjacent to each other.
    Unless we added an edge between them.
    
    So, if we add edges within the 13-part, we might create Q3 subgraphs.
    
    Let's try to add edges carefully.
    
    Actually, let's try a different approach: use a random search to find a Q3-free graph with more edges.
    
    We'll start with K_{3,13} and try to add edges within the larger part, checking each time if a Q3 is created.
    
    To check if a Q3 exists, we can iterate over all 8-tuples of vertices and check if they form a Q3.
    But that's O(n^8) which is too slow for n=16.
    
    Better: for each pair of vertices (u,v) that we consider adding an edge between, check if adding this edge creates a Q3.
    A Q3 containing the edge (u,v) would have u and v adjacent in the Q3.
    In Q3, each edge is in exactly 4 cycles of length 4? No, Q3 has girth 4.
    Each edge in Q3 is in exactly 2 cycles of length 4.
    Actually, Q3 has 12 edges, and each edge is in 2 4-cycles.
    
    This is getting complicated. Let's use a simpler check.
    
    For a given graph G, to check if it contains Q3:
    For each set of 8 vertices, check if the induced subgraph contains Q3 as a subgraph.
    But 8-sets of 16 vertices are C(16,8) = 12870, and for each, checking Q3 subgraph is expensive.
    
    Alternative: since Q3 is small, we can precompute all possible embeddings of Q3 in K_16.
    There are P(16,8) * (number of labelings of Q3) which is huge.
    
    Let's try a different tactic: use a SAT solver or brute force with pruning.
    
    Given the time constraint (120s), let's try a heuristic search.
    
    Start with K_{3,13} (39 edges).
    Try to add edges within the 13-part.
    For each potential edge (u,v) in the 13-part, add it and check if Q3 exists.
    If not, keep it. If yes, remove it.
    
    To check if Q3 exists after adding (u,v):
    Any new Q3 must contain the edge (u,v).
    So we need to find 6 other vertices w1,...,w6 such that {u,v,w1,...,w6} induces a Q3.
    
    In Q3, u and v are adjacent. Let's say u has neighbors a,b,c in Q3, and v has neighbors d,e,f.
    Wait, in Q3, each vertex has degree 3.
    If u and v are adjacent, they share no other neighbors in Q3 (since Q3 is bipartite and has no triangles).
    Actually, in Q3, two adjacent vertices are in different parts, so they have no common neighbors.
    
    So, for edge (u,v) to be in a Q3, we need:
    - 3 neighbors of u (call them u1,u2,u3)
    - 3 neighbors of v (call them v1,v2,v3)
    - u1,u2,u3 are non-adjacent to each other
    - v1,v2,v3 are non-adjacent to each other
    - Each ui is adjacent to exactly 2 of vj (specifically, the Q3 structure)
    
    This is still complex. Let's just do a full check for Q3 whenever we add an edge.
    
    Full check: iterate over all 8-subsets of vertices. For each, check if the induced subgraph contains Q3.
    To check if an 8-vertex graph contains Q3, we can try all 8! = 40320 permutations of Q3 vertices and see if any is an isomorphism.
    12870 * 40320 ≈ 5e8, which is too slow.
    
    Let's optimize: for an 8-vertex graph, we can check if it contains Q3 by:
    1. Check if it's bipartite. If not, no Q3.
    2. If bipartite, check if the bipartition has parts of size 4 and 4.
    3. Check if the degrees match Q3 (all degree 3).
    4. If so, check if it's exactly Q3 (since Q3 is the unique 3-regular bipartite graph on 8 vertices? No, there are others, but Q3 is the only one that's a cube).
    
    Actually, Q3 is the only 3-regular bipartite graph on 8 vertices that is a cube. But there might be other 3-regular bipartite graphs on 8 vertices.
    Wait, the number of 3-regular bipartite graphs on 4+4 vertices is small.
    
    Let's just check if the 8-vertex subgraph is isomorphic to Q3 by comparing edge sets after relabeling.
    
    Given the complexity, let's try a different approach:
    
    Use a known result or a smarter construction.
    
    According to the task, ex(16; Q3) is not documented.
    
    Let's try to use a random greedy algorithm:
    1. Start with an empty graph.
    2. Randomly order all possible edges.
    3. For each edge, add it if it doesn't create a Q3.
    
    To check if adding an edge creates a Q3, we can maintain a list of Q3s and update it.
    
    But implementing this efficiently is hard.
    
    Let's try a simpler heuristic:
    Start with K_{3,13}.
    Try to add edges within the 13-part in a random order.
    For each edge, check if it creates a Q3 by checking all 6-tuples of other vertices that could complete a Q3 with this edge.
    
    For edge (u,v), a Q3 containing (u,v) would have 6 other vertices.
    The 6 other vertices must be chosen from the remaining 14 vertices.
    C(14,6) = 3003, which is manageable.
    For each 6-tuple, check if {u,v} + 6-tuple forms a Q3.
    
    Checking if a specific 8-vertex set forms a Q3:
    - Check if the induced subgraph on these 8 vertices contains Q3.
    - We can do this by checking all 8! permutations, but that's 40320 per 8-set, and 3003 8-sets, so ~1.2e8 per edge addition, which is too slow.
    
    We need a faster Q3 check.
    
    Let's precompute the Q3 structure.
    Q3 has vertices 0..7, edges:
    (0,1), (0,2), (0,4),
    (1,3), (1,5),
    (2,3), (2,6),
    (3,7),
    (4,5), (4,6),
    (5,7),
    (6,7)
    Wait, let's list them properly.
    Q3 vertices are binary triples.
    000, 001, 010, 011, 100, 101, 110, 111
    Edges connect vertices differing in one bit.
    
    Let's label:
    0: 000
    1: 001
    2: 010
    3: 011
    4: 100
    5: 101
    6: 110
    7: 111
    
    Edges:
    0-1, 0-2, 0-4
    1-3, 1-5
    2-3, 2-6
    3-7
    4-5, 4-6
    5-7
    6-7
    
    Total 12 edges.
    
    To check if a set of 8 vertices with a given adjacency matrix contains Q3, we can:
    1. Check if the graph is 3-regular. If not, no.
    2. Check if it's bipartite. If not, no.
    3. If it's 3-regular and bipartite, check if it's isomorphic to Q3.
       Since Q3 is the only 3-regular bipartite graph on 8 vertices with girth 4 and diameter 3?
       Actually, there is only one 3-regular bipartite graph on 8 vertices up to isomorphism: Q3.
       Wait, is that true?
       The number of 3-regular bipartite graphs on 4+4 vertices is 1? Let's check.
       Yes, I believe Q3 is the unique 3-regular bipartite graph on 8 vertices.
       
    So, the check is:
    - Is the 8-vertex induced subgraph 3-regular?
    - Is it bipartite?
    If yes to both, it's Q3.
    
    This is much faster!
    
    So, algorithm:
    1. Start with K_{3,13}.
    2. Generate all possible edges within the 13-part.
    3. Shuffle them.
    4. For each edge, try to add it.
    5. To check if adding the edge creates a Q3:
       - For each 6-tuple of other vertices, form the 8-set {u,v} + 6-tuple.
       - Check if the induced subgraph on this 8-set is 3-regular and bipartite.
       - If any such 8-set is Q3, don't add the edge.
    6. Otherwise, add the edge.
    
    This should be feasible.
    
    Let's implement this.
    
    """
    import random
    
    n = 16
    # Start with K_{3,13}
    # Part 0: vertices 0,1,2
    # Part 1: vertices 3..15
    edges = set()
    for a in range(3):
        for b in range(3, 16):
            edges.add((min(a,b), max(a,b)))
    
    # Generate all possible edges within part 1
    intra_edges = []
    for i in range(3, 16):
        for j in range(i+1, 16):
            intra_edges.append((i,j))
    
    random.shuffle(intra_edges)
    
    # Helper function to check if a set of 8 vertices induces a Q3
    def is_q3(vertices, edge_set):
        if len(vertices) != 8:
            return False
        # Check 3-regular
        for v in vertices:
            deg = 0
            for u in vertices:
                if u != v and (min(u,v), max(u,v)) in edge_set:
                    deg += 1
            if deg != 3:
                return False
        # Check bipartite
        # BFS/DFS to 2-color
        color = {}
        for v in vertices:
            color[v] = -1
        for v in vertices:
            if color[v] == -1:
                color[v] = 0
                queue = [v]
                while queue:
                    u = queue.pop(0)
                    for w in vertices:
                        if w != u and (min(u,w), max(u,w)) in edge_set:
                            if color[w] == -1:
                                color[w] = 1 - color[u]
                                queue.append(w)
                            elif color[w] == color[u]:
                                return False
        return True
    
    # Try to add intra-part edges
    for u, v in intra_edges:
        edge = (min(u,v), max(u,v))
        if edge in edges:
            continue
        # Check if adding this edge creates a Q3
        creates_q3 = False
        # We need to check all 6-tuples from the remaining 14 vertices
        remaining = [x for x in range(16) if x != u and x != v]
        for combo in itertools.combinations(remaining, 6):
            vertices = [u, v] + list(combo)
            # Temporarily add the edge
            edges.add(edge)
            if is_q3(vertices, edges):
                creates_q3 = True
                edges.remove(edge)
                break
            edges.remove(edge)
        if not creates_q3:
            edges.add(edge)
    
    return [list(e) for e in edges]

# EVOLVE-BLOCK-END
