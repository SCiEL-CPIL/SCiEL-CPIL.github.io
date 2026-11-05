import sys
import time
import random
from itertools import combinations

def construct():
    """
    Build a Q3-free graph on 16 vertices maximizing edge count.
    Strategy:
    1. Start with a strong seed graph (Hadamard graph or similar Q3-free base).
    2. Use greedy edge addition with fast Q3-existence checks.
    3. The key insight: Q3 (cube) is 3-regular, bipartite, 8 vertices, 12 edges.
       A graph contains Q3 if there exist 8 vertices inducing a subgraph that contains Q3.
       Since Q3 is not complete, we need a more targeted check.
       
    Actually, let's think about this differently. The seed program uses K_{3,13} which has 39 edges.
    We can do much better. Known results: ex(n, Q3) for n=16.
    
    Let's use a constructive approach:
    - Start with a random or structured Q3-free graph
    - Try to add edges greedily while maintaining Q3-free property
    
    Fast Q3 check: A graph contains Q3 iff there exist 8 vertices {v0..v7} such that
    the induced subgraph on these vertices contains all 12 edges of Q3.
    But checking all subsets of 8 from 16 is C(16,8) = 12870, and for each checking if
    it contains Q3 as a subgraph (not necessarily induced) is expensive.
    
    Better approach: Precompute all possible Q3 embeddings.
    Q3 can be labeled as vertices (x,y,z) in {0,1}^3. Two vertices are adjacent if they differ in exactly 1 coordinate.
    
    For a given graph G, G contains Q3 iff there exist 8 distinct vertices v0..v7 and a bijection
    from {0,1}^3 to {v0..v7} such that for each pair of cube vertices differing in one coordinate,
    the corresponding graph vertices are adjacent in G.
    
    This is still hard to check quickly. Let's use a different approach:
    
    Since we're maximizing edges, let's use a known strong Q3-free graph on 16 vertices.
    
    The Hadamard graph H(8) on 16 vertices (from Hadamard matrix of order 16) is Q3-free?
    Actually, the Clebsch graph on 16 vertices is Q3-free and has 40 edges? No, Clebsch has 40 edges.
    
    Let me think: The seed has 39 edges (K_{3,13}). We need to do better.
    
    Approach: Use a branch-and-bound or greedy search with efficient Q3 detection.
    
    For Q3 detection in a graph with 16 vertices:
    - Enumerate all ways to map the 8 cube vertices to distinct graph vertices
    - But that's 16*15*14*...*9 which is too many
    
    Better: Use the fact that Q3 is bipartite with parts of size 4.
    In any Q3 embedding, the 8 vertices split into two sets of 4 (the bipartition).
    Within each part of the cube, no edges exist. Between parts, each vertex in one part
    connects to exactly 3 vertices in the other part.
    
    So: G contains Q3 iff there exist two disjoint 4-sets A and B such that:
    - No edges within A, no edges within B (in the subgraph induced by A∪B)
    - Each vertex in A has exactly 3 neighbors in B, and each vertex in B has exactly 3 neighbors in A
    
    Wait, that's not quite right. The condition is that the bipartite graph between A and B
    is exactly the cube's bipartite graph, which is the 4-regular... no, each vertex in the cube
    has degree 3. So the bipartite graph between the two parts of Q3 is 3-regular on each side,
    with 4 vertices per side, so 12 edges total.
    
    But it's not just any 3-regular bipartite graph on 4+4 vertices. There are multiple 3-regular
    bipartite graphs on 4+4 vertices. The cube's bipartite graph is specifically the one where
    the 4 vertices on each side form a specific structure.
    
    Actually, the bipartite graph of Q3 (splitting by parity of number of 1s) has:
    Side 0: vertices with even parity: (0,0,0), (0,1,1), (1,0,1), (1,1,0)
    Side 1: vertices with odd parity: (1,0,0), (0,1,0), (0,0,1), (1,1,1)
    
    The adjacency: each even vertex connects to 3 odd vertices. Specifically:
    (0,0,0) -> (1,0,0), (0,1,0), (0,0,1)
    (0,1,1) -> (1,1,1), (0,0,1), (0,1,0)
    (1,0,1) -> (1,1,1), (1,0,0), (0,0,1)
    (1,1,0) -> (1,1,1), (1,0,0), (0,1,0)
    
    So this is a specific 3-regular bipartite graph. Not all 3-regular bipartite graphs on 4+4
    are isomorphic to this one. For example, the complete bipartite K_{4,4} minus a perfect matching
    is also 3-regular on 4+4, but is it the same?
    
    K_{4,4} minus a perfect matching: each vertex has degree 3. This is actually isomorphic to
    the cube's bipartite graph! Let me verify. In K_{4,4} minus a perfect matching, label side A as
    a1,a2,a3,a4 and side B as b1,b2,b3,b4, with ai not connected to bi. Then:
    a1: b2,b3,b4
    a2: b1,b3,b4
    a3: b1,b2,b4
    a4: b1,b2,b3
    
    In the cube's bipartite graph:
    (0,0,0): (1,0,0),(0,1,0),(0,0,1)
    (0,1,1): (1,1,1),(0,0,1),(0,1,0)
    (1,0,1): (1,1,1),(1,0,0),(0,0,1)
    (1,1,0): (1,1,1),(1,0,0),(0,1,0)
    
    Let me map: a1=(0,0,0), a2=(0,1,1), a3=(1,0,1), a4=(1,1,0)
    b1=(1,1,1), b2=(1,0,0), b3=(0,1,0), b4=(0,0,1)
    
    a1=(0,0,0) should connect to b2=(1,0,0), b3=(0,1,0), b4=(0,0,1). Yes.
    a2=(0,1,1) should connect to b1=(1,1,1), b3=(0,1,0), b4=(0,0,1). Yes.
    a3=(1,0,1) should connect to b1=(1,1,1), b2=(1,0,0), b4=(0,0,1). Yes.
    a4=(1,1,0) should connect to b1=(1,1,1), b2=(1,0,0), b3=(0,1,0). Yes.
    
    So the cube's bipartite graph IS K_{4,4} minus a perfect matching!
    
    So: G contains Q3 iff there exist two disjoint 4-sets A and B such that:
    1. The bipartite graph between A and B is K_{4,4} minus a perfect matching
    2. (No condition on edges within A or B, since Q3 has no edges within its bipartition parts)
    
    Wait, but Q3 has no edges within the same part of its bipartition. So if G has edges within A or within B,
    that doesn't create a Q3 as long as the bipartite part between A and B is exactly K_{4,4} minus a perfect matching.
    But we also need that there are no EXTRA edges between A and B beyond the 12 edges of the matching complement.
    Since K_{4,4} has 16 edges and we remove 4 (the perfect matching), we have 12 edges. If G has any additional
    edge between A and B, then the bipartite graph is not exactly K_{4,4} minus a perfect matching.
    
    So the condition is: there exist disjoint 4-sets A, B and a perfect matching M between A and B such that
    for every pair (a,b) in A×B, a~b in G iff (a,b) is NOT in M.
    
    This is a strong condition. Let's use this for checking.
    
    For a given graph G on 16 vertices, to check if it contains Q3:
    - For each pair of disjoint 4-sets (A,B) (there are C(16,4)*C(12,4)/2 = 1820*495/2 ≈ 450,000 pairs, which is a lot)
    - Check if the bipartite graph between A and B is K_{4,4} minus some perfect matching
    
    This is still expensive. Let's think of a faster way.
    
    Alternative: Since we're adding edges one at a time, we can check incrementally.
    When we add an edge (u,v), we only need to check if this edge completes a Q3.
    A Q3 containing edge (u,v) must have u and v adjacent in the cube. In the cube, each edge
    is in exactly 2 faces (squares), and the cube has 12 edges. Each edge is shared by 2 faces.
    For a given edge (u,v) in the cube, how many distinct cubes contain this edge? The cube is rigid,
    so once we fix edge (u,v), the rest of the cube is determined up to a few choices.
    
    Actually, let's just precompute all 8-vertex subsets that form Q3 patterns and check.
    But there are C(16,8) = 12870 subsets, and for each we need to check if it contains Q3 as a subgraph.
    For a given 8-vertex set, checking if it contains Q3 is: does there exist a labeling of the 8 vertices
    as cube vertices such that all 12 cube edges are present? This is an isomorphism check, which is
    expensive to do naively.
    
    Given the 120-second limit, let's use a practical approach:
    1. Start with a good seed (e.g., the Clebsch graph or a random Q3-free graph)
    2. Greedily add edges, checking Q3-freeness by enumerating all 8-subsets and checking
       if any contains Q3 as a subgraph
    
    For checking if an 8-vertex induced subgraph contains Q3:
    - The 8 vertices can be labeled in 8! = 40320 ways, but we can use the bipartite structure
    - Q3 is bipartite with two parts of 4. So we need to split the 8 vertices into two 4-sets
      and check if the bipartite graph is K_{4,4} minus a perfect matching.
    - There are C(8,4) = 70 ways to split into two 4-sets. For each split, check if the bipartite
      graph is K_{4,4} minus a perfect matching.
    
    To check if a bipartite graph on 4+4 vertices is K_{4,4} minus a perfect matching:
    - It must have exactly 12 edges
    - Each vertex must have degree 3
    - The non-edges must form a perfect matching (4 non-edges, each vertex incident to exactly 1 non-edge)
    
    This is O(1) for a given split!
    
    So for each 8-subset: 70 splits * O(1) check = O(70) per subset.
    Total: 12870 * 70 ≈ 900,000 checks. This is feasible in Python if optimized.
    
    Let's implement this.
    """
    n = 16
    # Precompute all 8-subsets
    from itertools import combinations as comb
    all_8_subsets = list(comb(range(n), 8))
    # Precompute all 4-subsets of an 8-subset
    # For each 8-subset, precompute its 4-subsets
    # Actually, let's just compute on the fly with bitmasks for speed
    
    # Represent the graph as a bitset for each vertex
    # adj[i] is a 16-bit integer where bit j is 1 if i~j
    adj = [0] * n
    
    def add_edge(u, v):
        if u == v:
            return
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def get_edge_count():
        return sum(bin(a).count('1') for a in adj) // 2
    
    def has_q3():
        """Check if the current graph contains Q3."""
        for subset in all_8_subsets:
            # For each way to split into two 4-sets
            # Precompute 4-subsets of this 8-subset
            s = set(subset)
            # Generate all 4-subsets of s
            four_sets = list(comb(s, 4))
            for i in range(len(four_sets)):
                for j in range(i+1, len(four_sets)):
                    A = four_sets[i]
                    B = four_sets[j]
                    if A & B:  # They should be disjoint
                        continue
                    if len(A) != 4 or len(B) != 4:
                        continue
                    # Check if the bipartite graph between A and B is K_{4,4} minus a perfect matching
                    # Count edges and non-edges
                    non_edges = []
                    for a in A:
                        for b in B:
                            if not (adj[a] & (1 << b)):
                                non_edges.append((a, b))
                    # For K_{4,4} minus a perfect matching: exactly 4 non-edges, each vertex in A
                    # has exactly 1 non-edge, each vertex in B has exactly 1 non-edge
                    if len(non_edges) != 4:
                        continue
                    # Check if non-edges form a perfect matching
                    deg_a = {a: 0 for a in A}
                    deg_b = {b: 0 for b in B}
                    for (a, b) in non_edges:
                        deg_a[a] += 1
                        deg_b[b] += 1
                    if all(d == 1 for d in deg_a.values()) and all(d == 1 for d in deg_b.values()):
                        return True
        return False
    
    # Start with a strong seed
    # Use the Clebsch graph or similar
    # Let's start with a random graph and greedily add edges
    # Or start with K_{3,13} and improve
    
    # Initialize with K_{3,13}
    for a in range(3):
        for b in range(3, 16):
            add_edge(a, b)
    
    # List of all possible edges not yet in the graph
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & (1 << j)):
                all_edges.append((i, j))
    
    # Greedy: try to add edges in random order, checking Q3-freeness
    random.seed(42)
    random.shuffle(all_edges)
    
    for (u, v) in all_edges:
        # Try adding edge (u,v)
        add_edge(u, v)
        if has_q3():
            remove_edge(u, v)
        # Keep it if no Q3
    
    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges

if __name__ == "__main__":
    start = time.time()
    result = construct()
    print(f"Edges: {len(result)}")
    print(f"Time: {time.time() - start:.2f}s")
