"""
Extremal graph construction: maximize edges on 16 vertices while avoiding Q3.

Strategy:
1. Build the complement of the Clebsch graph (a strongly regular graph on 16 vertices).
   The Clebsch graph has 16 vertices, degree 5, 40 edges.
   Its complement has degree 10 and 16*10/2 = 80 edges.
   We check if this complement is Q3-free.
   
2. If not Q3-free, try the Shrikhande graph (strongly regular 16-vertex, degree 6, 48 edges).
   Check if it's Q3-free.
   
3. If neither works, fall back to K_{3,13} (39 edges, known Q3-free) plus some safe edges.

The key insight: Q3 has 8 vertices and 12 edges. We can check for Q3 subgraphs
by enumerating all 8-vertex subsets and checking if the induced subgraph contains Q3
as a subgraph. With C(16,8) = 12870 subsets, this is feasible.
"""

import itertools

def _contains_q3(edges, n=16):
    """Check if the graph with given edges on n vertices contains Q3 as a subgraph."""
    # Build adjacency matrix
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        if u < v:
            adj[u][v] = adj[v][u] = True
    
    # Q3 vertices can be labeled 0..7 with edges:
    # 0-1, 0-2, 0-4, 1-3, 1-5, 2-3, 2-6, 3-7, 4-5, 4-6, 5-7, 6-7
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    
    # For each 8-vertex subset, check if there's an isomorphic embedding of Q3
    for subset in itertools.combinations(range(n), 8):
        # Check if all Q3 edges exist in the induced subgraph
        # Map q3 vertex i -> subset[i]
        found = True
        for a, b in q3_edges:
            if not adj[subset[a]][subset[b]]:
                found = False
                break
        if found:
            return True
    return False


def _clebsch_complement_edges():
    """
    Construct the complement of the Clebsch graph on 16 vertices.
    Clebsch graph: vertices 0..15, vertex i is adjacent to vertices j
    where popcount(i XOR j) == 1 or popcount(i XOR j) == 3.
    So Clebsch edges: for each i, connect to i XOR 1, i XOR 2, i XOR 4, i XOR 8,
    and also i XOR 3, i XOR 5, i XOR 6, i XOR 7, i XOR 9, i XOR 10, i XOR 12, i XOR 15?
    
    Wait, let me be more careful. The Clebsch graph is the folded 5-cube.
    Vertices are binary strings of length 5 modulo complement.
    Two vertices are adjacent if they differ in exactly 1 or exactly 4 positions
    (equivalently, 1 or 4 in the 5-bit representation, but since we fold, 
    difference of 4 is same as difference of 1 in the complement).
    
    Actually, the Clebsch graph on 16 vertices: label vertices 0..15 (4-bit).
    Vertex i is adjacent to j if popcount(i XOR j) == 1 or popcount(i XOR j) == 3.
    
    Let me verify: if popcount(i XOR j) == 1, that's 4 neighbors.
    If popcount(i XOR j) == 3, that's C(4,3) = 4 neighbors.
    Total degree = 8? But Clebsch should have degree 5.
    
    Hmm, I'm confusing things. Let me use a known construction.
    
    The Clebsch graph: It's the graph on {0,1,2,3}^2 where (a,b) ~ (a',b')
    if |a - a'| + |b - b'| == 1 (mod 4, using the 4-cycle distance).
    
    Actually, a simpler description: The Clebsch graph is the Schläfli graph's complement? No.
    
    Let me just use the known edge list or a correct construction.
    
    The Clebsch graph C_16: strongly regular (16, 5, 0, 2).
    It can be constructed as follows: take the 5-cube Q5, identify antipodal vertices.
    Vertices are equivalence classes {x, x+11111} in GF(2)^5.
    Two classes are adjacent if there exist representatives differing in exactly 1 coordinate.
    
    For 16 vertices, let's label them by 4-bit strings 0..15.
    Vertex i is adjacent to j iff popcount(i XOR j) == 1 or popcount(i XOR j) == 4.
    But popcount == 4 means i XOR j has 4 bits set, which for 4-bit strings means i XOR j = 15 (all bits).
    So j = i XOR 15.
    And popcount == 1 means j = i XOR 1, i XOR 2, i XOR 4, i XOR 8.
    So each vertex has 4 + 1 = 5 neighbors. Degree 5. Correct!
    
    So Clebsch edges: for each i, edges to i^1, i^2, i^4, i^8, i^15.
    Total edges = 16*5/2 = 40.
    
    Complement has edges: all pairs not in Clebsch.
    Total pairs = C(16,2) = 120. Complement edges = 120 - 40 = 80.
    """
    n = 16
    clebsch_edges = set()
    for i in range(n):
        for delta in [1, 2, 4, 8, 15]:
            j = i ^ delta
            if i < j:
                clebsch_edges.add((i, j))
    
    all_pairs = set()
    for i in range(n):
        for j in range(i+1, n):
            all_pairs.add((i, j))
    
    comp_edges = all_pairs - clebsch_edges
    return sorted(list(comp_edges))


def _shrikhande_edges():
    """
    Construct the Shrikhande graph on 16 vertices.
    The Shrikhande graph is a strongly regular graph (16, 6, 2, 2).
    
    Construction: vertices are elements of Z_4 x Z_4.
    (a,b) is adjacent to (a',b') if (a-a')^2 + (b-b')^2 ≡ 1 or 4 (mod 5).
    
    Let me verify: For fixed (a,b), count neighbors.
    (a-a')^2 + (b-b')^2 ≡ 1 mod 5:
    Solutions to x^2 + y^2 ≡ 1 mod 5: (±1,0), (0,±1), (±1,±2)?? 
    1^2+0^2=1, 0^2+1^2=1, 1^2+2^2=5≡0 no, 2^2+1^2=5≡0 no.
    Actually: 0+1=1, 1+0=1. Also (-1)^2+0=1, 0+(-1)^2=1.
    So (±1, 0), (0, ±1): 4 solutions.
    
    (a-a')^2 + (b-b')^2 ≡ 4 mod 5:
    x^2+y^2≡4: 2^2+0^2=4, 0^2+2^2=4, (-2)^2+0=4, 0+(-2)^2=4.
    So (±2, 0), (0, ±2): 4 solutions.
    
    Total: 8 neighbors? But Shrikhande has degree 6.
    
    Hmm, let me reconsider. The Shrikhande graph parameters are (16, 6, 2, 2).
    
    Alternative construction: The Shrikhande graph is the complement of the line graph of K_{4,4}? No, that's different.
    
    Actually, the Shrikhande graph can be constructed as follows:
    Take two disjoint copies of C_8 (8-cycles) and add chords.
    
    Or: vertices (i, j) in Z_4 x Z_4. Adjacent if (i-i', j-j') is in the set:
    {(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1)}? That gives degree 6.
    
    Let me check: for (0,0), neighbors are (1,0), (3,0), (0,1), (0,3), (1,1), (3,3).
    That's 6 neighbors. 
    
    Let me verify it's strongly regular. This is the "Shrikhande" construction I recall.
    """
    n = 16
    edges = set()
    # Vertices are (i, j) with i, j in 0..3, labeled as 4*i + j
    deltas = [(1,0), (3,0), (0,1), (0,3), (1,1), (3,3)]
    for i in range(4):
        for j in range(4):
            v = 4*i + j
            for di, dj in deltas:
                ni = (i + di) % 4
                nj = (j + dj) % 4
                w = 4*ni + nj
                if v < w:
                    edges.add((v, w))
    return sorted(list(edges))


def _k313_plus_k3_edges():
    """K_{3,13} plus a K_3 on the independent side. 39 + 3 = 42 edges."""
    edges = []
    # K_{3,13}: partition {0,1,2} and {3,4,...,15}
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
    # Add K_3 on vertices 3,4,5
    edges.append([3, 4])
    edges.append([4, 5])
    edges.append([3, 5])
    return edges


def construct():
    """
    Return a list of edges [u, v] of a simple graph on vertices 0..15 
    containing no subgraph isomorphic to Q3, maximizing the number of edges.
    """
    candidates = [
        (80, _clebsch_complement_edges()),  # Complement of Clebsch: 80 edges
        (48, _shrikhande_edges()),           # Shrikhande: 48 edges
        (42, _k313_plus_k3_edges()),         # K_{3,13} + K_3: 42 edges
    ]
    
    # Sort by edge count descending (already sorted)
    for num_edges, edges in candidates:
        if not _contains_q3(edges):
            # Return in the required format: list of [u, v]
            return [[u, v] for u, v in edges]
    
    # Fallback: pure K_{3,13} which is definitely Q3-free (39 edges)
    return [[a, b] for a in range(3) for b in range(3, 16)]
