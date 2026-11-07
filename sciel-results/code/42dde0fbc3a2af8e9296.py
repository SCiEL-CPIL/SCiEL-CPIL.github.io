"""
Constructs a diameter-2 graph on n=16 vertices with maximum edges such that
every edge is essential for maintaining diameter 2.

Strategy:
1. Analyze the constraints:
   - Diameter 2.
   - Every edge essential: removing any edge (u,v) must make d(u,v) >= 3 or disconnect.
   - This implies no edge (u,v) can have a common neighbor w in G, because if it did,
     d_{G-(u,v)}(u,v) would be 2 (via u-w-v). If the graph is dense, global diameter stays 2.
   - Thus, G must be triangle-free.
2. Maximize edges in a triangle-free graph on n=16 vertices.
   - By Mantel's Theorem, max edges is floor(n^2/4) = 64.
   - The unique extremal graph is the complete bipartite graph K_{8,8}.
3. Verify K_{8,8}:
   - Triangle-free: Yes.
   - Diameter 2: Yes (dist 1 between parts, dist 2 within parts).
   - Edge essentiality: Removing (u,v) with u in A, v in B removes the only link between u and v.
     No common neighbors exist (N(u) subset B, N(v) subset A, disjoint).
     Next shortest path is length 3 (u-b-a-v). Diameter becomes 3. Condition met.
4. Implementation:
   - Instead of directly generating bipartite edges, we define the structure via its complement.
   - The complement of K_{8,8} is two disjoint cliques K_8 and K_8.
   - We construct the edge set by taking all possible pairs and excluding those in the complement.
   - This is a structural reformulation of the same optimal object.
"""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 60  # 60s limit, keep margin

    # The optimal graph is K_{n/2, n/2}.
    # Its complement consists of two disjoint cliques of size n/2.
    # Let A = {0, ..., 7} and B = {8, ..., 15}.
    # Non-edges are pairs within A and pairs within B.
    
    half = n // 2
    set_a = set(range(half))
    set_b = set(range(half, n))
    
    # Define the set of non-edges (edges in the complement graph)
    # These are edges that MUST NOT be in G.
    non_edges = set()
    # Pairs within set_a
    for i in range(half):
        for j in range(i + 1, half):
            non_edges.add((i, j))
    # Pairs within set_b
    for i in range(half, n):
        for j in range(i + 1, n):
            non_edges.add((i, j))
            
    edges = []
    # Generate all possible edges and filter out non-edges
    for u in range(n):
        for v in range(u + 1, n):
            if time.time() > deadline - 5:
                # Safety exit, though generation is O(n^2) and very fast
                break
            if (u, v) not in non_edges:
                edges.append([int(u), int(v)])
        
        if time.time() > deadline - 5:
            break

    return edges
# EVOLVE-BLOCK-END
