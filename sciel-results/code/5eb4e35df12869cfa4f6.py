"""
Constructs a diameter-2 graph on n=16 vertices with maximum edges such that
every edge is essential for maintaining diameter 2.

Key Insight:
The task requires a graph G on n=16 vertices such that:
1. Diameter(G) = 2.
2. For every edge e in E(G), Diameter(G - e) > 2 or G - e is disconnected.
3. Maximize |E(G)|.

Analysis:
- Condition 2 implies that for any edge (u, v), there is no path of length 2 between u and v in G.
  If there were a vertex w such that (u, w) and (w, v) are edges, then in G - (u, v), the distance d(u, v) would be 2.
  Since the graph is connected and dense, removing one edge likely keeps it connected, so the diameter would remain 2.
  Thus, G must be triangle-free.
- By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n vertices is floor(n^2 / 4).
  For n=16, this is 16^2 / 4 = 64.
- The unique extremal graph is the complete bipartite graph K_{8,8}.
- Verification for K_{8,8}:
  - It is triangle-free.
  - Diameter is 2: Vertices in the same partition are distance 2 apart (via any vertex in the other partition). Vertices in different partitions are distance 1 apart.
  - Edge essentiality: Consider an edge (u, v) where u is in partition A and v is in partition B.
    In K_{8,8} - (u, v), u and v are no longer adjacent.
    Is there a path of length 2? A path of length 2 from u to v would be u-w-v.
    Since u is in A, w must be in B. Since v is in B, w must be in A.
    This is a contradiction (w cannot be in both A and B).
    Therefore, no path of length 2 exists. The shortest path is of length 3 (e.g., u-b-a-v where b in B, a in A).
    Thus, the distance between u and v becomes 3.
    The diameter of the graph becomes 3 (since other pairs still have distance <= 2).
    So, the diameter increases from 2 to 3. The condition is satisfied.

Conclusion:
The optimal graph is K_{8,8} with 64 edges.
"""

# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    half = n // 2
    
    # Partition vertices into two sets of size 8
    # A = {0, ..., 7}, B = {8, ..., 15}
    
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
            
    return edges
# EVOLVE-BLOCK-END
