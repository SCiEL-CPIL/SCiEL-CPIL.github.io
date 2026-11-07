"""
Constructs a dense graph with diameter 2 where every edge is essential.

The problem requires a graph on 16 vertices with:
1. Diameter exactly 2.
2. Every edge is essential: removing any single edge increases the diameter to > 2 or disconnects the graph.
3. Maximize the number of edges.

Analysis:
- Condition 2 implies that for every edge (u, v), there is no path of length 2 between u and v.
  If such a path existed (u-w-v), removing (u,v) would leave dist(u,v) = 2.
  Since distances in a graph can only increase or stay the same upon edge removal,
  if the original diameter was 2, the new diameter would remain 2 (unless some other pair
  distance increases to 3, but the condition requires the diameter to increase *above* 2).
  Actually, if dist(u,v) stays 2, and all other distances stay <= 2, the new diameter is 2.
  This violates "increase its diameter above two".
  Thus, no edge can have a common neighbor. The graph must be triangle-free.

- By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n vertices
  is floor(n^2 / 4). For n=16, this is 64.
- The unique extremal graph is the complete bipartite graph K_{8,8}.
- K_{8,8} has diameter 2 (vertices in same partition are dist 2, different are dist 1).
- Therefore, K_{8,8} is the unique optimal solution.
"""

import time

def construct():
    """
    Returns the edges of the complete bipartite graph K_{8,8} on 16 vertices.
    This graph has 64 edges, diameter 2, and is triangle-free, making every edge essential.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 16
    half = n // 2
    
    edges = []
    # Partition vertices into 0..7 and 8..15
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
            
    # Check time just in case, though this is O(n^2) and very fast
    if time.time() > deadline:
        # Fallback: return empty list if somehow we are close to timeout (should never happen)
        return []
        
    return edges
