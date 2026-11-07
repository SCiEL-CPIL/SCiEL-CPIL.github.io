"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    """
    Construct a graph on 16 vertices with diameter 2 where every edge is essential.
    
    Analysis:
    1. The condition "every edge is essential for diameter 2" is equivalent to the graph being triangle-free.
       - If an edge (u,v) is in a triangle (u,v,w), then removing (u,v) leaves path u-w-v of length 2.
         The distance between u and v remains 2. The diameter of the graph remains 2 (or less, but it was 2).
         So the edge was not essential for maintaining diameter 2 (it didn't increase the diameter).
       - Conversely, if the graph is triangle-free, removing edge (u,v) means u and v have no common neighbors.
         Thus, no path of length 2 exists between u and v.
         The distance becomes >= 3 or infinity.
         This increases the diameter (which was 2) to > 2.
       - So we need a Triangle-Free graph with Diameter 2.
    
    2. We want to maximize the number of edges.
       - By Turán's Theorem, the maximum number of edges in a triangle-free graph on n vertices is floor(n^2/4).
       - For n=16, max edges = 64.
       - The unique graph achieving this is the complete bipartite graph K_{8,8}.
    
    3. Does K_{8,8} have diameter 2?
       - Vertices in different partitions are adjacent (dist 1).
       - Vertices in the same partition share all vertices in the other partition as common neighbors (dist 2).
       - So diameter is exactly 2.
    
    4. Conclusion: K_{8,8} is the optimal solution.
    """
    n = P['n']
    half = n // 2
    
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
            
    return edges

# EVOLVE-BLOCK-END
