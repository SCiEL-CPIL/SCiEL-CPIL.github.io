"""
Constructs a graph on 16 vertices with diameter 2 such that removing any edge increases the diameter.
The optimal structure is the complete bipartite graph K_{8,8}, which achieves the
Mantel bound of 64 edges for triangle-free graphs on 16 vertices.
This is the maximum possible edge count for any graph satisfying the edge-criticality
constraint for diameter 2, as any graph with more edges would contain a triangle,
rendering at least one edge non-critical (removing it would not increase the diameter).
"""

def construct():
    n = 16
    edges = []
    
    # K_{8,8}: bipartition {0..7} and {8..15}
    # This graph is triangle-free and has diameter 2.
    # In K_{8,8}, every edge (u, v) with u in {0..7} and v in {8..15} is critical.
    # Removing (u, v) increases the distance between u and v to 3,
    # as there are no common neighbors (no triangles) and no alternative paths of length 2.
    # Thus, the diameter increases from 2 to 3.
    
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    return edges
