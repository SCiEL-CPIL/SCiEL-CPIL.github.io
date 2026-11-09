"""
Constructs the maximum edge count graph on 16 vertices with diameter 2
where every edge is critical.

Based on Mantel's Theorem, the maximum number of edges in a triangle-free
graph on n vertices is floor(n^2/4). For n=16, this is 64 edges, achieved
by the complete bipartite graph K_{8,8}.

In any triangle-free graph with diameter 2, every edge is critical because
removing an edge (u, v) increases the distance between u and v to at least 3
(since there is no common neighbor to provide a path of length 2), thus
increasing the diameter of the graph to at least 3.

Any graph with more than 64 edges must contain a triangle. In a graph with
diameter 2 containing a triangle, the edges within the triangle typically
have alternative paths of length 2, making them non-critical for increasing
the diameter above 2 upon removal. Thus, K_{8,8} is the optimal solution.
"""

def construct():
    # Vertices 0 to 15
    # Partition A: 0 to 7
    # Partition B: 8 to 15
    edges = []
    
    # Generate all edges between Partition A and Partition B
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    # Ensure types are strictly Python int and list
    # The loop above produces ints and lists natively.
    
    return edges
