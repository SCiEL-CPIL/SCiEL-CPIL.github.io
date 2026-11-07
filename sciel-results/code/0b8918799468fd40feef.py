"""
Rewritten program for the task: Few hyperedges forcing three colors.

The task is to find a 3-uniform hypergraph on 9 vertices that is 3-chromatic
(i.e., not 2-colorable) with the minimum number of edges.

It is a known mathematical fact that the Fano plane is the smallest 3-uniform
hypergraph that is not 2-colorable. The Fano plane has 7 vertices and 7 edges.
Any 3-uniform hypergraph with fewer than 7 edges is 2-colorable.
Therefore, the minimum number of edges for a 3-chromatic 3-uniform hypergraph
on any number of vertices is 7.

We construct a hypergraph on 9 vertices by taking the Fano plane on vertices
0..6 and leaving vertices 7 and 8 isolated. We provide a valid 3-coloring.
"""

def construct():
    """
    Returns a 3-uniform hypergraph on 9 vertices with 7 edges (Fano plane + 2 isolated vertices)
    which is not 2-colorable, and a valid 3-coloring.
    
    The Fano plane edges are defined on vertices 0-6.
    Vertices 7 and 8 are isolated.
    """
    # Fano plane edges on vertices 0-6
    # These 7 edges form the Fano plane, which is 3-chromatic.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Colors for vertices 0-8
    # We need a coloring with 3 colors (0, 1, 2) such that no edge is monochromatic.
    # The Fano plane is 3-chromatic, so such a coloring exists.
    # Let's assign:
    # c(0) = 0
    # c(1) = 1
    # c(2) = 2
    # c(3) = 1
    # c(4) = 1
    # c(5) = 2
    # c(6) = 2
    #
    # Verification:
    # {0,1,2}: 0,1,2 -> OK
    # {0,3,4}: 0,1,1 -> OK
    # {0,5,6}: 0,2,2 -> OK
    # {1,3,5}: 1,1,2 -> OK
    # {1,4,6}: 1,1,2 -> OK
    # {2,3,6}: 2,1,2 -> OK
    # {2,4,5}: 2,1,2 -> OK
    #
    # Vertices 7 and 8 are isolated, so their colors do not affect edge constraints.
    # We can assign them color 0.
    colors = [0, 1, 2, 1, 1, 2, 2, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }
