"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    """
    Construct a 3-uniform hypergraph on 9 vertices that is exactly 3-chromatic
    with the minimum number of edges.
    
    The Fano plane (7 edges on 7 vertices) is known to be the smallest 3-chromatic
    3-uniform hypergraph. We use the Fano plane on vertices 0-6 and leave vertices
    7-8 unconnected, which preserves 3-chromaticity (any 2-coloring of the Fano 
    plane is impossible, so the whole graph is also not 2-colorable).
    
    The Fano plane edges are:
    {0,1,2}, {0,3,4}, {0,5,6}, {1,3,5}, {1,4,6}, {2,3,6}, {2,4,5}
    
    A valid 3-coloring:
    0: 0, 1: 1, 2: 2
    3: 1, 4: 2, 5: 0
    6: 1 (wait, let me check)
    
    Let me verify a proper 3-coloring of the Fano plane:
    Standard Fano plane coloring:
    Vertices: 0,1,2,3,4,5,6
    Edges: (0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5)
    
    Try: 0->0, 1->1, 2->2
    Edge (0,1,2): 0,1,2 - ok
    Edge (0,3,4): need 3,4 not both 0. Say 3->1, 4->2
    Edge (0,5,6): need 5,6 not both 0. Say 5->1, 6->2
    Edge (1,3,5): 1,1,1 - monochromatic! Bad.
    
    Let me try another:
    0->0, 1->1, 2->2
    3->2, 4->1, 5->2, 6->1
    Edge (0,1,2): 0,1,2 ok
    Edge (0,3,4): 0,2,1 ok
    Edge (0,5,6): 0,2,1 ok
    Edge (1,3,5): 1,2,2 - not mono, ok
    Edge (1,4,6): 1,1,1 - mono! Bad.
    
    Try:
    0->0, 1->1, 2->2
    3->1, 4->2, 5->1, 6->2
    Edge (0,1,2): 0,1,2 ok
    Edge (0,3,4): 0,1,2 ok
    Edge (0,5,6): 0,1,2 ok
    Edge (1,3,5): 1,1,1 mono! Bad.
    
    Known 3-coloring of Fano plane:
    Actually the Fano plane is not 3-colorable in the sense that each edge has all 3 colors? No, it is 3-chromatic meaning chromatic number is 3. Let me find a valid 3-coloring.
    
    Fano plane vertices as elements of GF(2)^3 \ {0}:
    100, 010, 001, 110, 101, 011, 111
    
    Color by first nonzero bit position?
    100 -> color 0
    010 -> color 1
    001 -> color 2
    110 -> has bits 0,1. Color by... 
    
    Actually, a known fact: the Fano plane has chromatic number 3. A valid 3-coloring:
    Assign colors to the 7 points such that no line is monochromatic.
    
    Let's use:
    Point 1: 0
    Point 2: 0
    Point 3: 1
    Point 4: 1
    Point 5: 2
    Point 6: 2
    Point 7: 0
    
    Lines of Fano plane (using standard labeling):
    123, 145, 167, 246, 257, 347, 356
    
    With colors: 1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:0
    Line 123: 0,0,1 - ok
    Line 145: 0,1,2 - ok
    Line 167: 0,2,0 - ok
    Line 246: 0,1,2 - ok
    Line 257: 0,2,0 - ok
    Line 347: 1,1,0 - ok
    Line 356: 1,2,2 - ok
    
    Great! So mapping to our 0-indexed vertices 0-6:
    vertex 0 (point 1): color 0
    vertex 1 (point 2): color 0
    vertex 2 (point 3): color 1
    vertex 3 (point 4): color 1
    vertex 4 (point 5): color 2
    vertex 5 (point 6): color 2
    vertex 6 (point 7): color 0
    vertex 7: color 0
    vertex 8: color 0
    
    Edges (converting points 1-7 to vertices 0-6):
    123 -> (0,1,2)
    145 -> (0,3,4)
    167 -> (0,5,6)
    246 -> (1,3,5)
    257 -> (1,4,6)
    347 -> (2,3,6)
    356 -> (2,4,5)
    """
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    colors = [0, 0, 1, 1, 2, 2, 0, 0, 0]
    
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
