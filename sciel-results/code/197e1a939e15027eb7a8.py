import time
import random
import itertools

def construct():
    # Fallback: Fano plane (7 edges)
    # The Fano plane is the unique 3-uniform hypergraph on 7 vertices that is 3-chromatic.
    # We extend it to 9 vertices by adding 2 isolated vertices (7, 8) colored 0.
    # This ensures the hypergraph on 0..8 is 3-chromatic (since a 2-coloring of the 9 vertices
    # would restrict to a 2-coloring of the Fano plane, which is impossible).
    # It is 3-colorable by the Fano plane coloring extended with color 0 for the new vertices.
    
    edges = [
        [0, 1, 2], [0, 3, 4], [0, 5, 6],
        [1, 3, 5], [1, 4, 6], [2, 3, 6], [2, 4, 5]
    ]
    
    # Fano plane coloring:
    # 0:0, 1:1, 2:2
    # 3:0, 4:1, 5:2
    # 6:0
    # Check edges:
    # 0,1,2 -> 0,1,2 (ok)
    # 0,3,4 -> 0,0,1 (ok)
    # 0,5,6 -> 0,2,0 (ok)
    # 1,3,5 -> 1,0,2 (ok)
    # 1,4,6 -> 1,1,0 (ok)
    # 2,3,6 -> 2,0,0 (ok)
    # 2,4,5 -> 2,1,2 (ok)
    
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Ensure types are strictly Python int and list
    edges_out = []
    for e in edges:
        edges_out.append([int(v) for v in e])
    
    colors_out = [int(c) for c in colors]
    
    return {'edges': edges_out, 'colors': colors_out}
