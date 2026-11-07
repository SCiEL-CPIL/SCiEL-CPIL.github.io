"""
Constructs a graph on 12 vertices with minimum degree 3 and girth at least 4.
The goal is to minimize the number of distinct simple cycle lengths.
The two disjoint K_{3,3} components achieve cycle lengths {4, 6}, so the raw value is 2.
This is currently the best known value for n=12 with min degree 3.
"""

def construct():
    # Vertices are 0..11
    # Component 1: K_{3,3} on vertices 0,1,2 and 3,4,5
    # Component 2: K_{3,3} on vertices 6,7,8 and 9,10,11
    
    edges = []
    
    # Helper to add edges for a K_{3,3} component
    def add_k33(left_verts, right_verts):
        for u in left_verts:
            for v in right_verts:
                edges.append([u, v])
                
    # Define the two components
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    
    add_k33(left1, right1)
    add_k33(left2, right2)
    
    # The resulting graph has:
    # - 12 vertices
    # - Minimum degree 3
    # - Girth 4 (bipartite)
    # - Cycle lengths: 4 and 6
    # - Distinct cycle lengths: 2
    
    return edges
