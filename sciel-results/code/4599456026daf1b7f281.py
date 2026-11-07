"""
Optimized construction for the few cycle lengths task.
The graph consists of two disjoint K_{3,3} components.
This achieves the theoretical minimum of 2 distinct cycle lengths ({4, 6}).
"""

def construct():
    """
    Returns a list of edges for an undirected simple graph on vertices 0..11.
    Constraints:
    - Minimum degree >= 3
    - Girth >= 4 (triangle-free)
    - Minimize distinct simple cycle lengths
    
    Solution:
    Two disjoint K_{3,3} graphs.
    K_{3,3} has min degree 3, girth 4, and cycle lengths {4, 6}.
    """
    edges = []
    
    # Helper to add complete bipartite graph edges
    def add_k33(left_part, right_part):
        for u in left_part:
            for v in right_part:
                edges.append([int(u), int(v)])
                
    # Component 1: Vertices 0-5
    left_1 = [0, 1, 2]
    right_1 = [3, 4, 5]
    add_k33(left_1, right_1)
    
    # Component 2: Vertices 6-11
    left_2 = [6, 7, 8]
    right_2 = [9, 10, 11]
    add_k33(left_2, right_2)
    
    return edges
