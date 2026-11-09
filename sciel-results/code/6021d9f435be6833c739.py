"""
Constructs a graph on 12 vertices with minimum degree 3 and minimum distinct cycle lengths.
The graph is two disjoint copies of K_{3,3}.
K_{3,3} is 3-regular and bipartite (triangle-free).
The simple cycles in K_{3,3} have lengths 4 and 6.
Thus, the distinct cycle lengths are {4, 6}, count = 2.
This is the known minimum for this problem size and constraints.
"""

def construct():
    """
    Returns a list of edges [u, v] for a graph on vertices 0..11.
    The graph consists of two disjoint K_{3,3} components.
    """
    edges = []
    
    # First K_{3,3} on vertices 0..5
    # Partition A: 0, 1, 2
    # Partition B: 3, 4, 5
    for u in range(3):
        for v in range(3, 6):
            edges.append([int(u), int(v)])
            
    # Second K_{3,3} on vertices 6..11
    # Partition A: 6, 7, 8
    # Partition B: 9, 10, 11
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([int(u), int(v)])
            
    return edges
