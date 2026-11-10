"""
Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_{3,3} components.
This ensures:
1. Minimum degree is 3.
2. The graph is triangle-free (bipartite).
3. The graph contains cycles.
4. The distinct cycle lengths are {4, 6}, so the raw value is 2.
"""

def construct():
    edges = []
    
    # First K_{3,3} on vertices 0..5
    # Partitions: {0, 1, 2} and {3, 4, 5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6..11
    # Partitions: {6, 7, 8} and {9, 10, 11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # Ensure types are exactly Python int and list
    # The loop variables are Python ints, and append creates lists of ints.
    return edges
