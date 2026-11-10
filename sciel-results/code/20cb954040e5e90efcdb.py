"""
Constructs a graph with two disjoint K_{3,3} components.
This graph is 3-regular, bipartite, and triangle-free.
The distinct cycle lengths are {4, 6}, resulting in a raw value of 2.
"""

def construct():
    edges = []
    
    # Generate first K_{3,3} component on vertices 0-5
    # Bipartition sets: A1 = {0, 1, 2}, B1 = {3, 4, 5}
    for i in range(3):
        for j in range(3):
            u = i
            v = 3 + j
            edges.append([u, v])
            
    # Generate second K_{3,3} component on vertices 6-11
    # Bipartition sets: A2 = {6, 7, 8}, B2 = {9, 10, 11}
    for i in range(3):
        for j in range(3):
            u = 6 + i
            v = 9 + j
            edges.append([u, v])
            
    # Sort edges to ensure consistent output format
    # Ensure u < v within each edge
    processed_edges = []
    for e in edges:
        if e[0] > e[1]:
            processed_edges.append([e[1], e[0]])
        else:
            processed_edges.append([e[0], e[1]])
            
    processed_edges.sort()
    
    return processed_edges

# EVOLVE-BLOCK-START
# The construct function above is the complete solution.
# EVOLVE-BLOCK-END
