import time

"""
Constructs a graph consisting of two disjoint copies of the complete bipartite graph K_3,3.
Each K_3,3 is 3-regular, bipartite (girth 4), and contains only cycles of length 4 and 6.
Thus, the total graph has distinct cycle lengths {4, 6}, so the raw value is 2.
"""

def construct():
    # Define the two partitions for the first K_3,3 component (vertices 0..5)
    # Left partition: 0, 1, 2
    # Right partition: 3, 4, 5
    edges_part1 = []
    left_nodes_1 = [0, 1, 2]
    right_nodes_1 = [3, 4, 5]
    for u in left_nodes_1:
        for v in right_nodes_1:
            edges_part1.append([u, v])

    # Define the two partitions for the second K_3,3 component (vertices 6..11)
    # Left partition: 6, 7, 8
    # Right partition: 9, 10, 11
    edges_part2 = []
    left_nodes_2 = [6, 7, 8]
    right_nodes_2 = [9, 10, 11]
    for u in left_nodes_2:
        for v in right_nodes_2:
            edges_part2.append([u, v])

    # Combine edges from both components
    all_edges = edges_part1 + edges_part2
    
    # Sort edges to ensure deterministic output format
    all_edges.sort()
    
    return all_edges
