"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    edges = []
    
    # First K_{3,3} on vertices 0..5
    # Partitions: A={0,1,2}, B={3,4,5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6..11
    # Partitions: A={6,7,8}, B={9,10,11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # Sort edges for consistency (optional but good practice)
    edges.sort()
    return edges

# EVOLVE-BLOCK-END
