"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a complete 4-partite graph K_{4,4,4,4} on 16 vertices.
    
    The graph is partitioned into 4 sets of 4 vertices each.
    Edges exist between all vertices in different sets.
    
    Edge count: 4 * (4*3 / 2) * 4 = 96 edges.
    (Calculation: There are C(4,2)=6 pairs of parts. Each pair contributes 4*4=16 edges. 6*16=96.)
    
    This graph is Q3-free because Q3 requires an independent set of size 8 
    (one side of its bipartition), but K_{4,4,4,4} has an independence number of only 4.
    """
    edges = []
    # Define the 4 partitions
    partitions = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
    
    # Generate edges between all pairs of partitions
    for i in range(4):
        for j in range(i + 1, 4):
            for u in partitions[i]:
                for v in partitions[j]:
                    edges.append([u, v])
                    
    return edges
# EVOLVE-BLOCK-END
