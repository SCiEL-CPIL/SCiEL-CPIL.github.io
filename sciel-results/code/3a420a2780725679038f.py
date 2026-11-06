"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Build Kneser graph KG(6,2): vertices are 2-subsets of {0,1,2,3,4,5}
    # Two vertices are adjacent iff the subsets are disjoint.
    # This gives 15 vertices, 45 edges, triangle-free, chromatic number 4.
    # Add vertex 15 connected to a maximum independent set of KG(6,2) (size 3).
    # Total: 16 vertices, 48 edges.
    
    # Generate all 2-subsets of {0,1,2,3,4,5}
    subsets = []
    for i in range(6):
        for j in range(i + 1, 6):
            subsets.append((i, j))
    
    # Map each 2-subset to an index 0..14
    # Index order: (0,1)=0, (0,2)=1, (0,3)=2, (0,4)=3, (0,5)=4,
    #               (1,2)=5, (1,3)=6, (1,4)=7, (1,5)=8,
    #               (2,3)=9, (2,4)=10, (2,5)=11,
    #               (3,4)=12, (3,5)=13,
    #               (4,5)=14
    
    n = 16
    edges = []
    
    # Build adjacency for KG(6,2)
    # Vertex i corresponds to subsets[i]
    for i in range(15):
        for j in range(i + 1, 15):
            si = subsets[i]
            sj = subsets[j]
            # Check if disjoint
            if len(set(si) & set(sj)) == 0:
                edges.append([i, j])
    
    # Maximum independent set in KG(6,2): {0,1}, {2,3}, {4,5}
    # These correspond to indices 0, 9, 14
    # Connect vertex 15 to these three vertices
    max_independent_set = [0, 9, 14]
    for v in max_independent_set:
        edges.append([15, v])
    
    return edges
# EVOLVE-BLOCK-END
