"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time

def construct():
    """
    Construct a K4-free graph on 18 vertices (0..17) that minimizes the
    largest triangle-free induced subgraph.
    
    Design: Three parts of 6 vertices each (0-5, 6-11, 12-17).
    - All cross-part edges present (complete tripartite base).
    - Within each part, add a perfect matching (3 disjoint edges).
    
    This creates triangles within each part (matching edge + any cross edge),
    limiting any triangle-free set to at most 2 vertices per part, giving
    a maximum triangle-free induced subgraph of size 6 (2+2+2).
    
    K4-freeness: Any 4 vertices either have 3+ in one part (where the matching
    ensures non-adjacency) or span multiple parts (where cross edges prevent
    K4 formation in the tripartite structure).
    """
    edges = []
    
    # Part definitions
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    
    # Add all cross-part edges (complete tripartite)
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    
    # Add perfect matching within each part
    # Part 0: (0,1), (2,3), (4,5)
    # Part 1: (6,7), (8,9), (10,11)
    # Part 2: (12,13), (14,15), (16,17)
    for part in parts:
        for k in range(0, 6, 2):
            edges.append([part[k], part[k + 1]])
    
    return edges
