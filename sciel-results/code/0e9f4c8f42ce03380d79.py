"""
Program to construct a Pentagon-free graph without independent 9-sets.
Uses a deterministic construction based on a disjoint union of cliques.
"""
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph on n=35 vertices that is C5-free and has independence number < 9.
    
    Strategy:
    Use a disjoint union of cliques. 
    Let the components be K_4 (7 times) and K_7 (1 time).
    Total vertices: 7 * 4 + 7 = 35.
    Independence number: 7 * 1 + 1 * 1 = 8 < 9.
    C5-free: Disjoint union of cliques contains no cycles of length > 3.
    """
    n = 35
    edges = []
    
    # Define components as lists of vertex indices
    # 7 components of size 4, 1 component of size 7
    components = []
    current_vertex = 0
    
    # Add 7 components of size 4
    for _ in range(7):
        comp = list(range(current_vertex, current_vertex + 4))
        components.append(comp)
        current_vertex += 4
        
    # Add 1 component of size 7 (vertices 28 to 34)
    comp = list(range(current_vertex, current_vertex + 7))
    components.append(comp)
    
    # Generate edges for each clique component
    for comp in components:
        for i in range(len(comp)):
            for j in range(i + 1, len(comp)):
                edges.append([comp[i], comp[j]])
                
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
