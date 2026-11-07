"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Construct a maximally diameter-2 graph on 16 vertices where every edge is critical.
    
    The complete bipartite graph K_{8,8} has 64 edges and diameter 2.
    Removing any edge (u, v) with u in A, v in B eliminates all length-2 paths between u and v
    because N(u) subset B and N(v) subset A are disjoint. The shortest path becomes length 3,
    increasing the diameter to 3.
    
    Any additional edge would create a triangle or a same-partition edge.
    1. Same-partition edge (u1, u2): Removing it leaves a length-2 path via any v in B,
       so diameter remains 2. Not critical.
    2. Triangle edge: Similarly, triangles provide redundant length-2 paths.
    
    Thus K_{8,8} is the unique maximal solution.
    """
    # Vertices 0..7 in Partition A, 8..15 in Partition B
    # Edges connect every vertex in A to every vertex in B
    edges = [[i, j] for i in range(8) for j in range(8, 16)]
    
    # Ensure types are strictly Python int and list
    # The list comprehension above already produces Python ints and lists
    return edges

# EVOLVE-BLOCK-END
