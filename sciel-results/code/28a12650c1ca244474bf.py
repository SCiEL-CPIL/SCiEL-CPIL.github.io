"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Construct two disjoint K_{3,3} components.
    # Each K_{3,3} is bipartite with parts of size 3, so it's triangle-free.
    # The only cycle lengths in K_{3,3} are 4 and 6.
    # Since the graph is disconnected (two components), the set of distinct cycle lengths
    # is the union of cycle lengths from each component, which is {4, 6}.
    # This gives a raw value of 2.
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0..5
    # Partition: A = {0, 1, 2}, B = {3, 4, 5}
    part1_a = [0, 1, 2]
    part1_b = [3, 4, 5]
    for u in part1_a:
        for v in part1_b:
            edges.append([u, v])
    
    # Component 2: K_{3,3} on vertices 6..11
    # Partition: A = {6, 7, 8}, B = {9, 10, 11}
    part2_a = [6, 7, 8]
    part2_b = [9, 10, 11]
    for u in part2_a:
        for v in part2_b:
            edges.append([u, v])
    
    # Verify constraints:
    # 1. Simple graph: no loops (u != v always since parts are disjoint), no duplicates (each pair appears once).
    # 2. Degree >= 3: each vertex in part A has degree 3 (connected to all 3 in part B), same for part B.
    # 3. Triangle-free: K_{3,3} is bipartite, so no odd cycles, hence no triangles.
    # 4. Contains a cycle: K_{3,3} contains 4-cycles and 6-cycles.
    # 5. Girth >= 4: satisfied since triangle-free.
    
    return edges

# EVOLVE-BLOCK-END
