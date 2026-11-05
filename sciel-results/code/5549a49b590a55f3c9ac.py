"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n = 20
    # We'll try to build a dense graph with rainbow 4-cycles.
    # Strategy: use a complete bipartite graph K_{10,10} and find a valid coloring.
    # If that fails, fall back to a sparser construction.
    
    # First, let's try a simpler approach: 
    # Use a graph that is a union of disjoint 4-cycles or a specific structure.
    
    # Actually, let's try: complete graph K_5 on vertices 0-4, 
    # and connect remaining vertices as a star from vertex 0.
    # K_5 has 10 edges. Star from 0 to 5..19 has 15 edges. Total 25.
    # But we need to color K_5 so all 4-cycles are rainbow.
    
    # Let's try a different approach: 
    # Use a complete bipartite graph K_{10,10} with vertices 0-9 (left) and 10-19 (right).
    # Try to find a coloring using backtracking or a known scheme.
    
    # Known result: In a bipartite graph, if we color edges such that 
    # for any two vertices on the same side, their incident edges have 
    # distinct colors in a certain way...
    
    # Let me try: color edge (i, j) with i in 0..9, j in 10..19 as:
    # color = (i + (j - 10)) % 20
    # We showed this doesn't work. Let's try:
    # color = (i * 2 + (j - 10)) % 20
    # or use a Latin square based coloring.
    
    # Actually, let's try a much simpler valid construction:
    # A cycle C_20: vertices 0-19 in a cycle. 20 edges.
    # Every 4-cycle in C_20? C_20 has no 4-cycles at all! 
    # A 4-cycle requires 4 vertices where each is connected to two others in the cycle.
    # In a simple cycle, there are no chords, so no 4-cycles exist.
    # So C_20 is valid with 20 edges.
    
    # Can we do better? What about C_20 plus some chords?
    # Adding a chord (i, j) creates new 4-cycles.
    
    # Let's try: take C_20 and add chords carefully.
    # Or better: use a complete bipartite graph K_{2,18} = 36 edges.
    # In K_{2,18}, a 4-cycle needs 2 vertices from the side of size 2 and 2 from the side of size 18.
    # There's only C(2,2)=1 way to choose the first side, so all 4-cycles use both vertices from the small side.
    # Vertices: left = {0, 1}, right = {2, 3, ..., 19}
    # A 4-cycle: 0 - a - 1 - b - 0 where a, b are distinct vertices in right.
    # Edges: (0,a), (a,1), (1,b), (b,0)
    # For this to be rainbow: c(0,a) != c(a,1) != c(1,b) != c(b,0) and all distinct.
    
    # If we color: c(0, a) = a-2 for a in 2..19 (colors 0..17)
    #             c(1, a) = a-2 for a in 2..19 (colors 0..17)
    # Then c(0,a) = c(1,a) which violates rainbow for the 4-cycle using a and any b.
    
    # Instead: c(0, a) = (a-2) % 20, c(1, a) = (a-2+10) % 20
    # Then for 4-cycle 0-a-1-b-0:
    # c(0,a) = (a-2) % 20
    # c(1,a) = (a-2+10) % 20  
    # c(1,b) = (b-2+10) % 20
    # c(0,b) = (b-2) % 20
    # These are: x, x+10, y+10, y where x = a-2, y = b-2, x != y
    # All distinct iff x != y (true), x != x+10 (true since 10 != 0 mod 20), 
    # x != y+10, x+10 != y, x+10 != y+10 (true), y+10 != y (true)
    # So we need x != y+10 and x+10 != y, i.e., x-y != 10 and y-x != 10 mod 20
    # i.e., x-y != ±10 mod 20
    # This fails when |a-b| = 10, i.e., a and b differ by 10.
    
    # So this coloring works for K_{2,18} EXCEPT when two right vertices differ by 10.
    # We can avoid this by not including all edges, or by using a different coloring.
    
    # Alternative: use only right vertices 2..11 (10 vertices) to avoid the conflict.
    # K_{2,10} has 20 edges. Same as C_20.
    
    # Let's try K_{3,17}: left = {0,1,2}, right = {3..19}
    # 4-cycles: choose 2 from left (C(3,2)=3 ways) and 2 from right.
    # This gets complex.
    
    # Let me just implement a greedy edge-addition with backtracking for coloring.
    
    # Simpler: return C_20 (20 edges) as a baseline improvement over 19.
    edges = []
    for i in range(20):
        j = (i + 1) % 20
        if i < j:
            edges.append([i, j, i])  # color by starting vertex
        else:
            edges.append([j, i, i])  # ensure u < v
    
    # Verify: C_20 has no 4-cycles, so it's trivially valid.
    # But wait, does the evaluator check for 4-cycles as subgraphs? 
    # A 4-cycle subgraph requires 4 vertices and 4 edges forming a cycle.
    # In C_20, no such subgraph exists. So it's valid.
    
    return edges

# EVOLVE-BLOCK-END
