import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 26
    # colors[i][j] is the color of edge (i, j) for i < j
    # We will use a 2D list for easy access, or a dict.
    # 26x26 matrix is small.
    
    # Initialize with a heuristic for the first 25 vertices (Z5 x Z5)
    # Vertex v corresponds to (v % 5, v // 5)
    def get_coords(v):
        return (v % 5, v // 5)
        
    colors = [[0]*n for _ in range(n)]
    
    # Heuristic initialization for 0..24
    for i in range(25):
        for j in range(i+1, 25):
            x1, y1 = get_coords(i)
            x2, y2 = get_coords(j)
            if x1 != x2:
                # Slope m
                # (y2 - y1) * inv(x2 - x1) mod 5
                dx = (x2 - x1) % 5
                dy = (y2 - y1) % 5
                # Inverse of dx mod 5
                # 1->1, 2->3, 3->2, 4->4
                inv = {1:1, 2:3, 3:2, 4:4}
                m = (dy * inv[dx]) % 5
                colors[i][j] = m
            else:
                # Vertical edge
                # Use x1 mod 5 as color?
                colors[i][j] = x1 % 5
                
    # Initialize edges to vertex 25
    for i in range(25):
        colors[i][25] = i % 5
        colors[25][i] = colors[i][25]

    # Function to check if a specific cycle (v1, v2, v3, v4) is monochromatic
    # Returns the color if monochromatic, else -1
    def get_mono_color(v1, v2, v3, v4):
        c1 = colors[v1][v2]
        c2 = colors[v2][v3]
        c3 = colors[v3][v4]
        c4 = colors[v4][v1]
        if c1 == c2 == c3 == c4:
            return c1
        return -1

    # Precompute all 4-cycles (sets of 4 vertices and their pairings)
    # A 4-cycle is defined by 4 distinct vertices.
    # There are 3 ways to form a cycle from 4 vertices:
    # 1-2-3-4-1
    # 1-2-4-3-1
    # 1-3-2-4-1
    # We will store these as tuples of 4 vertices in order.
    cycles = []
    for v1 in range(n):
        for v2 in range(v1+1, n):
            for v3 in range(v2+1, n):
                for v4 in range(v3+1, n):
                    cycles.append((v1, v2, v3, v4))
                    cycles.append((v1, v2, v4, v3))
                    cycles.append((v1, v3, v2, v4))
                    
    # Local Search
    # We maintain a list of currently violating cycles?
    # Or just scan all cycles.
    # Number of cycles ~ 45,000.
    # Scanning all is fast enough.
    
    best_n = 11
    best_colors = None
    
    # Keep the initial n=11 solution as fallback
    # K11 split into 5 Hamiltonian cycles
    n_init = 11
    init_colors = [0]* (n_init*(n_init-1)//2)
    idx = 0
    for i in range(n_init):
        for j in range(i+1, n_init):
            init_colors[idx] = (j - i) % 5 # Simple heuristic, might not be optimal but valid?
            # Actually the seed program used: min(j-i, n-j+i)-1
            # Let's just use the seed program's logic for the fallback
            idx += 1
            
    # Re-implement seed fallback properly
    def get_seed_solution():
        n = 11
        cols = []
        for i in range(n):
            for j in range(i+1, n):
                d = j - i
                cols.append(min(d, n - d) - 1)
        return {'n': n, 'colors': cols}
        
    seed_sol = get_seed_solution()
    
    # Main loop
    while time.time() < deadline:
        # Find a violating cycle
        violating_cycle = None
        violating_color = -1
        
        # Shuffle cycles to avoid bias?
        # For speed, just iterate.
        for cycle in cycles:
            c = get_mono_color(*cycle)
            if c != -1:
                violating_cycle = cycle
                violating_color = c
                break
                
        if violating_cycle is None:
            # No violations found!
            # Construct the output list
            out_colors = []
            for i in range(n):
                for j in range(i+1, n):
                    out_colors.append(colors[i][j])
            return {'n': n, 'colors': out_colors}
            
        # Pick a random edge in the violating cycle to recolor
        # Cycle is (v1, v2, v3, v4)
        # Edges: (v1,v2), (v2,v3), (v3,v4), (v4,v1)
        v1, v2, v3, v4 = violating_cycle
        edges_to_change = [
            (v1, v2),
            (v2, v3),
            (v3, v4),
            (v4, v1)
        ]
        
        # Randomly select an edge
        edge = random.choice(edges_to_change)
        u, v = edge
        if u > v:
            u, v = v, u
            
        # Try a new random color
        new_color = random.randint(0, 4)
        
        # Optional: Check if this change reduces the number of violations?
        # Too expensive to check full impact. Just apply.
        colors[u][v] = new_color
        colors[v][u] = new_color

    # If time runs out, return the seed solution
    return seed_sol
