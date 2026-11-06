"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16) (the Clebsch graph three times),
    # doubled: two copies with colour 3 on all edges between them (a bipartite colour class).
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    # Helper to get edge index
    def edge_idx(n, i, j):
        if i > j:
            i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
    
    # Helper to check if adding vertex v with given colors creates a monochromatic triangle
    def check_vertex_valid(n, colors, v, new_colors):
        """Check if vertex v with new_colors to all existing vertices creates no mono triangle"""
        for i in range(v):
            for j in range(i + 1, v):
                # Check triangle (v, i, j)
                c_v_i = new_colors[i]  # color of edge (v, i)
                c_v_j = new_colors[j]  # color of edge (v, j)
                c_i_j = colors[edge_idx(n, i, j)]  # color of edge (i, j)
                if c_v_i == c_v_j == c_i_j:
                    return False
        return True
    
    # Helper to compute slack for a vertex: minimum over all pairs (i,j) of "distance" to mono triangle
    # We'll use a simpler heuristic: count how many (i,j) pairs would create a mono triangle if we pick each color
    def compute_slack(n, colors, v, new_colors):
        """For each possible color assignment to new vertex, compute the number of mono triangles created"""
        # This is expensive, so we'll use a simpler metric
        # Count potential conflicts
        conflicts = [0] * 4
        for i in range(v):
            for j in range(i + 1, v):
                c_i_j = colors[edge_idx(n, i, j)]
                # If we assign color c to (v,i) and (v,j), and c_i_j == c, we get a mono triangle
                # But we're assigning fixed colors in new_colors, so just check
                # Actually, let's just return 0 for valid, large number for invalid
        return 0  # Placeholder
    
    # Try to extend the coloring by adding vertices
    def try_extend(n, colors, max_n=50, time_limit=100):
        """Try to extend the coloring to larger n using greedy approach"""
        start_time = time.time()
        best_n = n
        best_colors = colors[:]
        
        for target_n in range(n + 1, max_n + 1):
            if time.time() - start_time > time_limit:
                break
            
            # Try to add vertex target_n-1
            found = False
            # Try different random orderings of color assignments
            for attempt in range(100):
                # Randomly assign colors to edges from new vertex to existing vertices
                new_colors = [random.randint(0, 3) for _ in range(target_n - 1)]
                
                if check_vertex_valid(target_n - 1, best_colors, target_n - 1, new_colors):
                    # Valid! Add the vertex
                    for c in new_colors:
                        best_colors.append(c)
                    best_n = target_n
                    found = True
                    break
            
            if not found:
                # Try local search: perturb existing colors and try again
                improved = False
                for _ in range(50):
                    # Randomly perturb some edges
                    perturbed = best_colors[:]
                    for _ in range(min(10, len(perturbed) // 10)):
                        idx = random.randint(0, len(perturbed) - 1)
                        perturbed[idx] = random.randint(0, 3)
                    
                    # Check if perturbed is still valid
                    valid = True
                    for i in range(best_n):
                        for j in range(i + 1, best_n):
                            for k in range(j + 1, best_n):
                                c_ij = perturbed[edge_idx(best_n, i, j)]
                                c_jk = perturbed[edge_idx(best_n, j, k)]
                                c_ik = perturbed[edge_idx(best_n, i, k)]
                                if c_ij == c_jk == c_ik:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Try to extend the perturbed coloring
                        for attempt in range(50):
                            new_colors = [random.randint(0, 3) for _ in range(target_n - 1)]
                            if check_vertex_valid(best_n, perturbed, target_n - 1, new_colors):
                                best_colors = perturbed + new_colors
                                best_n = target_n
                                improved = True
                                break
                        if improved:
                            break
                
                if not improved:
                    break  # Can't extend further
        
        return best_n, best_colors
    
    # Try to extend the initial coloring
    extended_n, extended_colors = try_extend(32, colors, max_n=50, time_limit=100)
    
    if extended_n > 32:
        return {'n': extended_n, 'colors': extended_colors}
    
    return {'n': 32, 'colors': colors}
# EVOLVE-BLOCK-END
