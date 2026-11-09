import time
import random
import sys

def construct():
    """
    Uses a local search heuristic to find a 4-coloring of K_n with no monochromatic triangles.
    Starts with n=50 (the known record) and attempts to find a valid coloring.
    If it fails within the time limit, it reduces n and tries again.
    """
    start_time = time.time()
    time_limit = 100.0  # Leave 20s buffer for evaluation overhead
    
    # We aim for n=50. If we can't find it, we try lower n.
    # The current seed is n=32. Any valid n > 32 improves the score.
    
    def count_triangles(colors, n):
        """Count monochromatic triangles in the coloring."""
        # colors is a list of length n*(n-1)//2
        # For efficiency, we can build adjacency lists for each color
        # But for n=50, a direct check is also feasible if optimized.
        # Let's build color-specific adjacency sets.
        adj = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c].add((i, j))
                idx += 1
        
        count = 0
        for c in range(4):
            # For each vertex, check pairs of neighbors in the same color
            # This is O(n^3) worst case, but n=50 is small.
            # Better: for each pair (i,j) in color c, check if (i,k) and (j,k) are in color c for any k.
            # Or simply: for each triangle (i,j,k), check if all 3 edges are same color.
            # Number of triangles is nC3 = 19600 for n=50. This is very fast.
            pass
        
        # Direct triangle check is simpler and fast enough for n=50
        # We need to map (i,j) to color quickly.
        # colors is flat list. Index for (i,j) i<j is i*n - i*(i+1)//2 + (j-i-1)
        # Let's just use the flat list and a helper to get color.
        
        tri_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[i * n - i * (i + 1) // 2 + (j - i - 1)]
                for k in range(j + 1, n):
                    c_ik = colors[i * n - i * (i + 1) // 2 + (k - i - 1)]
                    c_jk = colors[j * n - j * (j + 1) // 2 + (k - j - 1)]
                    if c_ij == c_ik == c_jk:
                        tri_count += 1
        return tri_count

    def get_color(colors, n, i, j):
        if i > j: i, j = j, i
        return colors[i * n - i * (i + 1) // 2 + (j - i - 1)]

    def set_color(colors, n, i, j, c):
        if i > j: i, j = j, i
        colors[i * n - i * (i + 1) // 2 + (j - i - 1)] = c

    # Heuristic Search
    best_n = 32
    best_colors = None
    
    # Fallback: The seed program's n=32 coloring (re-implemented simply or just return seed logic if search fails)
    # Actually, if search fails for all n > 32, we should return the seed result to ensure 1.0 score.
    # But the search should easily find n > 32.
    
    for target_n in [50, 48, 46, 44, 42, 40, 38, 36, 34, 33]:
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random restarts for this n
        found = False
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            n = target_n
            # Initialize colors randomly
            colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
            
            # Local Search
            # We want to eliminate all monochromatic triangles.
            # Strategy: Find a triangle, change color of one of its vertices.
            # To avoid cycling, we can use a simple tabu or just random selection.
            
            # Optimize: Keep track of "bad" vertices (involved in triangles).
            # For n=50, checking all triangles is ~20k checks. Very fast.
            
            # Iterate
            for _ in range(10000): # Max iterations per restart
                if time.time() - start_time > time_limit:
                    break
                
                # Find a monochromatic triangle
                # We can sample random triangles to find one quickly if many exist
                # Or just scan. Scanning 20k triangles is < 1ms in Python? Maybe 10ms.
                # Let's do a quick scan.
                
                tri_found = False
                # To speed up, we can check random triangles first
                # But for correctness, we just need to find ANY.
                
                # Efficient check:
                # We'll just iterate. If we find one, we break.
                for i in range(n):
                    if tri_found: break
                    for j in range(i + 1, n):
                        if tri_found: break
                        c_ij = get_color(colors, n, i, j)
                        for k in range(j + 1, n):
                            c_ik = get_color(colors, n, i, k)
                            c_jk = get_color(colors, n, j, k)
                            if c_ij == c_ik == c_jk:
                                # Found triangle (i,j,k) with color c_ij
                                # Pick a random vertex to recolor
                                # Options: i, j, k
                                # Try all 3 other colors for the chosen vertex?
                                # Or just pick one random vertex and one random new color.
                                # Better: Try to recolor one of the vertices to a color that breaks the triangle
                                # and hopefully doesn't create new ones.
                                
                                # Simple move: Pick random vertex v from {i,j,k}
                                # Pick random new color c_new != old_color(v)
                                # Actually, changing to ANY color other than the triangle's color breaks this specific triangle.
                                # But it might create new ones.
                                
                                v = random.choice([i, j, k])
                                old_c = get_color(colors, n, v, (i if v!=i else j)) # Just get current color of v? No, v is a vertex.
                                # We need the current color assigned to edges incident to v? No, the color is per edge.
                                # Wait, the "color of a vertex" is not defined. The color is per edge.
                                # Recoloring a vertex means changing the color of ALL edges incident to that vertex?
                                # NO. That is not a standard move in edge-coloring SAT.
                                # A standard move is changing the color of a SINGLE EDGE.
                                # Or, if we view it as vertex coloring of the line graph?
                                # The problem is edge coloring of K_n such that no 3 edges forming a triangle have the same color.
                                # This is equivalent to: For every triangle, the 3 edges must not all be the same color.
                                
                                # So, we can change the color of ONE edge in the triangle.
                                edge_idx = random.choice([(i, j), (i, k), (j, k)])
                                u, w = edge_idx
                                old_edge_c = get_color(colors, n, u, w)
                                # Choose a new color different from old_edge_c
                                new_edge_c = random.choice([c for c in range(4) if c != old_edge_c])
                                set_color(colors, n, u, w, new_edge_c)
                                tri_found = True
                                break
                            
                if not tri_found:
                    # No monochromatic triangles found!
                    # We must verify thoroughly before accepting.
                    if count_triangles(colors, n) == 0:
                        best_n = n
                        best_colors = colors
                        found = True
                        break
            if found:
                break
                
    if best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    else:
        # Fallback to seed program logic (n=32) to ensure score >= 1.0
        # Re-implementing the seed logic for safety
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
        return {'n': 32, 'colors': colors}
