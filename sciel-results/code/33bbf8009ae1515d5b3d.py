import time
import random
import sys
import numpy as np

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
    
    def get_color_idx(n, i, j):
        if i > j: i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def count_triangles(colors, n):
        """Count monochromatic triangles in the coloring."""
        tri_count = 0
        for i in range(n):
            base_i = i * n - i * (i + 1) // 2
            for j in range(i + 1, n):
                c_ij = colors[base_i + (j - i - 1)]
                base_j = j * n - j * (j + 1) // 2
                for k in range(j + 1, n):
                    c_ik = colors[base_i + (k - i - 1)]
                    c_jk = colors[base_j + (k - j - 1)]
                    if c_ij == c_ik == c_jk:
                        tri_count += 1
        return tri_count

    def get_color(colors, n, i, j):
        return colors[get_color_idx(n, i, j)]

    def set_color(colors, n, i, j, c):
        colors[get_color_idx(n, i, j)] = c

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
            # Strategy: Find a triangle, change color of one of its edges.
            
            # Iterate
            for _ in range(20000): # Max iterations per restart
                if time.time() - start_time > time_limit:
                    break
                
                # Find a monochromatic triangle
                tri_found = False
                # To speed up, we can check random triangles first
                # But for correctness, we just need to find ANY.
                
                # Efficient check:
                # We'll just iterate. If we find one, we break.
                for i in range(n):
                    if tri_found: break
                    base_i = i * n - i * (i + 1) // 2
                    for j in range(i + 1, n):
                        if tri_found: break
                        c_ij = colors[base_i + (j - i - 1)]
                        base_j = j * n - j * (j + 1) // 2
                        for k in range(j + 1, n):
                            c_ik = colors[base_i + (k - i - 1)]
                            c_jk = colors[base_j + (k - j - 1)]
                            if c_ij == c_ik == c_jk:
                                # Found triangle (i,j,k) with color c_ij
                                # Pick a random edge to recolor
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
