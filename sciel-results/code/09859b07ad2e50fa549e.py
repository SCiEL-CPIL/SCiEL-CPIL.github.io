"""Stronger algorithm for R(3,3,3,3): GF(16) Clebsch base + optimized greedy vertex extension."""
import time
import random

def construct():
    deadline = time.time() + 130.0
    
    # GF(16) multiplication for Clebsch graph construction
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    # Build log table for GF(16)
    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    # Construct base 32-vertex coloring: two Clebsch graphs + bipartite edges in color 3
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    
    n = 32
    
    def get_color(a, b, n):
        """Get color of edge (a,b) in lexicographic order."""
        if a > b:
            a, b = b, a
        # Index in the flat list
        idx = 0
        for k in range(a):
            idx += n - 1 - k
        idx += b - a - 1
        return colors[idx]
    
    def is_monochromatic_triangle(v1, v2, v3, n):
        """Check if triangle (v1,v2,v3) is monochromatic."""
        c12 = get_color(min(v1,v2), max(v1,v2), n)
        c13 = get_color(min(v1,v3), max(v1,v3), n)
        c23 = get_color(min(v2,v3), max(v2,v3), n)
        return c12 == c13 == c23
    
    def count_conflicts_for_new_vertex(new_v, new_colors, n):
        """Count monochromatic triangles involving new vertex new_v."""
        conflicts = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = get_color(i, j, n)
                c_in = new_colors[i]
                c_jn = new_colors[j]
                if c_ij == c_in == c_jn:
                    conflicts += 1
        return conflicts
    
    def try_add_vertex(n, colors, deadline, max_iter=500000):
        """Try to add one vertex using greedy + local search."""
        new_v = n
        
        # Initial greedy assignment
        new_colors = [0] * n
        for i in range(n):
            best_c = 0
            best_conf = float('inf')
            for c in range(4):
                # Count conflicts if we assign color c to edge (i, new_v)
                conf = 0
                for j in range(i):
                    c_ij = get_color(j, i, n)
                    if c_ij == c:
                        # Check if (j, i, new_v) would be monochromatic
                        # We need c_j_new to also be c, but we're assigning sequentially
                        # For greedy, just count how many j have c_ij == c
                        conf += 1
                if conf < best_conf:
                    best_conf = conf
                    best_c = c
            new_colors[i] = best_c
        
        # Local search to reduce conflicts
        iteration = 0
        while iteration < max_iter:
            if time.time() > deadline - 10.0:
                break
            total_conf = count_conflicts_for_new_vertex(new_v, new_colors, n)
            if total_conf == 0:
                return True, new_colors
            
            # Find a random edge to flip
            i = random.randint(0, n - 1)
            old_c = new_colors[i]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            new_colors[i] = new_c
            new_conf = count_conflicts_for_new_vertex(new_v, new_colors, n)
            
            if new_conf <= total_conf:
                iteration = 0
            else:
                new_colors[i] = old_c
                iteration += 1
            
            if time.time() > deadline - 10.0:
                break
        
        final_conf = count_conflicts_for_new_vertex(new_v, new_colors, n)
        if final_conf == 0:
            return True, new_colors
        return False, new_colors
    
    # Try to extend the graph
    attempts = 0
    while time.time() < deadline - 15.0 and attempts < 100:
        attempts += 1
        
        success, new_colors = try_add_vertex(n, colors, deadline, max_iter=100000)
        if success:
            # Add the new vertex's colors to the main list
            colors.extend(new_colors)
            n += 1
            attempts = 0  # Reset attempts on success
        else:
            # Try a different strategy: random restart with different seed
            random.seed(int(time.time() * 1000) % (2**31))
            continue
    
    # Final validation: ensure no monochromatic triangles
    valid = True
    if n > 62:
        n = 62
        colors = colors[:n*(n-1)//2]
    
    # Quick validity check
    for i in range(min(n, 62)):
        for j in range(i + 1, min(n, 62)):
            for k in range(j + 1, min(n, 62)):
                c_ij = get_color(i, j, n)
                c_ik = get_color(i, k, n)
                c_jk = get_color(j, k, n)
                if c_ij == c_ik == c_jk:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to known valid 32-vertex coloring
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        n = 32
    
    return {'n': n, 'colors': colors}
