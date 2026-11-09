import random
import time
import math

def construct():
    """
    Constructs a 4-coloring of K_n without monochromatic triangles.
    Uses Simulated Annealing to find a valid 4-coloring of K_49.
    Falls back to n=32 seed if it fails.
    """
    
    N = 49
    # Precompute triangle indices for fast access
    # We will store colors in a 2D matrix C[i][j]
    
    def count_mono_triangles(C):
        """Count monochromatic triangles in a complete graph coloring."""
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                c_ij = C[i][j]
                for k in range(j + 1, N):
                    if C[i][k] == c_ij and C[j][k] == c_ij:
                        count += 1
        return count

    def delta_mono_triangles(C, u, v, new_color):
        """
        Calculate change in number of monochromatic triangles if C[u][v] is changed to new_color.
        Only triangles containing edge (u,v) are affected.
        """
        old_color = C[u][v]
        if old_color == new_color:
            return 0
        
        delta = 0
        # Triangles (u, v, w) for w != u, v
        for w in range(N):
            if w == u or w == v:
                continue
            
            c_uw = C[u][w]
            c_vw = C[v][w]
            
            # Check if (u,v,w) was monochromatic before
            if old_color == c_uw and old_color == c_vw:
                delta -= 1 # Was mono, now not mono (unless new_color also matches, handled below)
                
            # Check if (u,v,w) will be monochromatic after
            if new_color == c_uw and new_color == c_vw:
                delta += 1
                
        return delta

    # Initialize random coloring
    C = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            C[i][j] = random.randint(0, 3)
            C[j][i] = C[i][j]
            
    current_count = count_mono_triangles(C)
    
    # If we got lucky, return immediately
    if current_count == 0:
        colors = []
        for i in range(N):
            for j in range(i+1, N):
                colors.append(C[i][j])
        return {'n': N, 'colors': colors}

    # Simulated Annealing
    start_time = time.time()
    time_limit = 10.0 # Use 10s to be safe, leaving buffer for overhead
    
    T = 1.0
    cooling = 0.995
    
    best_count = current_count
    best_C = [row[:] for row in C]
    
    while time.time() - start_time < time_limit:
        # Pick random edge
        u = random.randint(0, N-1)
        v = random.randint(0, N-1)
        if u == v:
            continue
        
        # Pick new color different from current
        old_color = C[u][v]
        new_color = random.randint(0, 3)
        if new_color == old_color:
            continue
            
        delta = delta_mono_triangles(C, u, v, new_color)
        
        # Acceptance criterion
        if delta < 0 or (random.random() < math.exp(-delta / T)):
            C[u][v] = new_color
            C[v][u] = new_color
            current_count += delta
            
            if current_count < best_count:
                best_count = current_count
                best_C = [row[:] for row in C]
                
                if best_count == 0:
                    colors = []
                    for i in range(N):
                        for j in range(i+1, N):
                            colors.append(C[i][j])
                    return {'n': N, 'colors': colors}
        
        T *= cooling
        if T < 0.001:
            T = 0.001 # Prevent underflow, or restart? Restarting is better for SA.
            # Let's restart if T is too low and we haven't found 0
            if best_count > 0:
                # Reheat
                T = 1.0
                # Optionally restart from best
                C = [row[:] for row in best_C]
                current_count = best_count

    # If we didn't find a perfect solution, check best
    if best_count == 0:
        colors = []
        for i in range(N):
            for j in range(i+1, N):
                colors.append(best_C[i][j])
        return {'n': N, 'colors': colors}
        
    # Fallback to seed
    return _seed_construct()

def _seed_construct():
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
