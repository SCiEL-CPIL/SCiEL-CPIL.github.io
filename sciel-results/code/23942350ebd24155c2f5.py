"""Four-colouring of complete graphs without monochromatic triangles (R(3,3,3,3))."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _mul(a, b):
    """Multiply in GF(16) with primitive polynomial 0b10011."""
    r = 0
    for i in range(4):
        if b >> i & 1:
            r ^= a << i
    for i in range(7, 3, -1):
        if r >> i & 1:
            r ^= 0b10011 << (i - 4)
    return r

def _build_seed_32():
    """Greenwood-Gleason colouring of K32 (two copies of K16 with colour 3 between)."""
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

def _color_index(n, i, j):
    """Index in the flat colors list for pair (i,j), i<j, n vertices."""
    # Number of pairs before vertex i: i*(i-1)/2
    # Within vertex i's block: j - i - 1
    return i * (i - 1) // 2 + (j - i - 1)

def _has_mono_triangle(colors, n, i, j, k):
    """Check if triangle (i,j,k) is monochromatic. Assumes i<j<k."""
    c1 = colors[_color_index(n, i, j)]
    c2 = colors[_color_index(n, i, k)]
    if c1 != c2:
        return False
    c3 = colors[_color_index(n, j, k)]
    return c1 == c3

def _find_mono_triangles(colors, n, limit=1000):
    """Find monochromatic triangles. Returns list of (i,j,k)."""
    tri = []
    for i in range(n):
        for j in range(i + 1, n):
            cij = colors[_color_index(n, i, j)]
            for k in range(j + 1, n):
                if len(tri) >= limit:
                    return tri
                if colors[_color_index(n, i, k)] == cij and colors[_color_index(n, j, k)] == cij:
                    tri.append((i, j, k))
    return tri

def _try_grow(colors, n, max_n, deadline, rng, max_iters=500000):
    """Try to grow the coloring by adding vertices. Returns (colors, new_n) or None."""
    cur_colors = colors[:]
    cur_n = n
    best_colors = colors[:]
    best_n = n
    iters = 0
    
    while cur_n < max_n:
        iters += 1
        if iters % 5000 == 0 and time.time() > deadline:
            break
        
        # Try adding a new vertex cur_n
        new_n = cur_n + 1
        new_colors = cur_colors[:]
        
        # For each existing vertex, assign a color to edge (v, cur_n)
        # such that no monochromatic triangle is created
        feasible = [True] * new_n
        for v in range(cur_n):
            # Color for edge (v, cur_n)
            used = set()
            for u in range(v):
                if _has_mono_triangle_with_new_edge(cur_colors, cur_n, u, v, cur_n):
                    used.add(cur_colors[_color_index(cur_n, u, v)])
            # Actually need: for each pair (u,v) where u<v<cur_n, if color(u,v)==c, 
            # then we cannot use c for edge (v, cur_n) if color(u,cur_n) would be c... 
            # This is complex. Let me use a simpler approach.
            pass
        
        # Simpler: use constraint propagation
        # For the new vertex, each edge (v, new) gets a color in {0,1,2,3}
        # Constraint: for any triangle (u,v,new) with u<v<new, the three edges must not all be same color
        # Edge (new,v) has color c_v. Edges (u,v) and (u,new) are: (u,v) is fixed, (u,new) is c_u.
        # So: NOT (c_u == c_v == color(u,v))
        
        # This is a CSP. Let's solve it greedily with backtracking for small cur_n.
        # For cur_n up to ~50, this might be too slow. Use random assignment + local repair.
        
        # Random assignment approach:
        for attempt in range(50):
            assignment = [rng.randint(0, 3) for _ in range(cur_n)]
            # Check constraints
            valid = True
            for v in range(cur_n):
                for u in range(v):
                    cv = assignment[v]
                    cu = assignment[u]
                    cuv = cur_colors[_color_index(cur_n, u, v)]
                    if cu == cv == cuv:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_colors = cur_colors[:] + assignment
                return new_colors, new_n
        
        # If we can't add a vertex, try to repair the current coloring by recoloring
        if iters % 100 == 0:
            # Local search: try to fix monochromatic triangles
            tri = _find_mono_triangles(cur_colors, cur_n, limit=10)
            if tri:
                i, j, k = tri[0]
                # Try recoloring one of the edges
                for idx, (a, b) in enumerate([(i, j), (i, k), (j, k)]):
                    old_c = cur_colors[_color_index(cur_n, a, b)]
                    new_c = rng.randint(0, 3)
                    if new_c != old_c:
                        cur_colors[_color_index(cur_n, a, b)] = new_c
                        # Check if this created new triangles or fixed old ones
                        # For simplicity, just accept and continue
                        break
        
        # If stuck, try restarting with a different random seed
        if iters % 1000 == 0:
            rng.seed()
    
    if cur_n > best_n:
        best_colors = cur_colors[:]
        best_n = cur_n
    
    return best_colors, best_n

def _has_mono_triangle_with_new_edge(colors, n, u, v, w):
    """Helper: check if triangle (u,v,w) would be monochromatic given colors for existing edges."""
    # This is a placeholder; the actual check is done inline.
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Keep margin before 150s kill
    
    rng = random.Random(42)
    
    # Start with the known 32-vertex coloring
    colors = _build_seed_32()
    n = 32
    best_colors = colors[:]
    best_n = n
    
    # Try to grow to 33, 34, ... up to 50 or time runs out
    max_target = 50
    
    while n < max_target:
        if time.time() > deadline:
            break
        
        result = _try_grow(colors, n, max_target, deadline, rng, max_iters=100000)
        if result is not None:
            new_colors, new_n = result
            if new_n > best_n:
                best_colors = new_colors
                best_n = new_n
                colors = new_colors
                n = new_n
            else:
                # Couldn't grow further; break
                break
        else:
            break
    
    # Verify the best coloring
    if best_n >= 3:
        tri = _find_mono_triangles(best_colors, best_n, limit=1)
        if tri:
            # Fallback to seed if invalid
            best_colors = _build_seed_32()
            best_n = 32
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
