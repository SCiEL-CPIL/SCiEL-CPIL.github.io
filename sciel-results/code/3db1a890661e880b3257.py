"""Model-and-optimize: construct large triangle-free 4-colorings via bitmask local search."""
import random
import time
import math

# EVOLVE-BLOCK-START

def _baseline32():
    """Greenwood-Gleason K16 doubled -> K32."""
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

def _build_idx(n):
    idx = [[0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    return idx

def _count_triangles(colors, idx, n):
    tri = 0
    for u in range(n):
        for v in range(u + 1, n):
            c1 = colors[idx[u][v]]
            for w in range(v + 1, n):
                if c1 == colors[idx[v][w]] and c1 == colors[idx[w][u]]:
                    tri += 1
    return tri

def _solve_n(n, rng, deadline, init_colors=None):
    """
    Attempt to find a valid 4-coloring of K_n such that no monochromatic triangle exists.
    Returns the colors list if successful, None otherwise.
    """
    m = n * (n - 1) // 2
    if m == 0:
        return []
    
    idx = _build_idx(n)
    
    # Initialize coloring
    if init_colors is not None:
        colors = init_colors[:]
    else:
        colors = [rng.randint(0, 3) for _ in range(m)]
    
    # Initialize masks
    masks = [[0] * 4 for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = colors[idx[u][v]]
            masks[u][c] |= (1 << v)
            masks[v][c] |= (1 << u)
    
    # Calculate initial triangle count
    tri_count = _count_triangles(colors, idx, n)
    
    if tri_count == 0:
        return colors
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    
    while T > 0.001 and time.time() < deadline - 2.0:
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        i_idx = idx[u][v]
        old_c = colors[i_idx]
        new_c = rng.randint(0, 3)
        if old_c == new_c:
            continue
        
        lost = (masks[u][old_c] & masks[v][old_c]).bit_count()
        gained = (masks[u][new_c] & masks[v][new_c]).bit_count()
        
        delta = gained - lost
        
        accept = False
        if delta < 0:
            accept = True
        else:
            if delta > 50 * T:
                accept = False
            else:
                accept = rng.random() < math.exp(-delta / T)
        
        if accept:
            colors[i_idx] = new_c
            tri_count += delta
            
            masks[u][old_c] &= ~(1 << v)
            masks[v][old_c] &= ~(1 << u)
            masks[u][new_c] |= (1 << v)
            masks[v][new_c] |= (1 << u)
            
            if tri_count == 0:
                return colors
            
            T *= cooling
            
    return None

def _extend_from(base_n, target_n, base_colors, rng, deadline):
    """
    Extend a valid coloring of K_{base_n} to K_{target_n} by adding vertices one by one.
    Returns colors list if successful, None otherwise.
    """
    if base_n >= target_n:
        return base_colors[:]
    
    # Start with base coloring
    colors = base_colors[:]
    current_n = base_n
    
    idx = _build_idx(target_n)
    
    # Initialize masks for the full graph (with placeholder colors for new edges)
    # We'll build up masks as we add vertices
    
    # For now, use a simpler approach: just extend and then run SA
    # Add new vertices with random colors
    for v in range(base_n, target_n):
        for u in range(v):
            # Find the index for edge (u, v)
            # In the full idx, this is precomputed
            i_idx = idx[u][v]
            if i_idx < len(colors):
                colors[i_idx] = rng.randint(0, 3)
            else:
                # This shouldn't happen if we're building correctly
                colors.append(rng.randint(0, 3))
    
    # Now run SA to fix any triangles
    return _solve_n(target_n, rng, deadline, init_colors=colors)

def _validate(colors, n):
    """Check if the coloring is valid (no monochromatic triangles)."""
    if len(colors) != n * (n - 1) // 2:
        return False
    idx = _build_idx(n)
    for u in range(n):
        for v in range(u + 1, n):
            c1 = colors[idx[u][v]]
            for w in range(v + 1, n):
                if c1 == colors[idx[v][w]] and c1 == colors[idx[w][u]]:
                    return False
    return True

def construct():
    start = time.time()
    deadline = start + 115.0
    rng = random.Random(42)
    
    # Start with baseline
    best = _baseline32()
    best_n = 32
    
    # Try to extend from 32 to larger n
    # Strategy: try n=33, 34, ... up to 62
    # Use multiple restarts and different initialization strategies
    
    current_n = 33
    while current_n <= 62:
        if time.time() > deadline - 5.0:
            break
        
        # Allocate time for this n
        sub_deadline = time.time() + min(10.0, deadline - time.time() - 2.0)
        
        # Try multiple strategies
        res = None
        
        # Strategy 1: Random initialization
        while time.time() < sub_deadline - 1.0:
            res = _solve_n(current_n, rng, sub_deadline)
            if res is not None:
                break
        
        # Strategy 2: Extend from best known
        if res is None and time.time() < sub_deadline - 1.0:
            ext_deadline = sub_deadline - 1.0
            res = _extend_from(best_n, current_n, best['colors'], rng, ext_deadline)
        
        if res is not None:
            # Validate
            if _validate(res, current_n):
                best = {'n': current_n, 'colors': res}
                best_n = current_n
                current_n += 1
            else:
                # Invalid result, try again
                pass
        else:
            # Failed to find coloring for current_n
            # Break if we can't do this n
            break
    
    # Final validation
    if time.time() < deadline - 1.0:
        n = best['n']
        colors = best['colors']
        if not _validate(colors, n):
            best = _baseline32()
    
    return best

# EVOLVE-BLOCK-END
