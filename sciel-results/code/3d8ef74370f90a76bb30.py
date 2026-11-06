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
    return i * (i - 1) // 2 + (j - i - 1)

def _count_mono_triangles(colors, n):
    """Count monochromatic triangles."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            cij = colors[_color_index(n, i, j)]
            for k in range(j + 1, n):
                if colors[_color_index(n, i, k)] == cij and colors[_color_index(n, j, k)] == cij:
                    count += 1
    return count

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

def _fast_delta(colors, n, edge, old_c, new_c):
    """
    Compute the change in mono-triangle count if we change the color of edge to new_c.
    edge is (u, v) with u < v.
    Returns delta.
    """
    u, v = edge
    if u > v:
        u, v = v, u
    
    broken = 0
    created = 0
    
    # Precompute colors to u and v for all other vertices
    # To be safe and simple, iterate all k.
    for k in range(n):
        if k == u or k == v:
            continue
        
        # Get color(u, k)
        if u < k:
            c_uk = colors[_color_index(n, u, k)]
        else:
            c_uk = colors[_color_index(n, k, u)]
            
        # Get color(v, k)
        if v < k:
            c_vk = colors[_color_index(n, v, k)]
        else:
            c_vk = colors[_color_index(n, k, v)]
            
        if c_uk == old_c and c_vk == old_c:
            broken += 1
        if c_uk == new_c and c_vk == new_c:
            created += 1
            
    return created - broken

def _local_search(colors, n, deadline, rng):
    """
    Local search to minimize monochromatic triangles.
    Returns (colors, triangle_count).
    """
    cur_colors = colors[:]
    cur_count = _count_mono_triangles(cur_colors, n)
    best_colors = cur_colors[:]
    best_count = cur_count
    
    # Simulated Annealing parameters
    temp = 5.0
    cooling = 0.9995
    
    num_edges = n * (n - 1) // 2
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute index for all edges
    edge_indices = [_color_index(n, i, j) for i, j in all_edges]
    
    while time.time() < deadline:
        if cur_count == 0:
            break
            
        # Pick a random edge
        e_idx = rng.randint(0, num_edges - 1)
        u, v = all_edges[e_idx]
        old_c = cur_colors[edge_indices[e_idx]]
        
        # Try all possible new colors
        best_delta = float('inf')
        best_new_c = None
        
        for new_c in range(4):
            if new_c == old_c:
                continue
            delta = _fast_delta(cur_colors, n, (u, v), old_c, new_c)
            if delta < best_delta:
                best_delta = delta
                best_new_c = new_c
            elif delta == best_delta and rng.random() < 0.5:
                best_new_c = new_c
                
        if best_new_c is None:
            break
            
        # Acceptance criterion
        accept = False
        if best_delta < 0:
            accept = True
        elif best_delta == 0:
            if rng.random() < 0.5:
                accept = True
        else:
            # Accept with probability exp(-delta/temp)
            # Approx: if random < temp / delta? No, standard is exp(-delta/temp)
            # For small temp and positive delta, this is very small.
            # Let's use a simple heuristic: accept with prob 0.1 * exp(-delta/temp)
            # Or just 0 if delta > 0 and temp is small.
            # Let's stick to a simplified SA:
            if rng.random() < min(1.0, temp / (1.0 + abs(best_delta))):
                accept = True
                
        if accept:
            cur_colors[edge_indices[e_idx]] = best_new_c
            cur_count += best_delta
            temp *= cooling
            if cur_count < best_count:
                best_count = cur_count
                best_colors = cur_colors[:]
                
    return best_colors, best_count

def _greedy_construct(n, deadline, rng):
    """
    Greedy randomized construction.
    """
    colors = [0] * (n * (n - 1) // 2)
    
    for v in range(1, n):
        for u in range(v):
            best_c = 0
            best_count = float('inf')
            for c in range(4):
                cnt = 0
                for w in range(u):
                    if colors[_color_index(n, w, u)] == c and colors[_color_index(n, w, v)] == c:
                        cnt += 1
                if cnt < best_count:
                    best_count = cnt
                    best_c = c
                elif cnt == best_count:
                    if rng.random() < 0.5:
                        best_c = c
            colors[_color_index(n, u, v)] = best_c
            
    tri_count = _count_mono_triangles(colors, n)
    return colors, tri_count

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Keep margin before 150s kill
    
    rng = random.Random(42)
    
    best_result = {'n': 32, 'colors': _build_seed_32()}
    
    # Try to find a valid coloring for n=50, 49, ... down to 33.
    for n in range(50, 32, -1):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 5 # Fewer attempts to allow more time for local search
        
        while not found and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            # Try greedy construction first, then random
            if attempts == 1:
                colors, tri_count = _greedy_construct(n, deadline, rng)
            else:
                colors = [rng.randint(0, 3) for _ in range(n * (n - 1) // 2)]
            
            # Run local search
            res_colors, res_count = _local_search(colors, n, deadline, rng)
            
            if res_count == 0:
                # Valid coloring found!
                if n > best_result['n']:
                    best_result = {'n': n, 'colors': res_colors}
                found = True
                break
        
        # If we found a valid n, we can stop if it's 50? 
        # The task says "Maximize n".
        # If we found 50, we are done.
        # If we found 49, we should try 50 again?
        # The loop goes 50 down to 33.
        # If we found 50, we break.
        if best_result['n'] == n:
            break
            
    # Final verification
    if best_result['n'] >= 3:
        tri = _find_mono_triangles(best_result['colors'], best_result['n'], limit=1)
        if tri:
            # Fallback
            best_result = {'n': 32, 'colors': _build_seed_32()}
            
    return best_result
# EVOLVE-BLOCK-END
