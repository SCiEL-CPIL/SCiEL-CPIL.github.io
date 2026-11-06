"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Local search approach: start from a valid small 4-coloring and iteratively
    add vertices, using local search to find valid color assignments for new edges.
    """
    random.seed(42)
    start_time = time.time()
    time_limit = 115  # seconds
    
    def make_initial_coloring(n):
        """Create an initial random coloring of K_n."""
        colors = {}
        for i in range(n):
            for j in range(i + 1, n):
                colors[(i, j)] = random.randint(0, 3)
        return colors
    
    def count_mono_triangles(colors, n):
        """Count monochromatic triangles in the coloring."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[(i, j)]
                for k in range(j + 1, n):
                    if colors[(i, k)] == c and colors[(j, k)] == c:
                        count += 1
        return count
    
    def get_mono_triangle_vertices(colors, n):
        """Return a list of monochromatic triangles (i,j,k) with their color."""
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[(i, j)]
                for k in range(j + 1, n):
                    if colors[(i, k)] == c and colors[(j, k)] == c:
                        triangles.append((i, j, k, c))
        return triangles
    
    def local_search(colors, n, max_iter=100000):
        """Local search to eliminate monochromatic triangles."""
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            triangles = get_mono_triangle_vertices(colors, n)
            if not triangles:
                return True
            # Pick a random triangle and flip one of its edges
            tri = random.choice(triangles)
            i, j, k, c = tri
            # Try flipping one of the three edges to a different color
            edges = [(i, j), (i, k), (j, k)]
            random.shuffle(edges)
            improved = False
            for edge in edges:
                a, b = edge
                old_c = colors[(a, b)]
                # Try each alternative color
                new_colors = [cc for cc in range(4) if cc != old_c]
                random.shuffle(new_colors)
                for new_c in new_colors:
                    colors[(a, b)] = new_c
                    # Check if this reduces the number of mono triangles
                    # Quick check: count triangles involving this edge
                    before = len(get_mono_triangle_vertices(colors, n))
                    colors[(a, b)] = old_c
                    # Actually, let's just do a full check for simplicity (n is small-ish)
                    # But that's O(n^3) per trial... Let's use a smarter approach.
                    break
                # Just flip to a random different color
                colors[(a, b)] = new_colors[0]
                improved = True
                break
            if not improved:
                # Flip a random edge
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    if i > j:
                        i, j = j, i
                    colors[(i, j)] = random.randint(0, 3)
        return count_mono_triangles(colors, n) == 0
    
    def extend_coloring(colors, n, new_n):
        """Extend coloring from n to new_n vertices."""
        for new_v in range(n, new_n):
            # Assign colors to edges (new_v, v) for v in 0..new_v-1
            for v in range(new_v):
                colors[(v, new_v)] = random.randint(0, 3)
            # Local search to fix conflicts
            max_iter = 50000
            for iteration in range(max_iter):
                if time.time() - start_time > time_limit:
                    break
                triangles = get_mono_triangle_vertices(colors, new_v + 1)
                if not triangles:
                    break
                tri = random.choice(triangles)
                i, j, k, c = tri
                # Only flip edges involving new_v if possible, or any edge
                edges = [(i, j), (i, k), (j, k)]
                # Prefer edges involving new_v
                new_v_edges = [e for e in edges if new_v in e]
                if new_v_edges:
                    edge = random.choice(new_v_edges)
                else:
                    edge = random.choice(edges)
                a, b = edge
                old_c = colors[(a, b)]
                colors[(a, b)] = (old_c + 1) % 4  # Simple flip
        return colors
    
    # Start with a small valid coloring
    # Use the GF(16) construction for n=16 as seed
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log_table = {}
    x = 1
    for e in range(15):
        log_table[x] = e
        x = mul(x, 2)
    
    # Build initial n=16 coloring
    n_start = 16
    colors = {}
    for i in range(n_start):
        for j in range(i + 1, n_start):
            diff = i ^ j
            if diff == 0:
                colors[(i, j)] = 0
            else:
                colors[(i, j)] = log_table[diff] % 3
    
    # Verify n=16 is valid
    assert count_mono_triangles(colors, n_start) == 0, "Initial 16-vertex coloring is invalid!"
    
    # Try to extend to larger n
    target_n = 50
    best_n = n_start
    best_colors = dict(colors)
    
    for target in range(n_start + 1, min(target_n + 1, 63)):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Try multiple times to extend
        success = False
        for attempt in range(20):
            trial_colors = dict(best_colors)
            # Extend from best_n to target
            for new_v in range(best_n, target):
                for v in range(new_v):
                    trial_colors[(v, new_v)] = random.randint(0, 3)
            
            # Local search
            max_iter = 30000
            for iteration in range(max_iter):
                if time.time() - start_time > time_limit - 5:
                    break
                triangles = get_mono_triangle_vertices(trial_colors, target)
                if not triangles:
                    success = True
                    break
                tri = random.choice(triangles)
                i, j, k, c = tri
                edges = [(i, j), (i, k), (j, k)]
                # Prefer edges involving the newest vertices
                newest = max(target - best_n, 1)
                new_v_edges = [e for e in edges if max(e) >= best_n]
                if new_v_edges:
                    edge = random.choice(new_v_edges)
                else:
                    edge = random.choice(edges)
                a, b = edge
                old_c = trial_colors[(a, b)]
                # Try all other colors, pick one that reduces conflicts
                best_new_c = old_c
                best_conflicts = count_mono_triangles(trial_colors, target)
                for new_c in range(4):
                    if new_c == old_c:
                        continue
                    trial_colors[(a, b)] = new_c
                    conflicts = count_mono_triangles(trial_colors, target)
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_new_c = new_c
                trial_colors[(a, b)] = best_new_c
            
            if success:
                best_n = target
                best_colors = trial_colors
                break
        
        if not success:
            # Can't extend further with current approach
            # Try a few more random restarts
            for restart in range(5):
                if time.time() - start_time > time_limit - 5:
                    break
                trial_colors = dict(best_colors)
                for new_v in range(best_n, target):
                    for v in range(new_v):
                        trial_colors[(v, new_v)] = random.randint(0, 3)
                max_iter = 20000
                for iteration in range(max_iter):
                    if time.time() - start_time > time_limit - 5:
                        break
                    triangles = get_mono_triangle_vertices(trial_colors, target)
                    if not triangles:
                        best_n = target
                        best_colors = trial_colors
                        break
                    tri = random.choice(triangles)
                    i, j, k, c = tri
                    edges = [(i, j), (i, k), (j, k)]
                    edge = random.choice(edges)
                    a, b = edge
                    trial_colors[(a, b)] = random.randint(0, 3)
                if best_n == target:
                    break
            if best_n < target:
                break
    
    # Convert to output format
    n = best_n
    color_list = []
    for i in range(n):
        for j in range(i + 1, n):
            color_list.append(best_colors[(i, j)])
    
    return {'n': n, 'colors': color_list}
# EVOLVE-BLOCK-END
