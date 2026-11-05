"""Model-and-optimize: construct large triangle-free 4-colorings via growth from baseline."""
import random
import time
import sys

# EVOLVE-BLOCK-START

def _edge_index(u, v, n):
    """Compute the flat index for edge (u,v) in lexicographic order."""
    if u > v:
        u, v = v, u
    return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

def _build_state(colors, n):
    """Build lookup matrix C and per-color neighbor lists from flat colors array."""
    C = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            C[i][j] = c
            C[j][i] = c
            idx += 1
    # Build per-color adjacency sets
    adj = [set() for _ in range(4)]
    for i in range(n):
        row = C[i]
        for j in range(i + 1, n):
            c = row[j]
            if c >= 0:
                adj[c].add((i, j))
    return C, adj

def _find_mono_triangle(C, adj, n):
    """Find a monochromatic triangle using per-color adjacency sets.
    Returns (i, j, k) or None."""
    for c in range(4):
        edges = adj[c]
        if len(edges) < 3:
            continue
        # Build adjacency map for this color
        nbr = {}
        for (u, v) in edges:
            if u not in nbr:
                nbr[u] = set()
            if v not in nbr:
                nbr[v] = set()
            nbr[u].add(v)
            nbr[v].add(u)
        # Check for triangles
        for (u, v) in edges:
            if u not in nbr or v not in nbr:
                continue
            # Find common neighbors
            if len(nbr[u]) < len(nbr[v]):
                small, big = nbr[u], nbr[v]
            else:
                small, big = nbr[v], nbr[u]
            for k in small:
                if k != u and k != v and k in big:
                    return (min(u, v, k), min(max(u, v), max(u, v, k)), max(u, v, k))
    return None

def _count_conflicts(C, u, v, new_c, n):
    """Count how many k would form a monochromatic triangle with edge (u,v) color new_c."""
    count = 0
    for k in range(n):
        if k == u or k == v:
            continue
        if C[u][k] == new_c and C[v][k] == new_c:
            count += 1
    return count

def _local_search(colors, n, C, adj, rng, deadline):
    """Local search with conflict counting.
    Returns (colors, n, C, adj) at termination."""
    stuck = 0
    MAX_STUCK = 3000
    
    while time.time() < deadline - 1.0:
        tri = _find_mono_triangle(C, adj, n)
        if tri is None:
            return colors, n, C, adj
        
        i, j, k = tri
        edges = [(i, j), (i, k), (j, k)]
        rng.shuffle(edges)
        
        flipped = False
        for (u, v) in edges:
            if u > v:
                u, v = v, u
            idx = _edge_index(u, v, n)
            old_c = colors[idx]
            
            # Find the best new color (fewest conflicts)
            best_new_c = -1
            best_conflicts = n + 1
            for new_c in range(4):
                if new_c == old_c:
                    continue
                conflicts = _count_conflicts(C, u, v, new_c, n)
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_new_c = new_c
            
            if best_new_c >= 0 and best_conflicts == 0:
                # Safe flip
                colors[idx] = best_new_c
                C[u][v] = best_new_c
                C[v][u] = best_new_c
                adj[old_c].discard((u, v))
                adj[best_new_c].add((u, v))
                flipped = True
                break
            elif best_new_c >= 0 and best_conflicts <= 1:
                # Acceptable flip
                if rng.random() < 0.5:
                    colors[idx] = best_new_c
                    C[u][v] = best_new_c
                    C[v][u] = best_new_c
                    adj[old_c].discard((u, v))
                    adj[best_new_c].add((u, v))
                    flipped = True
                    break
        
        if not flipped:
            stuck += 1
            if stuck > MAX_STUCK:
                # Perturbation: random flips
                for _ in range(20):
                    u = rng.randint(0, n - 1)
                    v = rng.randint(0, n - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    idx = _edge_index(u, v, n)
                    new_c = rng.randint(0, 3)
                    old_c = colors[idx]
                    if new_c == old_c:
                        continue
                    colors[idx] = new_c
                    C[u][v] = new_c
                    C[v][u] = new_c
                    adj[old_c].discard((u, v))
                    adj[new_c].add((u, v))
                stuck = 0
        else:
            stuck = 0
    
    return colors, n, C, adj

def _try_add_vertex(C, adj, colors, n, new_vertex_idx, deadline):
    """Try to add a new vertex (index n) to the coloring.
    Returns (new_colors, new_n) or None if no valid assignment found."""
    # New vertex is at index n
    # We need to assign colors to edges (i, n) for i in range(n)
    # such that no monochromatic triangle is formed
    
    # For each edge (i, n) with color c, we need:
    # For all k < i: NOT (C[k][i] == c AND C[k][n] == c)
    # But C[k][n] is also being assigned, so this is a constraint satisfaction problem
    
    # Use backtracking with forward checking
    # Order: assign edges from i=0 to i=n-1
    # For edge (i, n) with color c, check conflicts with already-assigned edges
    
    # Build a list of constraints
    # When assigning color c to edge (i, n):
    # For each k < i that has C[k][n] already assigned (but we're going in order, so k < i means k is already assigned if we go i=0,1,...)
    # Wait, we're assigning in order i=0,1,...,n-1, so when we assign (i,n), all (k,n) for k<i are already assigned.
    # Conflict: exists k < i such that C[k][i] == c AND C[k][n] == c
    
    # Let's do backtracking
    assignment = [-1] * n  # assignment[i] = color of edge (i, n)
    
    def backtrack(i):
        if time.time() > deadline - 1.0:
            return False
        if i == n:
            return True
        
        # Try each color for edge (i, n)
        # Order colors by fewest conflicts
        color_order = []
        for c in range(4):
            conflicts = 0
            for k in range(i):
                if C[k][i] == c and assignment[k] == c:
                    conflicts += 1
            color_order.append((conflicts, c))
        color_order.sort()
        
        for _, c in color_order:
            # Check if assigning c to (i, n) creates any conflict
            valid = True
            for k in range(i):
                if C[k][i] == c and assignment[k] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            assignment[i] = c
            if backtrack(i + 1):
                return True
            assignment[i] = -1
        
        return False
    
    if backtrack(0):
        # Build new colors array
        new_colors = colors[:]
        for i in range(n):
            new_colors.append(assignment[i])
        new_n = n + 1
        return new_colors, new_n
    
    return None

def _grow_coloring(colors, n, rng, deadline):
    """Try to grow a valid coloring by adding vertices one at a time.
    Returns (best_colors, best_n) found."""
    best_colors = colors[:]
    best_n = n
    
    current_colors = colors[:]
    current_n = n
    
    # Build state
    C, adj = _build_state(current_colors, current_n)
    
    for target_n in range(n + 1, min(n + 20, 51)):
        if time.time() > deadline - 2.0:
            break
        
        # Try to add one vertex
        result = _try_add_vertex(C, adj, current_colors, current_n, target_n - 1, deadline)
        if result is not None:
            current_colors, current_n = result
            best_colors = current_colors[:]
            best_n = current_n
            # Rebuild state
            C, adj = _build_state(current_colors, current_n)
        else:
            # Try local search on current state to see if we can make it easier to grow
            # Or just try a few random perturbations and retry
            break
    
    return best_colors, best_n

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

def construct():
    start = time.time()
    deadline = start + 115.0  # leave 5s margin before 120s kill
    rng = random.Random(12345)
    
    # Start with baseline
    best = _baseline32()
    best_n = best['n']
    best_colors = best['colors'][:]
    
    # Try to grow from baseline
    C, adj = _build_state(best_colors, best_n)
    
    # Attempt growth multiple times with different random seeds for perturbation
    for attempt in range(5):
        if time.time() > deadline - 10.0:
            break
        
        # Use the current best as starting point
        cur_colors = best_colors[:]
        cur_n = best_n
        
        # Try to grow
        new_colors, new_n = _grow_coloring(cur_colors, cur_n, rng, deadline)
        
        if new_n > best_n:
            best_n = new_n
            best_colors = new_colors[:]
        
        # Also try local search on a random K_50 to see if we can find a valid one
        if time.time() < deadline - 15.0:
            # Try local search from random initialization for n=50
            n_try = 50
            colors_try = [rng.randint(0, 3) for _ in range(n_try * (n_try - 1) // 2)]
            C_try, adj_try = _build_state(colors_try, n_try)
            colors_try, n_out, C_try, adj_try = _local_search(colors_try, n_try, C_try, adj_try, rng, deadline)
            if _find_mono_triangle(C_try, adj_try, n_out) is None:
                if n_out > best_n:
                    best_n = n_out
                    best_colors = colors_try[:]
        
        # Try growing from the locally-searched result if it's valid
        if time.time() < deadline - 10.0 and best_n < 50:
            C_cur, adj_cur = _build_state(best_colors, best_n)
            new_colors, new_n = _grow_coloring(best_colors, best_n, rng, deadline)
            if new_n > best_n:
                best_n = new_n
                best_colors = new_colors[:]
    
    # Final validation
    if time.time() < deadline - 1.0:
        n = best_n
        colors = best_colors
        if len(colors) != n * (n - 1) // 2:
            best = _baseline32()
        else:
            C, adj = _build_state(colors, n)
            if _find_mono_triangle(C, adj, n) is not None:
                best = _baseline32()
            else:
                best = {'n': n, 'colors': colors}
    
    return best

# EVOLVE-BLOCK-END
