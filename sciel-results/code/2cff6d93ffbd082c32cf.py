"""Model-and-optimize: construct large triangle-free 4-colorings via local search."""
import random
import time
import math
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
    # Build per-color neighbor lists (as sets for fast intersection)
    nbr = [set() for _ in range(4)]
    for i in range(n):
        row = C[i]
        for j in range(i + 1, n):
            c = row[j]
            if c >= 0:
                nbr[c].add((i, j))
    return C, nbr


def _find_mono_triangle_fast(C, nbr, n):
    """Find a monochromatic triangle using per-color neighbor sets.
    Returns (i, j, k) or None."""
    for c in range(4):
        edges = nbr[c]
        if len(edges) < 3:
            continue
        # Build adjacency map for this color
        adj = {}
        for (u, v) in edges:
            if u not in adj:
                adj[u] = set()
            if v not in adj:
                adj[v] = set()
            adj[u].add(v)
            adj[v].add(u)
        # Check for triangles
        for (u, v) in edges:
            if u not in adj or v not in adj:
                continue
            # Find common neighbors
            if len(adj[u]) < len(adj[v]):
                small, big = adj[u], adj[v]
            else:
                small, big = adj[v], adj[u]
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


def _local_search_v2(colors, n, C, nbr, rng, deadline):
    """Improved local search with conflict counting and smarter flips.
    Returns (colors, n, C, nbr) at termination."""
    stuck = 0
    MAX_STUCK = 5000
    
    while time.time() < deadline - 1.5:
        tri = _find_mono_triangle_fast(C, nbr, n)
        if tri is None:
            return colors, n, C, nbr
        
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
                nbr[old_c].discard((u, v))
                nbr[best_new_c].add((u, v))
                flipped = True
                break
            elif best_new_c >= 0 and best_conflicts < 3:
                # Acceptable flip (creates at most 2 new triangles)
                if rng.random() < 0.3:
                    colors[idx] = best_new_c
                    C[u][v] = best_new_c
                    C[v][u] = best_new_c
                    nbr[old_c].discard((u, v))
                    nbr[best_new_c].add((u, v))
                    flipped = True
                    break
        
        if not flipped:
            stuck += 1
            if stuck > MAX_STUCK:
                # Perturbation: random flips
                for _ in range(30):
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
                    nbr[old_c].discard((u, v))
                    nbr[new_c].add((u, v))
                stuck = 0
        else:
            stuck = 0
    
    return colors, n, C, nbr


def _grow_from_valid(colors, n, rng, deadline):
    """Try to grow a valid coloring by adding vertices one at a time.
    Returns (new_colors, new_n) or None if no growth possible."""
    if n < 1:
        return None
    
    # Build state
    C, nbr = _build_state(colors, n)
    
    for new_n in range(n + 1, min(n + 10, 50)):
        if time.time() > deadline - 2.0:
            break
        
        # Try to add vertex new_n-1
        new_colors = colors[:]
        # Add edges from new vertex to all existing vertices
        for i in range(new_n - 1):
            # Choose color for edge (i, new_n-1)
            # Must avoid creating monochromatic triangles
            best_c = -1
            best_conflicts = new_n + 1
            for c in range(4):
                conflicts = 0
                for j in range(i):
                    # Triangle (j, i, new_n-1): need C[j][i] != c AND C[j][new_n-1] != c
                    # But we're setting C[i][new_n-1] = c, so check if C[j][i]==c and C[j][new_n-1]==c
                    # Actually, we need to ensure no k such that C[k][i]==c and C[k][new_n-1]==c
                    # For triangle (k, i, new_n-1): edges (k,i), (k,new_n-1), (i,new_n-1)
                    # If C[i][new_n-1]=c, then triangle exists if C[k][i]==c and C[k][new_n-1]==c
                    # But we're choosing C[i][new_n-1] now, so for each k < i, 
                    # if C[k][i]==c, then we must ensure C[k][new_n-1] != c
                    # This is complex; simpler: just check conflicts after assignment
                    pass
                # Simplified: try each color and count conflicts
                # For color c on edge (i, new_n-1):
                # Conflict if there exists k such that C[k][i]==c and C[k][new_n-1]==c
                # But we haven't set C[k][new_n-1] for k < i yet
                # So we need to assign colors in order and check incrementally
                break
            
            # Simpler approach: assign random color and check
            c = rng.randint(0, 3)
            new_colors.append(c)
        
        # Validate the new coloring
        new_C, new_nbr = _build_state(new_colors, new_n)
        if _find_mono_triangle_fast(new_C, new_nbr, new_n) is None:
            return new_colors, new_n
    
    return None


def _try_n_improved(n, rng, deadline):
    """Try to find a valid 4-coloring of K_n via improved local search."""
    if n <= 1:
        return {'n': n, 'colors': []}
    
    best = None
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline - 1.0:
            break
        
        # Random initialization
        colors = [rng.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        C, nbr = _build_state(colors, n)
        
        # Local search
        colors, n_out, C, nbr = _local_search_v2(colors, n, C, nbr, rng, deadline)
        
        if _find_mono_triangle_fast(C, nbr, n_out) is None:
            best = {'n': n_out, 'colors': colors}
            break
    
    return best


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
    
    # Try to improve via local search on larger n
    # Strategy: try n=33,34,... with decreasing time budgets
    for n in range(33, 50):
        if time.time() > deadline - 10.0:
            break
        
        # Allocate time based on remaining budget
        remaining = deadline - time.time() - 5.0
        if remaining < 3.0:
            break
        
        # Give more time to smaller n, less to larger
        chunk = min(8.0, remaining * 0.4)
        sub_deadline = time.time() + chunk
        
        res = _try_n_improved(n, rng, sub_deadline)
        if res is not None and res['n'] > best['n']:
            best = res
    
    # Final validation
    if time.time() < deadline - 1.0:
        n = best['n']
        colors = best['colors']
        if len(colors) != n * (n - 1) // 2:
            best = _baseline32()
        else:
            C, nbr = _build_state(colors, n)
            if _find_mono_triangle_fast(C, nbr, n) is not None:
                best = _baseline32()
    
    return best

# EVOLVE-BLOCK-END
