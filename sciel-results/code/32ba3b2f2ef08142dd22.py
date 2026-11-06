"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _is_valid_coloring(n, colors):
    """Check if coloring has no monochromatic C4 using numpy matrix products."""
    if len(colors) != n * (n - 1) // 2:
        return False
    # Build adjacency matrices per color
    adjs = [np.zeros((n, n), dtype=np.uint8) for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 4:
                return False
            adjs[c][i, j] = 1
            adjs[c][j, i] = 1
            idx += 1
    # For each color, A^2 off-diagonal entries must be <= 1
    for c in range(5):
        A2 = adjs[c] @ adjs[c]
        # Zero out diagonal for check
        np.fill_diagonal(A2, 0)
        if np.max(A2) > 1:
            return False
    return True

def _lazy_woldar_25():
    """Construct a valid 5-coloring of K_25 using Z_5 x Z_5."""
    n = 25
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = i // 5, i % 5
            xj, yj = j // 5, j % 5
            if xi == xj:
                c = 4
            else:
                dx = (xj - xi) % 5
                dy = (yj - yi) % 5
                # dx^{-1} mod 5
                inv = pow(dx, -1, 5)
                c = (dy * inv) % 5
            colors.append(c)
    return n, colors

def _lazy_woldar_26():
    """
    Attempt to construct a valid 5-coloring of K_26 using a finite field approach.
    This is based on the Lazebnik-Woldar construction ideas, adapted for n=26.
    
    We use the group Z_26 is not a field, so we use a different approach:
    We try to map vertices to pairs (a, b) where a in Z_2, b in Z_13, 
    and use the structure of Z_13 (a prime field) to define colors.
    
    Actually, let's try a direct algebraic construction for n=26.
    We can use the fact that 26 = 2 * 13. Let vertices be (x, y) where x in {0,1}, y in Z_13.
    
    For two vertices (x1, y1) and (x2, y2):
    If x1 == x2, color = 4 (or some fixed color)
    If x1 != x2, we have a "cross" edge. We need to assign colors 0-3 based on y1, y2.
    
    This is tricky. Let's try a different known construction or a smarter search.
    
    Alternative: Use the known result that R_5(C4) >= 26. 
    The construction often uses a 5-coloring of K_26.
    
    Let's try to generate a candidate using a heuristic based on finite fields.
    We'll use Z_26 but that's not a field.
    
    Let's try: vertices are 0..25.
    Color(i, j) = (i - j) mod 5 ? No, that's for cyclic.
    
    Let's use the "Latin square" approach or similar.
    
    Actually, a known construction for R_5(C4) >= 26 uses the group Z_2 x Z_13.
    Vertices: (a, b) with a in {0, 1}, b in {0, ..., 12}.
    Map to index: a * 13 + b.
    
    For two vertices (a1, b1) and (a2, b2):
    If a1 == a2:
        Color = 4
    Else (a1 != a2):
        Let d = (b2 - b1) mod 13.
        We need to map d in {1, ..., 12} to colors {0, 1, 2, 3}.
        But there are 12 differences and 4 colors.
        We can use the quadratic residues in Z_13.
        Quadratic residues mod 13: 1, 4, 9, 3, 12, 10 (6 values).
        Non-residues: 2, 5, 6, 7, 8, 11 (6 values).
        
        This gives 2 classes. We need 4.
        
        Maybe use a different structure.
        
    Let's try a simpler heuristic: Random restart local search for n=26, 
    but with a better initialization and conflict counting.
    """
    n = 26
    # This function is a placeholder for now, we'll rely on local search in construct()
    # But let's try a specific algebraic guess.
    
    # Try: Color(i, j) = (i - j) mod 5 if i, j < 25, and handle vertex 25 specially?
    # No, that breaks symmetry.
    
    # Let's return a random coloring for now, and let local search fix it.
    # Or better, use the 25-vertex construction and add one vertex.
    return None

def _count_conflicts_fast(colors, n):
    """Count total number of monochromatic C4s (excess common neighbors)."""
    # Use bitmasks for speed
    adj = [0] * (5 * n)  # adj[c*n + v] = bitmask of neighbors of v in color c
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c * n + i] |= (1 << j)
            adj[c * n + j] |= (1 << i)
            idx += 1
    
    total_conf = 0
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[c * n + i] & adj[c * n + j]
                cnt = bin(common).count('1')
                if cnt > 1:
                    total_conf += cnt - 1
    return total_conf

def _local_search_26_optimized(n, initial_colors, deadline):
    """
    Optimized local search to find valid 26-vertex coloring.
    Uses incremental conflict update for speed.
    """
    n_edges = n * (n - 1) // 2
    
    # Precompute edge index
    def get_edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - (i * (i - 1)) // 2 + (j - i - 1)

    # Build initial adjacency bitsets
    adj = [0] * (5 * n)
    def rebuild_adj(colors):
        for c in range(5):
            for v in range(n):
                adj[c * n + v] = 0
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c * n + i] |= (1 << j)
                adj[c * n + j] |= (1 << i)
                idx += 1

    def count_conflicts():
        total = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[c * n + i] & adj[c * n + j]
                    cnt = bin(common).count('1')
                    if cnt > 1:
                        total += cnt - 1
        return total

    rebuild_adj(initial_colors)
    current_conf = count_conflicts()
    if current_conf == 0:
        return initial_colors[:]

    best_conf = current_conf
    best_colors = initial_colors[:]

    step = 0
    while time.time() < deadline:
        step += 1
        if current_conf == 0:
            return initial_colors[:]

        # Pick a random edge
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i

        idx = get_edge_idx(i, j)
        old_c = initial_colors[idx]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue

        # Remove edge from old color
        adj[old_c * n + i] &= ~(1 << j)
        adj[old_c * n + j] &= ~(1 << i)
        # Add edge to new color
        adj[new_c * n + i] |= (1 << j)
        adj[new_c * n + j] |= (1 << i)

        new_conf = count_conflicts()
        if new_conf < current_conf or (new_conf == current_conf and random.random() < 0.3):
            current_conf = new_conf
            initial_colors[idx] = new_c
            if new_conf < best_conf:
                best_conf = new_conf
                best_colors = initial_colors[:]
        else:
            # Revert
            adj[new_c * n + i] &= ~(1 << j)
            adj[new_c * n + j] &= ~(1 << i)
            adj[old_c * n + i] |= (1 << j)
            adj[old_c * n + j] |= (1 << i)

        if step % 1000 == 0 and current_conf > best_conf:
            if random.random() < 0.1:
                initial_colors = best_colors[:]
                rebuild_adj(initial_colors)
                current_conf = best_conf

    if best_conf == 0:
        return best_colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin

    # Start with the known 25-vertex construction
    n, colors = _lazy_woldar_25()
    best_n, best_colors = n, colors[:]

    # Verify base construction
    if not _is_valid_coloring(n, colors):
        n_small = 11
        return {'n': n_small, 'colors': [min(j - i, n_small - j + i) - 1 for i in range(n_small) for j in range(i + 1, n_small)]}

    # Try local search for 26 vertices with multiple restarts
    # Initialize edges from vertex 25 to 0..24 randomly
    # We will try several different initializations
    
    found_26 = False
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random initialization for the 25 new edges
        init_26 = colors[:] + [random.randint(0, 4) for _ in range(25)]
        
        ls_result = _local_search_26_optimized(26, init_26, deadline - 10)
        if ls_result is not None:
            if _is_valid_coloring(26, ls_result):
                return {'n': 26, 'colors': ls_result}
            found_26 = True
    
    # If all else fails, return the 25-vertex solution
    return {'n': 25, 'colors': best_colors}
# EVOLVE-BLOCK-END
