import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Try to build the Lazebnik-Woldar construction for n=26
    # This uses the affine plane of order 5 (or related structure)
    # The construction: vertices are (x,y) in Z_5 x Z_5, plus two extra points at infinity
    # Actually, the standard construction for R_5(C4) > 26 uses:
    # V = {∞_0, ∞_1, ..., ∞_4} ∪ {(x,y) : x,y ∈ Z_5}
    # That's 5 + 25 = 30 vertices, but we need exactly 26.
    
    # Let me reconsider. The Lazebnik-Woldar construction for 5-coloring C4-free:
    # Actually, the known result is R_5(C4) >= 27, achieved by a construction on 26 vertices.
    # Let me use a different approach: use the finite field GF(5) structure.
    
    # For a C4-free 5-coloring of K_26, we can use:
    # Vertices: Z_5 × Z_5 ∪ {∞_0, ∞_1, ∞_2, ∞_3} (5+25+3 = 33? No, that's too many)
    # Actually 5*5 = 25, so 25 vertices in the grid, plus 1 extra = 26.
    
    # Let me try a direct algebraic construction:
    # Vertices are elements of Z_5 × Z_5 (25 vertices) plus one vertex at infinity (26 total).
    # Color edge ((x1,y1),(x2,y2)) based on the slope (y2-y1)/(x2-x1) mod 5, if x1≠x2.
    # If x1=x2, color by 4 (vertical).
    # For edges to infinity, assign colors carefully.
    
    # Actually, let me try a simpler known construction.
    # The Paley-type construction or the one from the paper.
    
    # Let me just use a direct search with a strong heuristic, starting from n=20 and building up.
    # But first, let me try to construct n=26 directly using the known algebraic method.
    
    # Known construction for 5 colors, C4-free, n=26:
    # Use vertices Z_5 × Z_5 ∪ {∞}
    # For (x1,y1), (x2,y2) with x1 ≠ x2: color = (y2-y1) * (x2-x1)^{-1} mod 5
    # For x1 = x2: color = 4 (or some fixed color for verticals)
    # For edges to ∞: need to assign colors such that no C4 is formed.
    
    # Let me verify this works for the 25 grid vertices first.
    # Two vertices (a,b) and (c,d) have a common neighbor (e,f) in color k if:
    # (f-b)/(e-a) = k and (f-d)/(e-c) = k, with a≠e, c≠e
    # This means f = b + k(e-a) = d + k(e-c), so b - d = k(e-a) - k(e-c) = k(c-a)
    # So for given (a,b),(c,d) and color k, there's at most one e (if c≠a) that works,
    # and then f is determined. So at most 1 common neighbor per color for x-distinct pairs.
    # For x1=x2=a: common neighbor must have x≠a (since vertical edges are color 4, and we're looking at color k≠4)
    # So common neighbor (e,f) with e≠a: (f-b)/(e-a)=k and (f-d)/(e-a)=k, so b=d.
    # If b=d, then (a,b) and (a,d) are the same vertex. So no C4 from verticals.
    # Good, the 25-vertex grid is C4-free with 5 colors (4 for slopes 0-3, 4 for vertical).
    
    # Now add vertex ∞. Edges from ∞ to (x,y) need colors.
    # A C4 involving ∞ would be ∞-u-v-w-∞ where u,v,w are grid vertices.
    # Colors: c(∞,u), c(u,v), c(v,w), c(w,∞) all equal.
    # To avoid this, for each color k, the set of grid vertices connected to ∞ by color k
    # must be an independent set in the color-k graph.
    
    # Let me assign: c(∞, (x,y)) = x mod 5 (or some function).
    # Let's try c(∞, (x,y)) = y mod 5.
    # Then for color k, the neighbors of ∞ are {(x,y) : y=k}, which is a row of 5 vertices.
    # In color k, two vertices (x1,k) and (x2,k) with x1≠x2: their edge color is (k-k)/(x2-x1) = 0.
    # So if k≠0, no edge between them in color k. Good.
    # If k=0, edges between (x1,0) and (x2,0) have color 0. So they are adjacent in color 0.
    # A C4 in color 0 involving ∞: ∞-(x1,0)-(x2,0)-(x3,0)-∞? 
    # Need c(x1,0,x2,0)=0, c(x2,0,x3,0)=0, c(∞,x1,0)=0, c(∞,x3,0)=0.
    # (x1,0) and (x2,0) connected by color 0 iff x1≠x2 (slope 0).
    # So we need x1≠x2, x2≠x3, and all y=0.
    # ∞ connected to (x1,0) by color 0, and to (x3,0) by color 0.
    # This forms a path ∞-(x1,0)-(x2,0)-(x3,0)-∞ in color 0 if x1≠x2≠x3.
    # That's a C4! So this assignment doesn't work.
    
    # Let me try c(∞, (x,y)) = (x+y) mod 5.
    # For color k, neighbors of ∞ are {(x,y) : x+y=k}, a diagonal of 5 vertices.
    # Two vertices (x1,y1) and (x2,y2) on this diagonal: y1=k-x1, y2=k-x2.
    # Slope = (y2-y1)/(x2-x1) = (k-x2-k+x1)/(x2-x1) = (x1-x2)/(x2-x1) = -1.
    # So all edges between distinct vertices on the same diagonal have color -1 = 4.
    # So if k≠4, no edges between neighbors of ∞ in color k. Good.
    # If k=4, neighbors are {(x,y):x+y=4}, and edges between them have color 4.
    # C4: ∞-(x1,4-x1)-(x2,4-x2)-(x3,4-x3)-∞ in color 4?
    # Need x1≠x2, x2≠x3, and all on diagonal x+y=4.
    # This is a C4 in color 4. So this also doesn't work.
    
    # The issue is that for any linear assignment, the neighbors of ∞ in color k form an arithmetic progression
    # and have edges between them in some color.
    
    # Let me try a non-linear assignment. Or use a different construction.
    
    # Alternative: use n=25 (the grid without ∞) as the base, which is guaranteed C4-free.
    # Then try to add vertices one by one with local search.
    
    # Actually, let me just use a strong local search starting from n=20, building up to n=26.
    # With efficient incremental updates, this should be fast enough.
    
    best_n = 11
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_colors = colors_fallback
    
    # First, try to get a valid n=25 from the algebraic construction
    n_algebraic = 25
    colors_algebraic = []
    for i in range(n_algebraic):
        for j in range(i + 1, n_algebraic):
            x1, y1 = divmod(i, 5)
            x2, y2 = divmod(j, 5)
            if x1 != x2:
                dx = (x2 - x1) % 5
                dy = (y2 - y1) % 5
                # Inverse of dx mod 5
                inv_dx = pow(dx, -1, 5)
                color = (dy * inv_dx) % 5
                if color == 4:
                    # This would be vertical, but x1≠x2, so slope is not undefined
                    # Just use it as is
                    pass
                colors_algebraic.append(color)
            else:
                # Same x, different y: vertical edge
                colors_algebraic.append(4)
    
    # Verify n=25 is valid
    def count_c4s(colors, n):
        """Count total C4s across all 5 color classes."""
        mats = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                mats[c][i, j] = 1
                mats[c][j, i] = 1
                idx += 1
        
        total = 0
        for m in mats:
            M = m @ m
            # C4 count = sum over all pairs (u,v) of binom(M[u,v], 2) for u<v
            for u in range(n):
                for v in range(u + 1, n):
                    cnt = M[u, v]
                    if cnt >= 2:
                        total += cnt * (cnt - 1) // 2
        return total
    
    c4_count_25 = count_c4s(colors_algebraic, n_algebraic)
    if c4_count_25 == 0:
        best_n = n_algebraic
        best_colors = colors_algebraic[:]
    
    # Now try to extend to n=26, 27, 28 using local search
    # Start from the best solution found so far
    
    for target_n in range(best_n + 1, 29):
        if time_left() < 30:
            break
        
        num_edges = target_n * (target_n - 1) // 2
        
        # Initialize: extend the current best solution
        # If we have a solution for n_old, extend to n_new by adding vertices
        # with random colors, then local search
        
        if target_n <= 25:
            # Use the algebraic construction as starting point
            colors = colors_algebraic[:]
            # Add extra edges for new vertices
            while len(colors) < num_edges:
                colors.append(random.randint(0, 4))
        else:
            # Extend from best_colors
            colors = best_colors[:]
            while len(colors) < num_edges:
                colors.append(random.randint(0, 4))
        
        colors = colors[:num_edges]
        
        # Build adjacency matrices
        mats = [np.zeros((target_n, target_n), dtype=np.int8) for _ in range(5)]
        idx = 0
        for i in range(target_n):
            for j in range(i + 1, target_n):
                c = colors[idx]
                mats[c][i, j] = 1
                mats[c][j, i] = 1
                idx += 1
        
        # Build common neighbor matrices
        common_mats = [m @ m for m in mats]
        
        def compute_cost(common_mats):
            total = 0
            for M in common_mats:
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        cnt = M[u, v]
                        if cnt >= 2:
                            total += cnt * (cnt - 1) // 2
            return total
        
        current_cost = compute_cost(common_mats)
        
        # Local search with incremental updates
        edge_lookup = []
        idx = 0
        for i in range(target_n):
            for j in range(i + 1, target_n):
                edge_lookup.append((i, j))
                idx += 1
        
        max_restarts = 5
        for restart in range(max_restarts):
            if time_left() < 10:
                break
            
            if current_cost == 0:
                best_n = target_n
                best_colors = colors[:]
                break
            
            stuck_count = 0
            max_stuck = 5000
            while stuck_count < max_stuck:
                if time_left() < 5:
                    break
                
                if current_cost == 0:
                    break
                
                # Pick a random edge
                e_idx = random.randint(0, num_edges - 1)
                old_c = colors[e_idx]
                new_c = random.randint(0, 4)
                if old_c == new_c:
                    continue
                
                u, v = edge_lookup[e_idx]
                
                # Update: remove edge (u,v) from old_c, add to new_c
                A_old = mats[old_c]
                A_new = mats[new_c]
                
                A_old[u, v] = 0
                A_old[v, u] = 0
                A_new[u, v] = 1
                A_new[v, u] = 1
                
                common_mats[old_c] = A_old @ A_old
                common_mats[new_c] = A_new @ A_new
                
                new_cost = compute_cost(common_mats)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    colors[e_idx] = new_c
                    stuck_count = 0
                else:
                    # Revert
                    A_old[u, v] = 1
                    A_old[v, u] = 1
                    A_new[u, v] = 0
                    A_new[v, u] = 0
                    common_mats[old_c] = A_old @ A_old
                    common_mats[new_c] = A_new @ A_new
                    stuck_count += 1
            
            if current_cost == 0:
                best_n = target_n
                best_colors = colors[:]
                break
            
            # Random restart: perturb some edges
            for _ in range(20):
                e_idx = random.randint(0, num_edges - 1)
                colors[e_idx] = random.randint(0, 4)
            
            # Rebuild matrices
            for c in range(5):
                mats[c] = np.zeros((target_n, target_n), dtype=np.int8)
            idx = 0
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    c = colors[idx]
                    mats[c][i, j] = 1
                    mats[c][j, i] = 1
                    idx += 1
            common_mats = [m @ m for m in mats]
            current_cost = compute_cost(common_mats)
        
        if current_cost == 0:
            best_n = target_n
            best_colors = colors[:]
            # Continue trying to extend further
        else:
            # Couldn't solve this target_n, try next
            continue
    
    return {'n': best_n, 'colors': best_colors}
