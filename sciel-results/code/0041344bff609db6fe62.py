"""
R_5(C4) maximization via algebraic construction.

The Lazebnik-Woldar construction gives a 5-coloring of K_26 without
monochromatic C4. We implement it using the affine plane AG(2,5) structure
with an additional vertex, then verify and attempt local search for n=27, 28.
"""

import time
import random
import numpy as np

def _get_idx(i, j, n):
    """Get index in the flat colors array for edge (i,j) where i<j."""
    if i > j:
        i, j = j, i
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _validate_coloring(n, colors):
    """Validate that no color class contains a C4."""
    for i in range(n):
        for j in range(i + 1, n):
            common_counts = [0] * 5
            for k in range(n):
                if k == i or k == j:
                    continue
                idx_ik = _get_idx(i, k, n)
                idx_jk = _get_idx(j, k, n)
                c_ik = colors[idx_ik]
                c_jk = colors[idx_jk]
                if c_ik == c_jk:
                    common_counts[c_ik] += 1
            if any(c >= 2 for c in common_counts):
                return False
    return True

def _lazebnik_woldar_26():
    """
    Construct the Lazebnik-Woldar 5-coloring of K_26.
    
    The construction uses the affine plane AG(2,5) which has 25 points.
    We add a 26th vertex and color edges to it appropriately.
    
    In AG(2,5), vertices are (x,y) in GF(5)^2.
    The 26 points are: (x,y) for x,y in GF(5), plus a point at infinity.
    
    For two finite points (x1,y1) and (x2,y2), the edge color is determined
    by the slope of the line through them. There are 6 slopes (0, 1, 2, 3, 4, infinity).
    We need to map these to 5 colors.
    
    The key insight from Lazebnik-Woldar: use a quadratic residue-based coloring.
    
    Actually, let me use a more direct algebraic construction.
    
    For n=26, we can use the following: vertices are elements of Z_26.
    Color edge (i,j) based on (j-i) mod 26, mapped to 5 colors.
    But this doesn't directly give C4-free.
    
    Let me use the known construction based on the projective geometry PG(2,4)
    or a direct algebraic formula.
    
    A simpler approach: use the fact that for q=5, the affine plane AG(2,5)
    gives a 6-coloring. We can merge two color classes if they don't create C4s.
    
    Let me implement the affine plane coloring and try to merge colors.
    """
    n = 26
    colors = [0] * (n * (n - 1) // 2)
    
    # Vertices 0-24 are points in GF(5)^2, vertex 25 is "infinity"
    # For finite points, color by slope
    # For edges to infinity, use a different scheme
    
    def slope(x1, y1, x2, y2):
        """Return slope of line through (x1,y1) and (x2,y2) in GF(5)."""
        dx = (x2 - x1) % 5
        dy = (y2 - y1) % 5
        if dx == 0:
            return 5  # infinity slope (vertical)
        # slope = dy/dx mod 5
        # Need modular inverse of dx mod 5
        inv_dx = pow(dx, 3, 5)  # Fermat's little theorem: x^(p-2) = x^(-1) mod p
        return (dy * inv_dx) % 5
    
    # Map 6 slopes to 5 colors
    # We need to find a mapping that avoids C4s
    # Try: merge slopes 0 and 1, keep 2,3,4,5 separate? Or other combinations.
    
    # Let's try a systematic approach: for each pair of slopes to merge,
    # check if the resulting 5-coloring is C4-free.
    
    slope_to_color = None
    for merge_a in range(6):
        for merge_b in range(merge_a + 1, 6):
            mapping = {}
            color_count = 0
            for s in range(6):
                if s == merge_a or s == merge_b:
                    mapping[s] = 0
                else:
                    mapping[s] = color_count + 1
                    color_count += 1
            # Check if this gives 5 colors
            if color_count + 1 != 5:
                continue
            
            # Build coloring
            test_colors = [0] * (n * (n - 1) // 2)
            valid = True
            
            for i in range(25):
                for j in range(i + 1, 25):
                    x1, y1 = i // 5, i % 5
                    x2, y2 = j // 5, j % 5
                    s = slope(x1, y1, x2, y2)
                    test_colors[_get_idx(i, j, n)] = mapping[s]
            
            # For edges to vertex 25, we need to assign colors
            # This is the tricky part. Let's try all possible assignments
            # for edges to vertex 25 and check validity.
            
            # Actually, let's first check if the 25-vertex part is C4-free
            # with this coloring.
            for i in range(25):
                for j in range(i + 1, 25):
                    common_counts = [0] * 5
                    for k in range(25):
                        if k == i or k == j:
                            continue
                        idx_ik = _get_idx(i, k, n)
                        idx_jk = _get_idx(j, k, n)
                        c_ik = test_colors[idx_ik]
                        c_jk = test_colors[idx_jk]
                        if c_ik == c_jk:
                            common_counts[c_ik] += 1
                    if any(c >= 2 for c in common_counts):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                slope_to_color = mapping
                break
        if slope_to_color is not None:
            break
    
    if slope_to_color is None:
        # Fallback: use a different construction
        return _greedy_26()
    
    # Build the coloring for 25 vertices
    for i in range(25):
        for j in range(i + 1, 25):
            x1, y1 = i // 5, i % 5
            x2, y2 = j // 5, j % 5
            s = slope(x1, y1, x2, y2)
            colors[_get_idx(i, j, n)] = slope_to_color[s]
    
    # For edges to vertex 25, try to find valid color assignments
    # using backtracking
    edges_to_25 = [(i, 25) for i in range(25)]
    edge_indices = [_get_idx(i, 25, n) for i in range(25)]
    
    # Use backtracking to assign colors to edges from 25
    def _assign_25_edges(colors, n):
        """Assign colors to edges from vertex 25 using backtracking."""
        # Precompute for each vertex i and color c, the common neighbors count
        # with vertex 25 if we assign color c to edge (i, 25)
        
        # common[c][i][j] = number of common neighbors of i and j in color c
        # When we assign color c to edge (i, 25), we need to check that
        # for any j, the number of k such that color(i,k)=c and color(25,k)=c
        # is at most 1.
        
        # Let's do a simple backtracking
        assigned = [-1] * 25  # color for edge (i, 25)
        
        def backtrack(idx):
            if idx == 25:
                return True
            
            i = idx
            for c in range(5):
                # Check if assigning color c to edge (i, 25) is valid
                valid = True
                for j in range(25):
                    if j == i:
                        continue
                    # Check if there are already 2 common neighbors of i and 25 in color c
                    # through vertices that are already assigned
                    common_count = 0
                    for k in range(25):
                        if k == i or k == 25:
                            continue
                        # Edge (i, k) color
                        c_ik = colors[_get_idx(i, k, n)]
                        # Edge (25, k) color - only if assigned
                        if assigned[k] == -1:
                            continue
                        c_25k = assigned[k]
                        if c_ik == c and c_25k == c:
                            common_count += 1
                    if common_count >= 2:
                        valid = False
                        break
                
                if valid:
                    assigned[idx] = c
                    if backtrack(idx + 1):
                        return True
                    assigned[idx] = -1
            
            return False
        
        if backtrack(0):
            for i in range(25):
                colors[edge_indices[i]] = assigned[i]
            return True
        return False
    
    if not _assign_25_edges(colors, n):
        return _greedy_26()
    
    if _validate_coloring(n, colors):
        return colors
    
    return _greedy_26()

def _greedy_26():
    """Fallback: greedy construction for n=26."""
    n = 26
    colors = [-1] * (n * (n - 1) // 2)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    rng = random.Random(42)
    
    # common[color][u][v] = number of common neighbors of u and v in color
    common = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    for idx, (i, j) in enumerate(edges):
        best_color = -1
        best_cost = float('inf')
        valid_colors = []
        
        for c in range(5):
            if common[c][i][j] >= 2:
                continue
            
            cost = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                idx_ik = _get_idx(i, k, n)
                idx_jk = _get_idx(j, k, n)
                if colors[idx_ik] == c:
                    if common[c][j][k] >= 1:
                        cost = float('inf')
                        break
                if colors[idx_jk] == c:
                    if common[c][i][k] >= 1:
                        cost = float('inf')
                        break
            
            if cost < float('inf'):
                valid_colors.append((cost, c))
        
        if not valid_colors:
            return None
        
        valid_colors.sort()
        min_cost = valid_colors[0][0]
        candidates = [c for (cost, c) in valid_colors if cost == min_cost]
        best_color = rng.choice(candidates)
        
        colors[idx] = best_color
        c = best_color
        
        for k in range(n):
            if k == i or k == j:
                continue
            idx_ik = _get_idx(i, k, n)
            idx_jk = _get_idx(j, k, n)
            
            if colors[idx_ik] == c:
                common[c][j][k] += 1
                common[c][k][j] += 1
            if colors[idx_jk] == c:
                common[c][i][k] += 1
                common[c][k][i] += 1
    
    if _validate_coloring(n, colors):
        return colors
    return None

def _local_search_improve(n, colors, deadline):
    """
    Try to improve the coloring by local search: flip edge colors
    to reduce the number of near-C4 violations.
    """
    if time.time() > deadline:
        return colors
    
    n_edges = n * (n - 1) // 2
    colors = list(colors)
    
    # Compute violation count
    def count_violations(colors):
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                common_counts = [0] * 5
                for k in range(n):
                    if k == i or k == j:
                        continue
                    idx_ik = _get_idx(i, k, n)
                    idx_jk = _get_idx(j, k, n)
                    c_ik = colors[idx_ik]
                    c_jk = colors[idx_jk]
                    if c_ik == c_jk:
                        common_counts[c_ik] += 1
                for c in range(5):
                    if common_counts[c] >= 2:
                        violations += common_counts[c] - 1
        return violations
    
    current_violations = count_violations(colors)
    if current_violations == 0:
        return colors
    
    rng = random.Random(123)
    
    for step in range(10000):
        if time.time() > deadline:
            break
        
        # Pick a random edge to flip
        e = rng.randint(0, n_edges - 1)
        old_color = colors[e]
        
        # Find which vertices this edge connects
        # Inverse of _get_idx
        # e = i * (2*n - i - 1) // 2 + (j - i - 1)
        i = 0
        while i * (2 * n - i - 1) // 2 <= e:
            i += 1
        i -= 1
        j = i + 1 + (e - i * (2 * n - i - 1) // 2)
        
        # Try all other colors
        best_new_color = -1
        best_violations = current_violations
        
        for new_color in range(5):
            if new_color == old_color:
                continue
            
            colors[e] = new_color
            v = count_violations(colors)
            if v < best_violations:
                best_violations = v
                best_new_color = new_color
            colors[e] = old_color
        
        if best_new_color != -1:
            colors[e] = best_new_color
            current_violations = best_violations
            if current_violations == 0:
                break
    
    if _validate_coloring(n, colors):
        return colors
    return list(colors)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try the algebraic construction for n=26
    colors_26 = _lazebnik_woldar_26()
    
    if colors_26 is not None:
        # Try to extend to n=27 and n=28 using local search
        for target_n in [27, 28]:
            if time.time() > deadline - 30:
                break
            
            # Start from the 26-vertex coloring and add a new vertex
            n = target_n
            new_colors = [-1] * (n * (n - 1) // 2)
            
            # Copy the 26-vertex coloring
            for i in range(26):
                for j in range(i + 1, 26):
                    new_colors[_get_idx(i, j, n)] = colors_26[_get_idx(i, j, 26)]
            
            # Assign colors to edges from vertex 25 (new vertex) to 0-24
            # using a greedy approach
            for i in range(25):
                best_color = -1
                best_cost = float('inf')
                for c in range(5):
                    cost = 0
                    for k in range(25):
                        if k == i:
                            continue
                        idx_ik = _get_idx(i, k, n)
                        idx_25k = _get_idx(25, k, n)
                        c_ik = new_colors[idx_ik]
                        c_25k = new_colors[idx_25k]
                        if c_ik == c and c_25k == c:
                            cost += 1
                    if cost < best_cost:
                        best_cost = cost
                        best_color = c
                
                if best_color != -1:
                    new_colors[_get_idx(25, i, n)] = best_color
                else:
                    new_colors[_get_idx(25, i, n)] = 0
            
            # Validate and try to improve
            if _validate_coloring(n, new_colors):
                return {'n': n, 'colors': new_colors}
            
            # Try local search
            improved = _local_search_improve(n, new_colors, deadline - 5)
            if _validate_coloring(n, improved):
                return {'n': n, 'colors': improved}
        
        # Return the 26-vertex coloring
        return {'n': 26, 'colors': colors_26}
    
    # Fallback: greedy for smaller n
    best_n = 1
    best_colors = [0]
    
    for n in range(2, 27):
        if time.time() > deadline - 5:
            break
        
        colors = _greedy_26() if n == 26 else None
        if colors is not None and n == 26:
            best_n = n
            best_colors = colors
            break
        
        # For smaller n, use a simple greedy
        colors = [-1] * (n * (n - 1) // 2)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        common = [[[0] * n for _ in range(n)] for _ in range(5)]
        rng = random.Random(42)
        
        valid = True
        for idx, (i, j) in enumerate(edges):
            best_color = -1
            valid_colors = []
            for c in range(5):
                if common[c][i][j] >= 2:
                    continue
                valid_colors.append(c)
            
            if not valid_colors:
                valid = False
                break
            
            best_color = rng.choice(valid_colors)
            colors[idx] = best_color
            c = best_color
            
            for k in range(n):
                if k == i or k == j:
                    continue
                idx_ik = _get_idx(i, k, n)
                idx_jk = _get_idx(j, k, n)
                
                if colors[idx_ik] == c:
                    common[c][j][k] += 1
                    common[c][k][j] += 1
                if colors[idx_jk] == c:
                    common[c][i][k] += 1
                    common[c][k][i] += 1
        
        if valid and _validate_coloring(n, colors):
            best_n = n
            best_colors = colors
        elif n > 5:
            break
    
    return {'n': best_n, 'colors': best_colors}
