"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

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

def _construct_agn(n):
    """
    Construct a coloring using the affine plane AG(2,5) structure.
    This gives a valid C4-free coloring for n=25.
    Vertices are points in GF(5)^2.
    Edge color is determined by the slope of the line connecting them.
    There are 5+1=6 possible slopes (including infinity).
    We map these 6 slopes to 5 colors by merging two slopes.
    
    Actually, for AG(2,q), the standard construction for C4-free colorings
    uses q+1 colors for n=q^2. Here q=5, so we need 6 colors for n=25.
    But we only have 5 colors. So we need a different approach.
    
    Alternative: Use the fact that the projective plane PG(2,4) has 21 points
    and each line has 5 points. But that's not directly applicable.
    
    Let me think differently. The Lazebnik-Woldar construction for n=26
    uses a specific algebraic structure. Let me try a different approach:
    
    For n=25, we can use a 5x5 grid. Color edge (i,j) where i=(r1,c1), j=(r2,c2)
    based on (r2-r1, c2-c1) mod 5. There are 24 non-zero differences.
    We need to map these to 5 colors such that no C4 is formed.
    
    A C4 in the grid would be four points forming a rectangle.
    If we color based on the difference vector, a rectangle (r1,c1),(r1,c2),(r2,c1),(r2,c2)
    has edges with differences:
    (0, c2-c1), (r2-r1, 0), (0, c2-c1), (r2-r1, 0)
    So the horizontal edges have the same color, and vertical edges have the same color.
    This creates a C4 if the horizontal and vertical colors are the same, but they're different.
    Wait, a C4 requires four edges of the same color. Here we have two horizontal and two vertical.
    If horizontal color != vertical color, then no monochromatic C4 from this rectangle.
    
    But we need to ensure that for ANY four points forming a C4 in the complete graph,
    the four edges are not all the same color.
    
    Let me try: color = (dr * 5 + dc) % 5 where dr = (r2-r1)%5, dc = (c2-c1)%5.
    This gives 25 possible values, but we only use 5 colors.
    Actually, (dr, dc) has 24 non-zero pairs. We map each to a color in {0,1,2,3,4}.
    
    A C4 in the complete graph on the grid points: four distinct points A,B,C,D
    such that edges AB, BC, CD, DA all have the same color.
    
    This is complex. Let me try a simpler approach: use the known fact that
    the affine plane AG(2,5) gives a 6-coloring of K_25 without C4.
    We need to reduce to 5 colors. We can try to merge two colors and check if it's still valid.
    
    For now, let me implement a greedy approach with better heuristics, and also
    try to use the AG(2,5) structure as a starting point.
    """
    if n != 25:
        return None
    
    colors = [0] * (n * (n - 1) // 2)
    
    # Map (r, c) to vertex index
    def idx(r, c):
        return r * 5 + c
    
    # For each pair of vertices, compute the difference vector and assign a color
    # There are 24 non-zero difference vectors in GF(5)^2
    # Group them into 5 colors. We need to ensure no C4.
    
    # A C4 corresponds to four points A, B, C, D such that
    # B-A = D-C and C-B = A-D (parallelogram)
    # So if we color edge (P, Q) by the direction Q-P, then a parallelogram
    # has two pairs of parallel edges with the same direction.
    # If the two directions are assigned the same color, we get a C4.
    
    # So we need to assign colors to the 24 directions such that
    # no two directions that form a parallelogram get the same color.
    # But every pair of directions (u, v) can form a parallelogram (take any point,
    # then P+u, P+v, P+u+v).
    # So we need all 24 directions to have distinct colors? But we only have 5.
    
    # This approach doesn't work directly. Let me use a different method.
    
    # Use the fact that the 5-coloring of K_25 without C4 exists (from AG(2,5) with merged colors).
    # I'll use a known construction: color edge (i,j) by the quadratic residue of the difference.
    
    # Actually, let me just use a greedy approach with multiple restarts.
    return None

def _greedy_cost_minimize(n, deadline, seed=42):
    """
    Greedy coloring with cost minimization.
    """
    rng = random.Random(seed)
    n_edges = n * (n - 1) // 2
    colors = [-1] * n_edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # common[color][u][v] = number of common neighbors of u and v in color
    common = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    for idx, (i, j) in enumerate(edges):
        if time.time() > deadline:
            return None
        
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
                idx_jk = _get_idx(j, k, n)
                if colors[idx_jk] == c:
                    if common[c][i][k] == 0:
                        cost += 1
                    elif common[c][i][k] >= 1:
                        cost = float('inf')
                        break
                
                idx_ik = _get_idx(i, k, n)
                if colors[idx_ik] == c:
                    if common[c][j][k] == 0:
                        cost += 1
                    elif common[c][j][k] >= 1:
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

def _backtracking_csp(n, deadline, seed=42):
    """
    Backtracking CSP solver with forward checking and MRV heuristic.
    """
    rng = random.Random(seed)
    n_edges = n * (n - 1) // 2
    colors = [-1] * n_edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # common[color][u][v] = number of common neighbors of u and v in color
    common = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    # Track which edges are assigned
    assigned = [False] * n_edges
    
    # For each unassigned edge, track which colors are still possible
    # We'll use a simple approach: at each step, pick the edge with the fewest valid colors
    
    def _count_valid_colors(edge_idx):
        i, j = edges[edge_idx]
        count = 0
        for c in range(5):
            if common[c][i][j] < 2:
                # Check if assigning c would create a C4
                valid = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    idx_ik = _get_idx(i, k, n)
                    idx_jk = _get_idx(j, k, n)
                    if colors[idx_ik] == c and colors[idx_jk] == c:
                        if common[c][i][k] >= 1 or common[c][j][k] >= 1:
                            valid = False
                            break
                if valid:
                    count += 1
        return count
    
    def _try_assign(edge_idx, c):
        """Try to assign color c to edge edge_idx. Returns True if successful."""
        i, j = edges[edge_idx]
        
        # Check if this assignment is valid
        if common[c][i][j] >= 2:
            return False
        
        for k in range(n):
            if k == i or k == j:
                continue
            idx_ik = _get_idx(i, k, n)
            idx_jk = _get_idx(j, k, n)
            if colors[idx_ik] == c and colors[idx_jk] == c:
                if common[c][i][k] >= 1 or common[c][j][k] >= 1:
                    return False
        
        # Assign
        colors[edge_idx] = c
        assigned[edge_idx] = True
        
        # Update common counts
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
        
        return True
    
    def _unassign(edge_idx):
        """Unassign the color from edge edge_idx."""
        i, j = edges[edge_idx]
        c = colors[edge_idx]
        
        # Update common counts
        for k in range(n):
            if k == i or k == j:
                continue
            idx_ik = _get_idx(i, k, n)
            idx_jk = _get_idx(j, k, n)
            
            if colors[idx_ik] == c:
                common[c][j][k] -= 1
                common[c][k][j] -= 1
            if colors[idx_jk] == c:
                common[c][i][k] -= 1
                common[c][k][i] -= 1
        
        colors[edge_idx] = -1
        assigned[edge_idx] = False
    
    # Backtracking search
    def backtrack(depth):
        if time.time() > deadline:
            return False
        
        if depth == n_edges:
            return True
        
        # Find the unassigned edge with the fewest valid colors (MRV)
        best_edge = -1
        best_count = float('inf')
        for e in range(n_edges):
            if not assigned[e]:
                count = _count_valid_colors(e)
                if count < best_count:
                    best_count = count
                    best_edge = e
                    if count == 0:
                        break
        
        if best_count == 0:
            return False
        
        # Try each valid color for best_edge
        i, j = edges[best_edge]
        valid_cs = []
        for c in range(5):
            if common[c][i][j] < 2:
                valid = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    idx_ik = _get_idx(i, k, n)
                    idx_jk = _get_idx(j, k, n)
                    if colors[idx_ik] == c and colors[idx_jk] == c:
                        if common[c][i][k] >= 1 or common[c][j][k] >= 1:
                            valid = False
                            break
                if valid:
                    valid_cs.append(c)
        
        # Shuffle for diversity
        rng.shuffle(valid_cs)
        
        for c in valid_cs:
            if _try_assign(best_edge, c):
                if backtrack(depth + 1):
                    return True
                _unassign(best_edge)
        
        return False
    
    if backtrack(0):
        if _validate_coloring(n, colors):
            return colors
    return None

def _construct_cyclic(n):
    """Construct a cyclic coloring of K_n using distances."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            d = min(d, n - d)
            colors.append(d % 5)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_n = 1
    best_colors = [0]
    
    # Try backtracking for small n first, then greedy for larger n
    for n in range(2, 29):
        if time.time() > deadline - 5:
            break
        
        found = None
        
        # For n <= 15, try backtracking
        if n <= 15:
            for seed in [42, 123, 777]:
                if time.time() > deadline - 3:
                    break
                result = _backtracking_csp(n, deadline - 2, seed)
                if result is not None:
                    found = result
                    break
        
        # For all n, try greedy
        if found is None:
            for seed in [42, 123, 777, 2024]:
                if time.time() > deadline - 2:
                    break
                result = _greedy_cost_minimize(n, deadline - 1, seed)
                if result is not None:
                    found = result
                    break
        
        if found is not None:
            best_n = n
            best_colors = found
        else:
            if n > 3:
                break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
