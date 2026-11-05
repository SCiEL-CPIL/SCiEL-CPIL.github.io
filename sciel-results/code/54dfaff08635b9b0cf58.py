import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # GF(4) arithmetic: elements are 0, 1, a, a+1 where a^2 = a + 1
    # Represent as: 0->0, 1->1, a->2, a+1->3
    # Addition table (XOR-like but not exactly)
    gf4_add = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
    # Multiplication table
    gf4_mul = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
    
    # Verify: a^2 = a+1 => gf4_mul[2][2] should be 3
    # gf4_mul[2][2] = 3 ✓
    
    # Points in GF(4)^2: 16 points indexed 0-15
    # Point i = (i//4, i%4)
    
    def point_coords(i):
        return (i // 4, i % 4)
    
    def slope(p1, p2):
        """Compute slope of line through p1 and p2 in GF(4)^2"""
        x1, y1 = p1
        x2, y2 = p2
        dx = gf4_add[x1, gf4_add[x2, 15]]  # x2 - x1 (subtract = add with 15 since -1=3 in GF4? No...)
        # Actually in GF(4), subtraction is same as addition since char 2
        dx = gf4_add[x1, x2]
        dy = gf4_add[y1, y2]
        
        if dx == 0:
            return 4  # vertical slope
        # slope = dy / dx
        # Need inverse of dx in GF(4)
        # Inverses: 1->1, 2->3, 3->2, 0->undefined
        inv = {1: 1, 2: 3, 3: 2}
        if dx not in inv:
            return 4
        return gf4_mul[dy, inv[dx]]
    
    # Precompute slopes between all pairs
    slopes = [[0]*16 for _ in range(16)]
    for i in range(16):
        for j in range(i+1, 16):
            s = slope(point_coords(i), point_coords(j))
            slopes[i][j] = s
            slopes[j][i] = s
    
    # Count distinct slopes
    all_slopes = set()
    for i in range(16):
        for j in range(i+1, 16):
            all_slopes.add(slopes[i][j])
    
    # Map slopes to color indices 0, 1, 2, ...
    slope_to_color = {s: idx for idx, s in enumerate(sorted(all_slopes))}
    num_slope_colors = len(slope_to_color)
    
    # Edge index function
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return 15*i - i*(i-1)//2 + (j - i - 1)
    
    # Precompute K4 edges
    k4_edges = []
    k4_vertices = []
    for a in range(16):
        for b in range(a+1, 16):
            for c in range(b+1, 16):
                for d in range(c+1, 16):
                    edges = [
                        edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                        edge_idx(b, c), edge_idx(b, d),
                        edge_idx(c, d)
                    ]
                    k4_edges.append(tuple(edges))
                    k4_vertices.append((a, b, c, d))
    
    num_k4s = len(k4_edges)
    
    # Generate initial coloring based on slope construction
    def get_slope_coloring(perm):
        """Generate coloring using slope-based colors with vertex permutation"""
        colors = [0] * 120
        for i in range(16):
            for j in range(i+1, 16):
                pi, pj = perm[i], perm[j]
                s = slopes[pi][pj]
                colors[edge_idx(i, j)] = slope_to_color[s]
        return colors
    
    # Also try a simpler construction: color by (i+j) mod k for various k
    def get_mod_coloring(k, seed=0):
        """Simple modular coloring"""
        random.seed(seed)
        perm = list(range(16))
        random.shuffle(perm)
        colors = [0] * 120
        for i in range(16):
            for j in range(i+1, 16):
                colors[edge_idx(i, j)] = (perm[i] + perm[j]) % k
        return colors
    
    # Validate a coloring
    def count_colors(colors):
        return len(set(colors))
    
    def is_valid(colors):
        """Check if every K4 has at least 5 distinct colors"""
        for edges in k4_edges:
            distinct = set(colors[e] for e in edges)
            if len(distinct) < 5:
                return False
        return True
    
    # Try slope-based coloring
    best_colors = None
    best_count = 999
    
    # Identity permutation for slope coloring
    identity = list(range(16))
    slope_colors = get_slope_coloring(identity)
    slope_count = count_colors(slope_colors)
    
    if is_valid(slope_colors):
        best_colors = slope_colors
        best_count = slope_count
    
    # Try modular colorings with different moduli
    for k in range(13, 18):
        for seed in range(5):
            if time.time() > deadline:
                break
            mod_colors = get_mod_coloring(k, seed)
            mod_count = count_colors(mod_colors)
            if mod_count < best_count and is_valid(mod_colors):
                best_colors = mod_colors
                best_count = mod_count
    
    # Local search: try to reduce colors by merging
    def rebuild_k4s_using(col_list, max_color):
        new_k4s_using = [set() for _ in range(max_color)]
        for k4_idx, edges in enumerate(k4_edges):
            distinct_colors_in_k4 = set()
            for e in edges:
                c = col_list[e]
                distinct_colors_in_k4.add(c)
            for c in distinct_colors_in_k4:
                if c < max_color:
                    new_k4s_using[c].add(k4_idx)
        return new_k4s_using
    
    def is_valid_subset(k4_indices, col_list):
        for k4_idx in k4_indices:
            edges = k4_edges[k4_idx]
            c0 = col_list[edges[0]]
            c1 = col_list[edges[1]]
            c2 = col_list[edges[2]]
            c3 = col_list[edges[3]]
            c4 = col_list[edges[4]]
            c5 = col_list[edges[5]]
            mask = (1 << c0) | (1 << c1) | (1 << c2) | (1 << c3) | (1 << c4) | (1 << c5)
            if mask.bit_count() < 5:
                return False
        return True
    
    # Try multiple restarts with local search
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Choose initial coloring
        if restart % 2 == 0:
            # Slope-based with random permutation
            random.seed(restart * 7919 + 13)
            perm = list(range(16))
            random.shuffle(perm)
            current_colors = get_slope_coloring(perm)
        else:
            # Modular with random parameters
            k = random.randint(13, 16)
            current_colors = get_mod_coloring(k, restart * 31 + 7)
        
        current_count = count_colors(current_colors)
        if not is_valid(current_colors):
            continue
            
        if current_count < best_count:
            best_count = current_count
            best_colors = current_colors.copy()
        
        max_color = max(current_colors) + 1
        current_active = set(current_colors)
        current_k4s_using = rebuild_k4s_using(current_colors, max_color)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cols_list = list(current_active)
            random.shuffle(cols_list)
            
            for i in range(len(cols_list)):
                for j in range(i+1, len(cols_list)):
                    c1 = cols_list[i]
                    c2 = cols_list[j]
                    if c1 not in current_active or c2 not in current_active:
                        continue
                        
                    affected = current_k4s_using[c1] | current_k4s_using[c2]
                    
                    new_colors = current_colors.copy()
                    for k in range(120):
                        if new_colors[k] == c2:
                            new_colors[k] = c1
                    
                    if is_valid_subset(affected, new_colors):
                        current_colors = new_colors
                        current_active.remove(c2)
                        current_k4s_using[c1] = current_k4s_using[c1] | current_k4s_using[c2]
                        current_k4s_using[c2] = set()
                        
                        improved = True
                        if len(current_active) < best_count:
                            best_count = len(current_active)
                            best_colors = current_colors.copy()
                        break
                if improved:
                    break
    
    return best_colors
