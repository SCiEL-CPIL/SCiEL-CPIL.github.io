def construct():
    from itertools import combinations
    
    def check_valid(edge_colors):
        """Check every K4 in K16 has at least 5 distinct colors."""
        for quad in combinations(range(16), 4):
            a, b, c, d = quad
            cols = {
                edge_colors[a][b], edge_colors[a][c], edge_colors[a][d],
                edge_colors[b][c], edge_colors[b][d], edge_colors[c][d]
            }
            if len(cols) < 5:
                return False
        return True

    best_colors = None
    best_num = 100
    
    # Helper to build edge colors from a mapping of vertices to field elements
    def build_and_check(p, mapping):
        """
        mapping: list of 16 integers, the image of vertices 0..15 in Z_p^*
        Returns (num_colors, edge_colors) or (None, None) if invalid
        """
        edge_colors = [[0]*16 for _ in range(16)]
        used_colors = set()
        for i in range(16):
            for j in range(i+1, 16):
                c = (mapping[i] * mapping[j]) % p
                edge_colors[i][j] = c
                edge_colors[j][i] = c
                used_colors.add(c)
        
        if not check_valid(edge_colors):
            return None, None
        
        return len(used_colors), edge_colors

    # 1. Standard p=17, S={1..16}
    p = 17
    mapping = list(range(1, 17))
    num, ec = build_and_check(p, mapping)
    if num is not None and num < best_num:
        best_num = num
        best_colors = ec

    # 2. Search subsets for p=19
    p = 19
    all_elems = list(range(1, 19)) # 1..18
    # We need to choose 16 elements from 18. Equivalent to choosing 2 to exclude.
    # There are C(18, 2) = 153 pairs.
    for exclude in combinations(range(18), 2):
        excl_set = set(all_elems[exclude[0]], all_elems[exclude[1]]) if False else set([all_elems[exclude[0]], all_elems[exclude[1]]])
        # Actually all_elems is 1..18. indices 0..17.
        # Let's just use the values.
        S = [x for x in all_elems if x not in {all_elems[exclude[0]], all_elems[exclude[1]]}]
        
        # Optimization: Check color count first?
        # Just build and check. 153 cases is small.
        num, ec = build_and_check(p, S)
        if num is not None and num < best_num:
            best_num = num
            best_colors = ec

    # 3. Search subsets for p=23 (limited search or specific structures)
    # C(22, 16) is too big. 
    # Try arithmetic progressions or other structured sets?
    # Or just try p=17, 19 results.
    
    # Fallback if no valid found (should not happen)
    if best_colors is None:
        # Use the seed fallback
        edge_colors = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(i+1, 16):
                c = (i + j) % 17
                edge_colors[i][j] = c
                edge_colors[j][i] = c
        return [edge_colors[i][j] for i in range(16) for j in range(i+1, 16)]
        
    return [best_colors[i][j] for i in range(16) for j in range(i+1, 16)]
