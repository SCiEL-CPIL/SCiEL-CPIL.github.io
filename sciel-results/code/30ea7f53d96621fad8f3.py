import numpy as np
import time

def construct():
    """
    Construct a 4-coloring of K_50 with no monochromatic triangle.
    Uses GF(49) structure for the first 49 vertices.
    Attempts to add a 50th vertex (infinity) by solving a CSP for its edge colors.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    # GF(49) = GF(7)[x]/(x^2 + 3) since 3 is not a square mod 7
    # Elements are (a, b) representing a + b*alpha where alpha^2 = -3 = 4 mod 7
    
    def gf_add(a, b):
        return ((a[0] + b[0]) % 7, (a[1] + b[1]) % 7)
    
    def gf_neg(a):
        return ((-a[0]) % 7, (-a[1]) % 7)
        
    def gf_sub(a, b):
        return gf_add(a, gf_neg(b))

    def gf_mul(a, b):
        # (a0 + a1*alpha)*(b0 + b1*alpha) = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*alpha^2
        # alpha^2 = 4 mod 7
        a0, a1 = a
        b0, b1 = b
        return ((a0*b0 + a1*b1*4) % 7, (a0*b1 + a1*b0) % 7)
    
    def gf_pow(a, n):
        if n == 0:
            return (1, 0)
        result = (1, 0)
        base = a
        while n > 0:
            if n & 1:
                result = gf_mul(result, base)
            base = gf_mul(base, base)
            n >>= 1
        return result
    
    # Build all elements of GF(49)
    elements = []
    for a in range(7):
        for b in range(7):
            elements.append((a, b))
            
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    
    # Find a primitive element (generator of multiplicative group of order 48)
    nonzero = [e for e in elements if e != (0, 0)]
    primitive = None
    for g in nonzero:
        # Check if g is primitive: order 48
        # g^48 == 1 and g^(48/p) != 1 for p in {2, 3}
        if gf_pow(g, 48) == (1, 0) and gf_pow(g, 24) != (1, 0) and gf_pow(g, 16) != (1, 0):
            primitive = g
            break
            
    if primitive is None:
        # Fallback if no primitive found (should not happen)
        return {'n': 1, 'colors': []}
        
    g = primitive
    
    # Generate powers of g
    powers = []
    x = (1, 0)
    for i in range(48):
        x = gf_mul(x, g)
        powers.append(x)
        
    # Map each nonzero element to its discrete log base g
    log_table = {}
    for i, p in enumerate(powers):
        log_table[p] = i
        
    # Define color for edge between a and b in GF(49)
    def get_color_49(a, b):
        diff = gf_sub(a, b)
        if diff == (0, 0):
            return -1 # No edge
        idx = log_table[diff]
        return idx % 4

    # Base coloring for 49 vertices
    n_base = 49
    colors_base = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            a = elements[i]
            b = elements[j]
            colors_base.append(get_color_49(a, b))
            
    # Try to add the 50th vertex (index 49)
    # We need to assign colors c_a for edge (49, a) for a in 0..48
    # Constraint: For any a, b < 49, if c_a == c_b == k, then color(a,b) != k
    
    # Precompute for each color k, the set of pairs (a,b) such that color(a,b) == k
    # Actually, we need to ensure that for each k, the set S_k = {a | c_a == k} is "k-independent"
    # i.e., no two elements in S_k have difference in H_k.
    # H_k = {x | log(x) % 4 == k}
    
    # Let's use a backtracking search with heuristics for the 50th vertex colors
    # There are 4^49 possibilities, but constraints are tight.
    # We can model this as a graph coloring problem on the 49 vertices?
    # No, it's simpler:
    # Let c_a be the color of edge (inf, a).
    # Conflict if c_a == c_b == color(a,b).
    
    # We can try random assignments and repair.
    # Or use a simple backtracking.
    # Given the symmetry, maybe a structured assignment works?
    # For now, let's try a few random restarts of local search.
    
    best_n = 49
    best_colors = colors_base
    
    # If time allows, try to find a valid 50th vertex
    if time.time() < deadline - 5:
        # Represent constraints
        # For each pair (a,b), we know color(a,b).
        # If we assign c_a, c_b, we must not have c_a == c_b == color(a,b).
        
        # We can view this as finding a 4-coloring of a hypergraph?
        # Let's just use a simple randomized local search on the 49 variables c_0..c_48.
        
        def count_conflicts(c):
            conf = 0
            for a in range(49):
                for b in range(a+1, 49):
                    if c[a] == c[b] == colors_base[a*49 - a*(a+1)//2 + b]: # Index calc is messy, let's precompute
                        conf += 1
            return conf

        # Precompute color matrix for 49x49 for fast access
        color_mat = [[0]*49 for _ in range(49)]
        idx = 0
        for i in range(49):
            for j in range(i+1, 49):
                color_mat[i][j] = colors_base[idx]
                color_mat[j][i] = colors_base[idx]
                idx += 1
        
        def count_conflicts_fast(c):
            conf = 0
            for a in range(49):
                ca = c[a]
                for b in range(a+1, 49):
                    if c[b] == ca and color_mat[a][b] == ca:
                        conf += 1
            return conf

        import random
        random.seed(42)
        
        found_50 = False
        # Try 1000 random restarts
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
                
            c = [random.randint(0, 3) for _ in range(49)]
            # Local search: pick a vertex, change its color to minimize conflicts
            for step in range(2000):
                if time.time() > deadline - 1:
                    break
                    
                conf = count_conflicts_fast(c)
                if conf == 0:
                    found_50 = True
                    break
                    
                # Pick a random vertex to flip
                v = random.randint(0, 48)
                old_c = c[v]
                
                # Try all 4 colors for v, pick best
                best_new_c = old_c
                best_conf = conf
                for new_c in range(4):
                    if new_c == old_c:
                        continue
                    c[v] = new_c
                    new_conf = count_conflicts_fast(c)
                    if new_conf < best_conf:
                        best_conf = new_conf
                        best_new_c = new_c
                
                c[v] = best_new_c
                
            if found_50:
                break
        
        if found_50:
            # Construct full list of colors for n=50
            # Order: lexicographic (i,j)
            # Edges among 0..48 are in colors_base
            # Edges involving 49 are c[0]..c[48]
            # Lexicographic order:
            # (0,1)..(0,48), (1,2)..(1,48), ..., (48,49)
            # Wait, the list `colors` must be in lexicographic order of (i,j) for i<j in 0..n-1.
            
            full_colors = []
            # Edges not involving 49
            # We already have colors_base in the correct order for 0..48
            # But we need to interleave the edges to 49?
            # No, lexicographic order for n=50:
            # (0,1), (0,2), ..., (0,49)
            # (1,2), ..., (1,49)
            # ...
            # (48,49)
            
            # So we can't just append c[0..48] at the end.
            # We need to build it carefully.
            
            # Let's rebuild the color list for n=50
            # We have color_mat for 0..48.
            # c[a] is color of (49, a).
            
            for i in range(49):
                for j in range(i+1, 50):
                    if j < 49:
                        full_colors.append(color_mat[i][j])
                    else: # j == 49
                        full_colors.append(c[i])
                        
            best_n = 50
            best_colors = full_colors

    return {'n': best_n, 'colors': best_colors}
