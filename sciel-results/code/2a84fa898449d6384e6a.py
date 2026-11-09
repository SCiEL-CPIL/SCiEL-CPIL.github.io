import itertools

def construct():
    """
    Construct a valid 5-coloring of K_26 without monochromatic C4s
    using a cyclic difference set construction (Lazebnik-Woldar style).
    """
    n = 26
    # Differences 1 to 13
    diffs = list(range(1, n // 2 + 1))
    
    # We partition diffs into 5 sets, each C4-free.
    # A set D is C4-free in Z_n if no a,b,c,d in D satisfy a+b=c+d (mod n)
    # with distinct vertices 0, a, a+b, a+b-c.
    
    color_diffs = [[] for _ in range(5)]
    
    def is_c4_free_fast(diffs_list):
        # Check if any 4-tuple forms a C4
        sums = {}
        for i, a in enumerate(diffs_list):
            for b in diffs_list[i:]: 
                s = (a + b) % n
                if s in sums:
                    c, d = sums[s]
                    # Check distinctness of 0, a, a+b, a+b-c
                    v0 = 0
                    v1 = a % n
                    v2 = (a + b) % n
                    v3 = (a + b - c) % n
                    if len({v0, v1, v2, v3}) == 4:
                        return False
                else:
                    sums[s] = (a, b)
        return True

    def backtrack(idx):
        if idx == len(diffs):
            return True
        
        diff = diffs[idx]
        # Symmetry breaking: first diff must go to color 0
        if idx == 0:
            candidates = [0]
        else:
            candidates = range(5)
            
        for c in candidates:
            color_diffs[c].append(diff)
            
            if is_c4_free_fast(color_diffs[c]):
                if backtrack(idx + 1):
                    return True
            
            color_diffs[c].pop()
        
        return False

    if backtrack(0):
        # Construct the color array
        colors_out = []
        # Map difference to color
        diff_to_color = {}
        for c in range(5):
            for d in color_diffs[c]:
                diff_to_color[d] = c
                
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                d = min(d, n - d)
                # d is in 1..13
                colors_out.append(diff_to_color[d])
        
        # Final validation
        if validate(n, colors_out):
            return {'n': n, 'colors': colors_out}
    
    # Fallback to n=11 cyclic if 26 fails
    n = 11
    colors_out = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            d = min(d, n - d)
            colors_out.append(d - 1)
    return {'n': n, 'colors': colors_out}

def validate(n, colors):
    """Check no monochromatic C4."""
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add((i, j))
            idx += 1
            
    nbrs = [dict() for _ in range(5)]
    for c in range(5):
        for (u, v) in adj[c]:
            if u not in nbrs[c]: nbrs[c][u] = set()
            if v not in nbrs[c]: nbrs[c][v] = set()
            nbrs[c][u].add(v)
            nbrs[c][v].add(u)
            
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                if u in nbrs[c] and v in nbrs[c]:
                    if len(nbrs[c][u] & nbrs[c][v]) >= 2:
                        return False
    return True
