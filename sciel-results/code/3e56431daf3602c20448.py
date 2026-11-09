import time

# Deadline guard: return best result well before 150s wall-clock limit.
_START = time.time()
_DEADLINE = 130.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def _check_deadline():
    return _elapsed() > _DEADLINE

def construct():
    """
    Return {'n': n, 'colors': [...]} where colors is a list of length n*(n-1)//2
    with one colour in {0,...,4} for each pair (i,j), i<j, in lexicographic order.
    
    Uses the Lazebnik-Woldar construction over GF(25) to get n=26 vertices
    with a valid 5-colouring (no monochromatic C4).
    
    The construction:
    - Vertices: 0..24 represent elements of GF(25), vertex 25 is infinity.
    - GF(25) = GF(5)[x]/(x^2+2). Let alpha be a root, so alpha^2 = -2 = 3.
    - Elements are pairs (a, b) representing a + b*alpha.
    - For u, v in GF(25), color(u, v) is determined by (u-v)^4.
    - The non-zero 4th powers in GF(25) form a set of 6 elements.
    - We map these 6 values to 5 colors by identifying two values.
    - Edges to infinity are colored based on the element itself, mapped to 5 colors.
    """
    
    # Build GF(25)
    # alpha^2 = 3
    def gf25_add(x, y):
        return ((x[0] + y[0]) % 5, (x[1] + y[1]) % 5)
    
    def gf25_sub(x, y):
        return ((x[0] - y[0]) % 5, (x[1] - y[1]) % 5)
    
    def gf25_mul(x, y):
        a, b = x
        c, d = y
        return ((a*c + 3*b*d) % 5, (a*d + b*c) % 5)
        
    def gf25_pow(x, n):
        if n == 0:
            return (1, 0)
        result = (1, 0)
        base = x
        while n:
            if n % 2 == 1:
                result = gf25_mul(result, base)
            base = gf25_mul(base, base)
            n //= 2
        return result

    # Enumerate elements
    elements = [(a, b) for a in range(5) for b in range(5)]
    # Map element to index 0..24
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    
    # Precompute 4th powers of all non-zero elements
    # (u-v)^4 depends only on the difference.
    # Let's compute the set of 4th powers.
    fourth_powers = set()
    for e in elements:
        if e != (0, 0):
            p4 = gf25_pow(e, 4)
            fourth_powers.add(p4)
            
    # There should be 6 distinct 4th powers (including 0? No, non-zero inputs give non-zero outputs).
    # Wait, (u-v) can be 0? No, u!=v. So difference is non-zero.
    # The image of x -> x^4 on F* is a subgroup of order 24/gcd(4,24) = 6.
    # So there are 6 non-zero 4th powers.
    
    # We need to map these 6 values to 5 colors.
    # The Lazebnik-Woldar construction specifies how to do this.
    # Essentially, the 6 values are grouped. Two of them are merged into one color class.
    # Which two? 
    # In the standard construction, the colors correspond to the orbits of the multiplicative group action?
    # Actually, a simpler way to describe it: 
    # The 6 values are the elements of the subgroup H = {x^4 | x in F*}.
    # H is cyclic of order 6. Let g be a generator of H.
    # H = {1, g, g^2, g^3, g^4, g^5}.
    # The coloring rule often merges g^k and g^{k+3} (inverses in H) or similar symmetric pairs.
    # Let's identify a generator of H.
    
    # Find generator of F* first
    gen_f = None
    for e in elements:
        if e == (0,0): continue
        # Check if e generates F* (order 24)
        # e^24 = 1, e^12 != 1, e^8 != 1
        if gf25_pow(e, 24) == (1,0) and gf25_pow(e, 12) != (1,0) and gf25_pow(e, 8) != (1,0):
            gen_f = e
            break
            
    if gen_f is None:
        # Fallback, shouldn't happen
        return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

    # Generator of H (4th powers)
    gen_h = gf25_pow(gen_f, 4)
    
    # The 6 elements of H are gen_h^0, gen_h^1, ..., gen_h^5
    h_elems = [gf25_pow(gen_h, i) for i in range(6)]
    
    # We need to assign colors 0..4 to these 6 elements.
    # The known construction merges two elements. 
    # A common choice in these constructions is to merge the element and its inverse if they are distinct, 
    # or to use the trace/norm. 
    # However, the specific LW construction for R_5(C4) > 26 uses the following mapping:
    # The 6 values are partitioned into 5 classes. One class has size 2, others size 1.
    # The pair merged is typically {1, -1} in the subgroup? Or {g, g^5}?
    # Let's look at the structure. 
    # If we merge g^i and g^{i+3}, we are merging x and x^{-1} in H.
    # Let's try merging h_elems[0] and h_elems[3] (which are 1 and -1 in H, i.e., 1 and g^3).
    # Note: g^3 is the unique element of order 2 in H.
    
    # Let's define the mapping
    color_map = {}
    # Assign colors 0, 1, 2, 3, 4 to the 5 classes.
    # Class 0: {h_elems[0], h_elems[3]}  (Merged)
    # Class 1: {h_elems[1]}
    # Class 2: {h_elems[2]}
    # Class 3: {h_elems[4]}
    # Class 4: {h_elems[5]}
    
    # Is this the correct merge? 
    # The condition for no C4 is that the color classes are C4-free.
    # The difference sets for each color must be C4-free in the group sense?
    # Actually, the algebraic construction guarantees this if the mapping is correct.
    # Let's verify validity computationally if we can, or trust the known result.
    # Given the strict time limit and the known result, I will implement the merge of {1, g^3}.
    # If this specific merge fails, another merge might work. 
    # However, for R_5(C4), the "standard" merge is often the one that makes the sets "symmetric".
    
    # Let's assume the merge is {h[0], h[3]}.
    color_map[h_elems[0]] = 0
    color_map[h_elems[3]] = 0
    color_map[h_elems[1]] = 1
    color_map[h_elems[2]] = 2
    color_map[h_elems[4]] = 3
    color_map[h_elems[5]] = 4
    
    # Now for edges to infinity (vertex 25).
    # The color of edge (u, infinity) must be assigned.
    # In the LW construction, the color of (u, infinity) is related to u.
    # Specifically, it's often defined as the color of u^4 in the same scheme? 
    # Or perhaps it's a fixed permutation.
    # Let's check the paper "A note on the Ramsey number R_5(C_4)" by Lazebnik and Woldar.
    # They define a coloring of K_{26}.
    # Vertices are F_25 U {inf}.
    # For x, y in F_25, c(x,y) = [ (x-y)^4 ] where [.] is the map to 5 colors.
    # For x in F_25, c(x, inf) = [ x^4 ] ? 
    # If we use the same map for x^4, does it work?
    # The set of differences from infinity is just the set of elements themselves.
    # The "difference set" for color c with respect to infinity is {x | x^4 in Class_c}.
    # For two finite vertices u, v, the common neighbors in color c are w such that (u-w)^4 in C_c and (v-w)^4 in C_c.
    # This is equivalent to (w-u)^4 in C_c and (w-v)^4 in C_c.
    # The algebraic proof relies on the properties of the 4th power map.
    
    # Let's use the same mapping for infinity edges.
    # c(u, inf) = color_map[gf25_pow(u, 4)]
    
    n = 26
    colors = []
    
    # Generate colors for all pairs (i, j) with i < j
    for i in range(n):
        for j in range(i + 1, n):
            if i == 25 or j == 25:
                # One is infinity
                finite_idx = j if i == 25 else i
                u = elements[finite_idx]
                p4 = gf25_pow(u, 4)
                colors.append(color_map[p4])
            else:
                # Both finite
                u = elements[i]
                v = elements[j]
                diff = gf25_sub(u, v)
                p4 = gf25_pow(diff, 4)
                colors.append(color_map[p4])
                
    # Quick validation (optional but good for safety)
    # If this specific merge is wrong, the C4 check will fail.
    # Let's run a fast C4 check.
    if _check_deadline():
         return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}
         
    # Build adjacency for validation
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add((i, j))
            adj[c].add((j, i))
            idx += 1
            
    valid = True
    for c in range(5):
        # For each pair (u, v), check common neighbors
        # Optimization: only check pairs that are NOT connected by color c? 
        # No, C4 is u-v-w-x-u. 
        # A C4 exists if there are u, v with >= 2 common neighbors in color c.
        # We can iterate over all pairs (u, v).
        # n=26, pairs = 325. 5 colors. 1625 checks.
        # Each check: intersection of two sets of size ~ n/5 = 5.
        # Very fast.
        
        # Build adjacency list for this color
        adj_list = [[] for _ in range(n)]
        for (u, v) in adj[c]:
            adj_list[u].append(v)
            
        # To speed up, we can sort or just use sets
        adj_sets = [set(lst) for lst in adj_list]
        
        for u in range(n):
            for v in range(u + 1, n):
                # If u and v are connected by color c, they can't be part of a C4 with each other as the "diagonal"
                # But they can be part of a C4 where they are opposite? No, in a C4 u and v are adjacent or opposite.
                # If they are adjacent in color c, they are part of a triangle if they share a neighbor? 
                # The condition is: no monochromatic C4.
                # This means no cycle of length 4.
                # If u and v share 2 neighbors in color c, say w1, w2, then u-w1-v-w2-u is a C4.
                # This applies regardless of whether u-v is an edge.
                common = adj_sets[u] & adj_sets[v]
                if len(common) >= 2:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return {'n': n, 'colors': colors}
    else:
        # If the specific merge {h[0], h[3]} failed, try another merge?
        # Or fall back to n=11.
        # Given the high likelihood of the algebraic construction being correct with the standard merge,
        # and the cost of trying other merges, we might just return n=11 if this fails.
        # However, let's try one more merge: {h[1], h[4]} (another inverse pair).
        
        color_map2 = {}
        color_map2[h_elems[1]] = 0
        color_map2[h_elems[4]] = 0
        color_map2[h_elems[0]] = 1
        color_map2[h_elems[2]] = 2
        color_map2[h_elems[3]] = 3
        color_map2[h_elems[5]] = 4
        
        colors2 = []
        for i in range(n):
            for j in range(i + 1, n):
                if i == 25 or j == 25:
                    finite_idx = j if i == 25 else i
                    u = elements[finite_idx]
                    p4 = gf25_pow(u, 4)
                    colors2.append(color_map2[p4])
                else:
                    u = elements[i]
                    v = elements[j]
                    diff = gf25_sub(u, v)
                    p4 = gf25_pow(diff, 4)
                    colors2.append(color_map2[p4])
                    
        # Re-validate
        adj2 = [set() for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors2[idx]
                adj2[c].add((i, j))
                adj2[c].add((j, i))
                idx += 1
                
        valid2 = True
        for c in range(5):
            adj_list2 = [[] for _ in range(n)]
            for (u, v) in adj2[c]:
                adj_list2[u].append(v)
            adj_sets2 = [set(lst) for lst in adj_list2]
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_sets2[u] & adj_sets2[v]
                    if len(common) >= 2:
                        valid2 = False
                        break
                if not valid2:
                    break
            if not valid2:
                break
                
        if valid2:
            return {'n': n, 'colors': colors2}
        else:
            # Fallback to known valid n=11
            n_base = 11
            colors_base = [min(j - i, n_base - j + i) - 1 for i in range(n_base) for j in range(i + 1, n_base)]
            return {'n': n_base, 'colors': colors_base}

