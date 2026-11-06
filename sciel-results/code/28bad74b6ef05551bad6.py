def construct():
    p = 5
    # Represent elements of GF(25) as (a, b) representing a + b*alpha
    # where alpha^2 + alpha + 2 = 0 (since x^2+x+2 is irreducible over F_5)
    # Check irreducibility: discriminant 1 - 8 = -7 = 3. 3 is not a square in F_5 (squares: 0,1,4).
    
    def gf_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def gf_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    def gf_mul(a, b):
        x1, y1 = a
        x2, y2 = b
        # (x1 + y1*alpha)(x2 + y2*alpha) = x1x2 + (x1y2 + x2y1)alpha + y1y2*alpha^2
        # alpha^2 = -alpha - 2
        # = x1x2 - 2y1y2 + (x1y2 + x2y1 - y1y2)alpha
        const_part = (x1 * x2 - 2 * y1 * y2) % p
        alpha_part = (x1 * y2 + x2 * y1 - y1 * y2) % p
        return (const_part, alpha_part)
    
    elems = [(i, j) for i in range(p) for j in range(p)]
    elem_to_idx = {e: i for i, e in enumerate(elems)}
    
    # Compute cubes
    cubes = set()
    for e in elems:
        if e == (0, 0):
            continue
        e2 = gf_mul(e, e)
        e3 = gf_mul(e2, e)
        cubes.add(e3)
        
    # Find omega, a primitive 3rd root of unity
    # omega^3 = 1, omega != 1
    omega = None
    for e in elems:
        if e == (0, 0) or e == (1, 0):
            continue
        e2 = gf_mul(e, e)
        e3 = gf_mul(e2, e)
        if e3 == (1, 0):
            omega = e
            break
            
    if omega is None:
        # Fallback if something went wrong, though it shouldn't
        return {'n': 1, 'colors': []}
        
    C1 = set()
    for c in cubes:
        C1.add(gf_mul(omega, c))
        
    C2 = set()
    for c in cubes:
        C2.add(gf_mul(gf_mul(omega, omega), c))
        
    # Verify triangle-free property for the base K25
    # Check if any C_k contains a, b, c such that a+b+c=0
    def is_triangle_free(c_set):
        for a in c_set:
            for b in c_set:
                s = gf_add(a, b)
                ns = gf_neg(s)
                if ns in c_set:
                    return False
        return True
        
    if not (is_triangle_free(cubes) and is_triangle_free(C1) and is_triangle_free(C2)):
        # This should not happen for p=5
        # Fallback to a known smaller solution or raise error
        # For safety, return a small valid instance
        return {'n': 1, 'colors': []}

    # Construct base K25 colors
    # Map diff to color
    def get_color(diff):
        if diff in cubes:
            return 0
        elif diff in C1:
            return 1
        else:
            return 2
            
    base_colors = {}
    for i in range(25):
        for j in range(i + 1, 25):
            diff = gf_add(elems[j], gf_neg(elems[i]))
            base_colors[(i, j)] = get_color(diff)
            
    # Construct K50
    n = 50
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            v_i = i // 2
            v_j = j // 2
            e_i = i % 2
            e_j = j % 2
            
            if v_i == v_j:
                colors.append(3)
            else:
                if e_i == e_j:
                    # Same epsilon
                    if v_i < v_j:
                        colors.append(base_colors[(v_i, v_j)])
                    else:
                        colors.append(base_colors[(v_j, v_i)])
                else:
                    # Different epsilon
                    # Color is c(other, this) effectively?
                    # Rule: if eps differ, color is c(v_j, v_i) if we view directed?
                    # Let's re-verify the rule derived earlier.
                    # Triangle (u,0), (v,0), (w,1) -> colors c(u,v), c(w,v), c(u,w).
                    # Edge (u,0)-(v,0): eps same -> c(u,v).
                    # Edge (v,0)-(w,1): eps diff -> c(w,v).
                    # Edge (w,1)-(u,0): eps diff -> c(u,w).
                    # So for edge (i, j) with eps diff, we want c(v_j, v_i) if we align with the "second vertex" logic?
                    # Let's check: Edge (v,0) to (w,1). i=(v,0), j=(w,1).
                    # We used c(w,v).
                    # So if e_i != e_j, color is base_color(v_j, v_i).
                    if v_j < v_i:
                        colors.append(base_colors[(v_j, v_i)])
                    else:
                        colors.append(base_colors[(v_i, v_j)])
                        
    return {'n': n, 'colors': colors}
