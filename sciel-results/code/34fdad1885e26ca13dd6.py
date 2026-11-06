def construct():
    # Helper to check if a 3-coloring based on Paley cubic residues is valid
    def is_triangle_free_paley(q, poly_coeffs):
        # poly_coeffs is list [a, b, c] for x^2 + a*x + b over F_p
        # We assume q = p^2 for simplicity in this specific check, or just use general GF
        # For q=25, p=5.
        
        p = 5
        # Implement GF(p^2)
        # Reps are (a, b) representing a + b*alpha
        def gf_mul(a, b):
            # a, b are tuples (x, y)
            # (x1 + y1*a) * (x2 + y2*a) = x1x2 + (x1y2 + x2y1)a + y1y2 a^2
            # a^2 = -a - b_const
            x1, y1 = a
            x2, y2 = b
            x1x2 = (x1 * x2) % p
            y1y2 = (y1 * y2) % p
            x1y2 = (x1 * y2) % p
            x2y1 = (x2 * y1) % p
            
            # a^2 = -a - 2  (for x^2 + x + 2)
            # So y1y2 * a^2 = y1y2 * (-a - 2) = -2*y1y2 - y1y2*a
            const_part = (x1x2 - 2 * y1y2) % p
            alpha_part = (x1y2 + x2y1 - y1y2) % p
            
            return (const_part, alpha_part)
        
        def gf_add(a, b):
            return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
        def gf_neg(a):
            return ((-a[0]) % p, (-a[1]) % p)
        
        # Generate field elements
        elems = []
        for i in range(p):
            for j in range(p):
                elems.append((i, j))
        
        # Find cubic residues
        # Generator g?
        # We need an element of order (q-1)/3 = 8? No, order 3 root of unity.
        # Let's just compute powers of a primitive element.
        # Find a primitive element.
        # Try (1, 1)
        g = (1, 1)
        # Check order
        # We need the set of cubes.
        # The set of non-zero elements is cyclic of order 24.
        # Cubic residues are powers $g^{3k}$.
        # So we need a generator g.
        
        # Let's verify g is primitive.
        # Order of g should be 24.
        # g^12 != 1, g^8 != 1, g^6 != 1, g^4 != 1, g^3 != 1, g^2 != 1, g != 1.
        
        # Simpler: just compute all elements, sort by "power" if we find a gen.
        # Or just check the condition:
        # Triangle exists if there are a,b in C0 such that -(a+b) in C0.
        # We can just pick a candidate set C0.
        # In F_25, the map x -> x^3 is not a bijection.
        # The image is the cubic residues.
        
        cubes = set()
        for e in elems:
            if e == (0, 0): continue
            # Compute e^3
            e2 = gf_mul(e, e)
            e3 = gf_mul(e2, e)
            cubes.add(e3)
            
        # Check triangle-free
        for a in cubes:
            for b in cubes:
                s = gf_add(a, b)
                ns = gf_neg(s)
                if ns in cubes:
                    return False
        return True

    # Check q=25
    if is_triangle_free_paley(25, [1, 2]):
        # Construct K50
        # We need the actual coloring for K25
        # Re-do the logic to get the colors
        p = 5
        def gf_mul(a, b):
            x1, y1 = a
            x2, y2 = b
            x1x2 = (x1 * x2) % p
            y1y2 = (y1 * y2) % p
            x1y2 = (x1 * y2) % p
            x2y1 = (x2 * y1) % p
            const_part = (x1x2 - 2 * y1y2) % p
            alpha_part = (x1y2 + x2y1 - y1y2) % p
            return (const_part, alpha_part)
            
        def gf_add(a, b):
            return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
            
        elems = [(i, j) for i in range(p) for j in range(p)]
        cubes = set()
        for e in elems:
            if e == (0, 0): continue
            e2 = gf_mul(e, e)
            e3 = gf_mul(e2, e)
            cubes.add(e3)
            
        # Map element to index
        elem_to_idx = {e: i for i, e in enumerate(elems)}
        
        # Determine color for pair (u, v)
        # diff = v - u
        # If diff in C0 -> 0
        # If diff in C1 -> 1
        # If diff in C2 -> 2
        # C0 = cubes
        # C1 = {w * c | w in F*? No. C1 = omega * C0}
        # Find omega: primitive 3rd root of unity.
        # x^2 + x + 1 = 0 in F_25.
        # Roots are elements of order 3.
        # We can find it by iterating.
        omega = None
        for e in elems:
            if e == (0,0): continue
            e2 = gf_mul(e, e)
            e3 = gf_mul(e2, e)
            if e3 == (1, 0) and e != (1, 0):
                omega = e
                break
                
        C1 = set()
        for c in cubes:
            C1.add(gf_mul(omega, c))
            
        C2 = set()
        for c in cubes:
            C2.add(gf_mul(gf_mul(omega, omega), c))
            
        colors_25 = []
        for i in range(25):
            for j in range(i+1, 25):
                diff = gf_add(elems[j], gf_neg(elems[i]))
                if diff in cubes:
                    col = 0
                elif diff in C1:
                    col = 1
                else:
                    col = 2
                colors_25.append(col)
                
        # Double to K50
        colors = []
        for i in range(50):
            for j in range(i+1, 50):
                if i < 25 and j < 25:
                    colors.append(colors_25[i*24 - i*(i+1)//2 + j - i - 1]) # This index math is risky
                elif i >= 25 and j >= 25:
                    # Second copy
                    i2 = i - 25
                    j2 = j - 25
                    colors.append(colors_25[i2*24 - i2*(i2+1)//2 + j2 - i2 - 1])
                else:
                    colors.append(3)
                    
        return {'n': 50, 'colors': colors}

    # Fallback to current 32
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
