def construct():
    p = 5
    # GF(5^2) with polynomial x^2 + x + 2 (irreducible over F_5)
    # Element (a, b) represents a + b*alpha where alpha^2 = -alpha - 2
    
    def gf_mul(a, b):
        x1, y1 = a
        x2, y2 = b
        x1x2 = (x1 * x2) % p
        y1y2 = (y1 * y2) % p
        x1y2 = (x1 * y2) % p
        x2y1 = (x2 * y1) % p
        # alpha^2 = -alpha - 2
        const_part = (x1x2 - 2 * y1y2) % p
        alpha_part = (x1y2 + x2y1 - y1y2) % p
        return (const_part, alpha_part)
    
    def gf_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def gf_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    # Generate all 25 elements
    elems = [(i, j) for i in range(p) for j in range(p)]
    
    # Compute cubic residues (cubes of all nonzero elements)
    cubes = set()
    for e in elems:
        if e == (0, 0):
            continue
        e2 = gf_mul(e, e)
        e3 = gf_mul(e2, e)
        cubes.add(e3)
    
    # Find primitive 3rd root of unity (element of order 3, not 1)
    omega = None
    for e in elems:
        if e == (0, 0) or e == (1, 0):
            continue
        e2 = gf_mul(e, e)
        e3 = gf_mul(e2, e)
        if e3 == (1, 0):
            omega = e
            break
    
    # Create the three cosets
    C0 = cubes
    C1 = set()
    for c in cubes:
        C1.add(gf_mul(omega, c))
    C2 = set()
    omega2 = gf_mul(omega, omega)
    for c in cubes:
        C2.add(gf_mul(omega2, c))
    
    # Build 3-coloring of K_25
    colors_25 = []
    for i in range(25):
        for j in range(i + 1, 25):
            diff = gf_add(elems[j], gf_neg(elems[i]))
            if diff in C0:
                col = 0
            elif diff in C1:
                col = 1
            else:
                col = 2
            colors_25.append(col)
    
    # Build 4-coloring of K_50: two copies of K_25, cross-edges get color 3
    n = 50
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if i < 25 and j < 25:
                # First copy
                idx = i * 24 - i * (i + 1) // 2 + (j - i - 1)
                colors.append(colors_25[idx])
            elif i >= 25 and j >= 25:
                # Second copy
                i2 = i - 25
                j2 = j - 25
                idx = i2 * 24 - i2 * (i2 + 1) // 2 + (j2 - i2 - 1)
                colors.append(colors_25[idx])
            else:
                # Cross edge
                colors.append(3)
    
    return {'n': n, 'colors': colors}
