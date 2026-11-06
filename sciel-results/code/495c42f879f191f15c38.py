"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct the polarity graph of the projective plane PG(2, 9).
    This graph is C4-free and has independence number alpha <= 10.
    Number of vertices n = 9^2 + 9 + 1 = 91.
    """
    # Define GF(9)
    # Elements are (a, b) representing a + b*i, where i^2 = -1 = 2 in GF(3)
    # GF(3) = {0, 1, 2}
    
    def gf9_add(x, y):
        return ((x[0] + y[0]) % 3, (x[1] + y[1]) % 3)

    def gf9_sub(x, y):
        return ((x[0] - y[0]) % 3, (x[1] - y[1]) % 3)

    def gf9_mul(x, y):
        # (a + bi)(c + di) = ac + adi + bci + bdi^2 = (ac - bd) + (ad + bc)i
        a, b = x
        c, d = y
        real = (a * c - b * d) % 3
        imag = (a * d + b * c) % 3
        return (real, imag)

    def gf9_neg(x):
        return ((-x[0]) % 3, (-x[1]) % 3)

    def gf9_inv(x):
        if x == (0, 0):
            raise ZeroDivisionError
        # (a+bi)^-1 = (a-bi)/(a^2+b^2)
        a, b = x
        denom = (a * a + b * b) % 3
        # denom is in GF(3), so 1 or 2
        inv_denom = 1 if denom == 1 else 2
        real = (a * inv_denom) % 3
        imag = ((-b) * inv_denom) % 3
        return (real, imag)

    # Generate all points in PG(2, 9)
    # Points are non-zero vectors in GF(9)^3 modulo scalar multiplication
    points = []
    seen = set()
    
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    for e in range(3):
                        for f in range(3):
                            v = ((a, b), (c, d), (e, f))
                            if v == ((0,0), (0,0), (0,0)):
                                continue
                            
                            # Canonicalize: find first non-zero component, scale to make it 1
                            # First non-zero component
                            idx = -1
                            for i in range(3):
                                if v[i] != (0, 0):
                                    idx = i
                                    break
                            
                            if idx == -1:
                                continue
                                
                            inv_first = gf9_inv(v[idx])
                            canon = []
                            for i in range(3):
                                canon.append(gf9_mul(v[i], inv_first))
                            
                            canon_tuple = tuple(canon)
                            if canon_tuple not in seen:
                                seen.add(canon_tuple)
                                points.append(canon_tuple)
    
    n = len(points)
    # n should be 91
    
    # Build edges based on polarity B(u, v) = u0v0 + u1v1 + u2v2
    # u ~ v if B(u, v) == 0 and u != v
    edges = []
    
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            # Calculate B(u, v)
            b_val = gf9_add(
                gf9_mul(u[0], v[0]),
                gf9_add(
                    gf9_mul(u[1], v[1]),
                    gf9_mul(u[2], v[2])
                )
            )
            if b_val == (0, 0):
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
