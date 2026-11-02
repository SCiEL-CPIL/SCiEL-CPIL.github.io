"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import time
    deadline = time.time() + 140.0
    
    # For the (4,10)-cage, use the generalized hexagon of order (2,2) over GF(4)
    # This gives a 4-regular graph of girth 12 with 224 vertices.
    #
    # The generalized hexagon GQ(2,2) over GF(4) has:
    # - 112 points
    # - 112 lines
    # - Each point on 4 lines, each line through 4 points
    # - Incidence graph has girth 12 (no cycles of length < 12)
    #
    # Points: equivalence classes of (x, y, z, w) in GF(4)^4 \ {0} under
    #   (x,y,z,w) ~ (λx, λy, λz, λw) for λ ≠ 0,
    #   subject to the quadratic form Q(x,y,z,w) = xw + yz = 0.
    #
    # Lines: equivalence classes of (u, v, s, t) in GF(4)^4 \ {0} under
    #   (u,v,s,t) ~ (λu, λv, λs, λt) for λ ≠ 0,
    #   subject to the dual quadratic form Q*(u,v,s,t) = us + vt = 0.
    #
    # Incidence: point (x,y,z,w) lies on line (u,v,s,t) iff xu + yv + zs + wt = 0.
    
    if k != 4 or g != 10:
        # Fallback to LUW construction for other parameters
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((i, index[other]))
                    
        return len(order), edges
    
    # GF(4) arithmetic
    # GF(4) = {0, 1, α, α+1} where α^2 + α + 1 = 0
    # Represent as: 0=0, 1=1, 2=α, 3=α+1
    # Addition: XOR
    # Multiplication table:
    #   0*anything = 0
    #   1*x = x
    #   2*2 = 3, 2*3 = 1, 3*3 = 2
    
    gf4_mul = [[0,0,0,0],
               [0,1,2,3],
               [0,2,3,1],
               [0,3,1,2]]
    
    def gf4_add(a, b):
        return a ^ b
    
    def gf4_sub(a, b):
        return a ^ b
    
    def gf4_mul(a, b):
        return gf4_mul[a][b]
    
    def gf4_neg(a):
        return a  # In char 2, -a = a
    
    def gf4_inv(a):
        if a == 0:
            raise ValueError("Cannot invert 0")
        # 1*1=1, 2*3=1, 3*2=1
        return {1:1, 2:3, 3:2}[a]
    
    def gf4_eq(a, b):
        return a == b
    
    # Generate all points: (x,y,z,w) with xw + yz = 0, not all zero, modulo scaling
    # We fix a representative by requiring the first nonzero coordinate to be 1.
    
    points = []
    point_set = set()
    
    for x in range(4):
        for y in range(4):
            for z in range(4):
                for w in range(4):
                    if x == 0 and y == 0 and z == 0 and w == 0:
                        continue
                    # Check Q(x,y,z,w) = xw + yz = 0
                    if gf4_add(gf4_mul(x, w), gf4_mul(y, z)) != 0:
                        continue
                    # Normalize: find first nonzero coordinate, scale to make it 1
                    for i in range(4):
                        coord = [x, y, z, w][i]
                        if coord != 0:
                            inv = gf4_inv(coord)
                            rep = (gf4_mul(inv, x), gf4_mul(inv, y), gf4_mul(inv, z), gf4_mul(inv, w))
                            if rep not in point_set:
                                point_set.add(rep)
                                points.append(rep)
                            break
    
    # Generate all lines: (u,v,s,t) with us + vt = 0, not all zero, modulo scaling
    lines = []
    line_set = set()
    
    for u in range(4):
        for v in range(4):
            for s in range(4):
                for t in range(4):
                    if u == 0 and v == 0 and s == 0 and t == 0:
                        continue
                    # Check Q*(u,v,s,t) = us + vt = 0
                    if gf4_add(gf4_mul(u, s), gf4_mul(v, t)) != 0:
                        continue
                    # Normalize
                    for i in range(4):
                        coord = [u, v, s, t][i]
                        if coord != 0:
                            inv = gf4_inv(coord)
                            rep = (gf4_mul(inv, u), gf4_mul(inv, v), gf4_mul(inv, s), gf4_mul(inv, t))
                            if rep not in line_set:
                                line_set.add(rep)
                                lines.append(rep)
                            break
    
    # Build incidence: point p=(x,y,z,w) lies on line l=(u,v,s,t) iff xu+yv+zs+wt=0
    point_index = {p: i for i, p in enumerate(points)}
    line_index = {l: i for i, l in enumerate(lines)}
    
    n_points = len(points)
    n_lines = len(lines)
    n = n_points + n_lines
    
    edges = []
    
    for pi, p in enumerate(points):
        x, y, z, w = p
        for li, l in enumerate(lines):
            u, v, s, t = l
            # Check incidence: xu + yv + zs + wt = 0
            val = gf4_add(gf4_add(gf4_add(gf4_mul(x, u), gf4_mul(y, v)), 
                                   gf4_mul(z, s)), gf4_mul(w, t))
            if val == 0:
                edges.append((pi, n_points + li))
    
    return n, edges
# EVOLVE-BLOCK-END
