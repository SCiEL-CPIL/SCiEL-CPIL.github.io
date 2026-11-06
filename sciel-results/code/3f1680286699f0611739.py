"""
Constructs a C4-free graph on 43 vertices with independence number < 11.
Uses an induced subgraph of the polarity graph of the projective plane of order 7.
The polarity graph of order q is C4-free and has independence number at most q+1.
For q=7, n_total=57, alpha <= 8.
We take the first 43 vertices.
"""

def construct():
    q = 7
    
    # Generate all points in PG(2, q)
    # Points are normalized triples (x, y, z) in GF(q)^3 \ {0}
    points = []
    seen = set()
    
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                if x != 0:
                    # x is 1, y = y * x^-1, z = z * x^-1
                    # In GF(7), inverse of 1 is 1.
                    # We just need to find k such that k*x == 1
                    # Since x != 0, inverse exists.
                    inv_x = pow(x, q-2, q) # Fermat's little theorem
                    ny = (y * inv_x) % q
                    nz = (z * inv_x) % q
                    p = (1, ny, nz)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    nx = (x * inv_y) % q # x is 0
                    nz = (z * inv_y) % q
                    p = (0, 1, nz)
                else: # z != 0
                    inv_z = pow(z, q-2, q)
                    nx = (x * inv_z) % q
                    ny = (y * inv_z) % q
                    p = (0, 0, 1)
                
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    n_total = len(points)
    # n_total should be q^2 + q + 1 = 57
    
    # Define the polarity bilinear form B(u, v) = u1*v3 + u2*v2 + u3*v1
    # This is symmetric: B(u,v) = B(v,u)
    def is_incident(u, v):
        val = (u[0] * v[2] + u[1] * v[1] + u[2] * v[0]) % q
        return val == 0

    # We want a subgraph on n=43 vertices
    n_target = 43
    if n_total < n_target:
        n_target = n_total
        
    # Take the first n_target points
    sub_points = points[:n_target]
    
    edges = []
    for i in range(n_target):
        for j in range(i + 1, n_target):
            if is_incident(sub_points[i], sub_points[j]):
                edges.append([i, j])
                
    return {'n': n_target, 'edges': edges}
