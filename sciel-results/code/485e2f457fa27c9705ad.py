import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Construct Polarity Graph of PG(2,5)
    # This graph has 31 vertices, is C4-free, and has independence number 5.
    # This certifies R(C4, K11) > 31.
    
    q = 5
    points = []
    # Generate all points in PG(2, q)
    # Points are equivalence classes of non-zero vectors in F_q^3
    # We normalize such that the first non-zero component is 1
    
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    # Multiply by inverse of x
                    inv_x = pow(x, q-2, q)
                    px = 1
                    py = (y * inv_x) % q
                    pz = (z * inv_x) % q
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    px = 0
                    py = 1
                    pz = (z * inv_y) % q
                else: # z != 0
                    inv_z = pow(z, q-2, q)
                    px = 0
                    py = 0
                    pz = 1
                points.append((px, py, pz))
    
    n = len(points) # Should be 31
    
    # Build adjacency
    # u ~ v if u . v == 0 mod q
    # Note: u ~ u if u . u == 0. We ignore self-loops for simple graph.
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
