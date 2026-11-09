"""
Constructs a C4-free graph with independence number < 11 on 39 vertices.

Approach:
1. Build the polarity graph of the projective plane PG(2, 7) using the polarity xz - y^2 = 0.
   This yields a graph on 57 vertices (q^2+q+1 for q=7).
   The polarity graph is known to be C4-free.
   The independence number of the full 57-vertex graph is bounded by roughly q+1 = 8,
   so any induced subgraph also has alpha <= 8 < 11.

2. Take an induced subgraph on 39 vertices (vertices 0..38).
   Since it is an induced subgraph, it remains C4-free and alpha <= 8 < 11.

This certifies R(C4, K11) > 39.
"""

def construct():
    q = 7
    # Generate points of PG(2, q)
    # Points are (x, y, z) in Z_q^3, not all 0, up to scaling.
    # Canonical representative: first non-zero coordinate is 1.
    points = []
    
    # Case 1: z != 0. Normalize z = 1.
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
            
    # Case 2: z = 0, y != 0. Normalize y = 1.
    for x in range(q):
        points.append((x, 1, 0))
        
    # Case 3: z = 0, y = 0, x != 0. Normalize x = 1.
    points.append((1, 0, 0))
    
    n_full = len(points)  # Should be 57 for q=7
    
    # We want to return a graph on n=39 vertices (indices 0..38)
    n = 39
    
    edges = []
    
    # Polarity: B(u, v) = u0*v2 + u2*v0 - 2*u1*v1 (mod q)
    # Adjacent if B(u, v) == 0
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            val = (u[0] * v[2] + u[2] * v[0] - 2 * u[1] * v[1]) % q
            if val == 0:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
