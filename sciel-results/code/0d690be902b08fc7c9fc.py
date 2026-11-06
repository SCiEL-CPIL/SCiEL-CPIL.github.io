# EVOLVE-BLOCK-START
import itertools

def construct():
    # Construct Polarity Graph of PG(2, 5)
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    # Normalize points to unique representatives
    # Actually, the list above already has unique representatives for PG(2,q)
    # (x,y,1) for x,y in Fq, (x,1,0) for x in Fq, (1,0,0)
    # Total q^2 + q + 1 = 25 + 5 + 1 = 31
    
    n_polarity = len(points)
    edges = []
    
    # Compute edges for polarity graph
    # u . v = 0 mod q
    for i in range(n_polarity):
        for j in range(i + 1, n_polarity):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                edges.append([i, j])
                
    # Add 2 triangles and 1 edge
    # Current n = 31
    # Triangle 1: 31, 32, 33
    a, b, c = 31, 32, 33
    edges += [[a, b], [a, c], [b, c]]
    
    # Triangle 2: 34, 35, 36
    d, e, f = 34, 35, 36
    edges += [[d, e], [d, f], [e, f]]
    
    # Edge: 37, 38
    g, h = 37, 38
    edges += [[g, h]]
    
    n_total = 39
    return {'n': n_total, 'edges': edges}
# EVOLVE-BLOCK-END
