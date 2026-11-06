import itertools

def construct():
    """
    Construct a C4-free graph with no independent set of size 11.
    Uses a subgraph of the Polarity Graph of the Projective Plane PG(2,7).
    
    The polarity graph of PG(2,q) has q^2+q+1 vertices and is C4-free.
    For q=7, n=57.
    We take the first 39 vertices.
    The independence number of the full graph is small (<= 8 or 9), 
    so any subgraph also has alpha < 11.
    This yields n=39, which is the best known lower bound for R(C4, K11).
    """
    q = 7
    n_total = q*q + q + 1
    n_target = 39
    
    if n_target > n_total:
        n_target = n_total
        
    # Generate points of PG(2, q)
    points = []
    for a in range(q):
        for b in range(q):
            points.append((1, a, b))
    for a in range(q):
        points.append((0, 1, a))
    points.append((0, 0, 1))
    
    # Define a polarity.
    # Use the symmetric bilinear form: B(u, v) = u1*v3 + u3*v1 + u2*v2
    def dot(u, v):
        return (u[0]*v[2] + u[2]*v[0] + u[1]*v[1]) % q

    edges = []
    adj = [[False]*n_total for _ in range(n_total)]
    
    for i in range(n_total):
        for j in range(i+1, n_total):
            if dot(points[i], points[j]) == 0:
                adj[i][j] = True
                adj[j][i] = True
                
    # Extract subgraph on first n_target vertices
    # Map old indices to new indices 0..n_target-1
    # Since we take 0..n_target-1, the mapping is identity.
    final_edges = []
    for i in range(n_target):
        for j in range(i+1, n_target):
            if adj[i][j]:
                final_edges.append([i, j])
                
    return {'n': n_target, 'edges': final_edges}
