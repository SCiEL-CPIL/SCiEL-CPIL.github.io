import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 150  # seconds
    
    # Construct Polarity Graph of PG(2, 5)
    q = 5
    # Generate all non-zero vectors in F_5^3 modulo scalar multiplication
    # Points are represented as tuples (x, y, z) with x != 0 (normalized)
    # or simply generate all 31 points
    
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero component 1
                # This is tricky for projective space.
                # Easier: Just list all 31 unique projective points.
                pass
    
    # Standard way to generate PG(2, q) points:
    # 1. Points with x=0: (0, y, z). y,z not both 0.
    #    Normalize to (0, 1, k) for k in F_q, and (0, 0, 1).
    # 2. Points with x!=0: (x, y, z). Normalize to (1, a, b).
    
    pts = []
    # Points at infinity (x=0)
    for k in range(q):
        pts.append((0, 1, k))
    pts.append((0, 0, 1))
    
    # Affine points (x!=0)
    for a in range(q):
        for b in range(q):
            pts.append((1, a, b))
            
    n_pts = len(pts)
    assert n_pts == q*q + q + 1, f"Expected {q*q+q+1}, got {n_pts}"
    
    # Define Polarity
    # x^T A y = 0
    # A = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
    # x1*y1 + x2*y2 - x3*y3 = 0
    
    def orthogonal(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return (x1*x2 + y1*y2 - z1*z2) % q == 0
        
    # Build adjacency list
    adj = {i: set() for i in range(n_pts)}
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            if orthogonal(pts[i], pts[j]):
                adj[i].add(j)
                adj[j].add(i)
                
    # Check C4-free (should be by construction)
    # Check Independence Number
    # For q=5, alpha is small.
    
    edges = []
    for i in range(n_pts):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
                
    return {'n': n_pts, 'edges': edges}
