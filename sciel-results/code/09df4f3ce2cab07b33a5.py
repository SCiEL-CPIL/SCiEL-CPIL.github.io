import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 120  # seconds
    
    def check_time():
        return time.time() - start_time > deadline - 5
    
    q = 7
    # Generate points of PG(2, q)
    # Points are 1D subspaces of F_q^3
    # Represent by (x, y, z) with x,y,z in F_q, not all 0
    # Normalize: first non-zero is 1
    
    points = []
    # Iterate over all non-zero vectors
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    points.append((1, (y * pow(x, q-2, q)) % q, (z * pow(x, q-2, q)) % q))
                elif y != 0:
                    points.append((0, 1, (z * pow(y, q-2, q)) % q))
                else:
                    points.append((0, 0, 1))
    
    n = len(points)
    # n should be q^2 + q + 1
    
    # Polarity matrix M
    # M = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    # Dot product: u0*v1 + u1*v0 + u2*v2
    
    edges = []
    
    # Check time
    if check_time():
        return {'n': 30, 'edges': []} # Fallback
    
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            # Calculate u^T M v
            # u0*v1 + u1*v0 + u2*v2
            val = (u[0] * v[1] + u[1] * v[0] + u[2] * v[2]) % q
            if val == 0:
                edges.append([i, j])
                
    if check_time():
        return {'n': 30, 'edges': []}
        
    return {'n': n, 'edges': edges}
