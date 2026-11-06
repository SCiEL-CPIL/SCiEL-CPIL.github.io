import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # GF(9) implementation
    # Elements are (a, b) for a + b*omega, omega^2 = 2
    def gf9_add(x, y):
        return ((x[0] + y[0]) % 3, (x[1] + y[1]) % 3)
        
    def gf9_mul(x, y):
        a1, b1 = x
        a2, b2 = y
        real = (a1*a2 + 2*b1*b2) % 3
        imag = (a1*b2 + a2*b1) % 3
        return (real, imag)
        
    gf9_elements = [(a, b) for a in range(3) for b in range(3)]
    
    # Generate points in PG(2, 9)
    points = []
    # (1, a, b)
    for a in gf9_elements:
        for b in gf9_elements:
            points.append(((1, 0), a, b))
    # (0, 1, b)
    for b in gf9_elements:
        points.append(((0, 0), (1, 0), b))
    # (0, 0, 1)
    points.append(((0, 0), (0, 0), (1, 0)))
    
    n = len(points)
    edges = []
    
    for i in range(n):
        if time.time() > deadline:
            break
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            s1 = gf9_mul(u[0], v[0])
            s2 = gf9_mul(u[1], v[1])
            s3 = gf9_mul(u[2], v[2])
            total = gf9_add(gf9_add(s1, s2), s3)
            if total == (0, 0):
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
