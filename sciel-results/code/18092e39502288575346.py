import numpy as np

def construct():
    # Field GF(5)
    # Points of PG(2, 5)
    # (x, y, z)
    points = []
    # Affine part: z=1
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    # Line at infinity: z=0
    # (1, m, 0) for m in 0..4
    for m in range(5):
        points.append((1, m, 0))
    # (0, 1, 0)
    points.append((0, 1, 0))
    
    n = len(points)
    # n should be 31
    
    # Adjacency matrix
    A = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check orthogonality
            # x_i x_j + y_i y_j + z_i z_j == 0 mod 5
            val = (points[i][0] * points[j][0] + 
                   points[i][1] * points[j][1] + 
                   points[i][2] * points[j][2]) % 5
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1
                
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges.append([i, j])
                
    # Should have 90 edges
    # Compute A^3
    A2 = A @ A
    A3 = A2 @ A
    
    # Find non-edge with min (A^3)[i, j]
    min_val = float('inf')
    best_pair = None
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                val = A3[i, j]
                if val < min_val:
                    min_val = val
                    best_pair = (i, j)
                    
    edges.append([best_pair[0], best_pair[1]])
    
    return edges
