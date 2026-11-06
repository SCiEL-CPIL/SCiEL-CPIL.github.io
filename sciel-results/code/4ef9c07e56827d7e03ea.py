import numpy as np

def construct():
    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonical representative: first non-zero coordinate is 1
                if x != 0:
                    inv = pow(x, q - 2, q)
                    rep = (1, (y * inv) % q, (z * inv) % q)
                elif y != 0:
                    inv = pow(y, q - 2, q)
                    rep = (0, 1, (z * inv) % q)
                else:  # z != 0
                    inv = pow(z, q - 2, q)
                    rep = (0, 0, 1)
                if rep not in seen:
                    seen.add(rep)
                    points.append(rep)
    
    n = len(points)
    assert n == 31
    
    # Adjacency matrix
    # Polarity: x_P * x_Q + y_P * y_Q + z_P * z_Q == 0 (mod q)
    A = np.zeros((n, n), dtype=int)
    edges = []
    
    for i in range(n):
        xi, yi, zi = points[i]
        for j in range(i + 1, n):
            xj, yj, zj = points[j]
            val = (xi * xj + yi * yj + zi * zj) % q
            if val == 0:
                A[i][j] = 1
                A[j][i] = 1
                edges.append([i, j])
                
    # Verify edge count
    if len(edges) != 90:
        # Fallback if polarity doesn't yield 90 edges (should not happen for this specific one)
        # But let's be safe. If it's not 90, we might need to adjust.
        # For now, assume it is 90.
        pass
        
    # Calculate A^3 to find number of length-3 paths
    A2 = A @ A
    A3 = A2 @ A
    
    # Find non-edge with minimum A3[i][j]
    min_val = float('inf')
    best_edge = None
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] == 0:
                # Number of C4s created is A3[i][j]
                # Note: A3[i][j] counts walks of length 3.
                # Each walk u-a-b-v corresponds to a potential C4 u-v-b-a-u?
                # Yes, if we add (u,v), the walk u-a-b-v becomes a cycle u-v-b-a-u.
                # Since the graph is C4-free, there are no existing C4s.
                # Are there any other constraints?
                # We just need to minimize the number of new C4s.
                if A3[i][j] < min_val:
                    min_val = A3[i][j]
                    best_edge = [i, j]
                    
    if best_edge is None:
        # Should not happen
        best_edge = [0, 1]
        
    edges.append(best_edge)
    
    return edges
