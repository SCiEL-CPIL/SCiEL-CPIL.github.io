import numpy as np
import random

def construct():
    # Generate points of P^2(F_5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    cx = 1
                    cy = (y * inv_x) % 5
                    cz = (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    cx = 0
                    cy = 1
                    cz = (z * inv_y) % 5
                else:
                    cx = 0
                    cy = 0
                    cz = 1
                
                p = (cx, cy, cz)
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    n = len(points)
    A = np.zeros((n, n), dtype=int)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            b = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if b == 0:
                A[i][j] = 1
                A[j][i] = 1
                edges.append([i, j])
                
    # Add the best edge to reach 91
    A3 = np.linalg.matrix_power(A, 3)
    best_edge = None
    min_paths = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] == 0:
                paths = A3[i][j]
                if paths < min_paths:
                    min_paths = paths
                    best_edge = (i, j)
                    
    A[best_edge[0]][best_edge[1]] = 1
    A[best_edge[1]][best_edge[0]] = 1
    edges.append(list(best_edge))
    
    # Local Search / Simulated Annealing to minimize C4
    def count_c4(A_mat):
        A2 = A_mat @ A_mat
        # Sum of combinations of 2 for each entry
        # C4 = 1/2 * sum( C(A2[i][j], 2) )
        # C(k, 2) = k*(k-1)/2
        c4_sum = np.sum(A2 * (A2 - 1) // 2)
        return c4_sum // 2

    current_c4 = count_c4(A)
    
    # Simulated Annealing parameters
    T = 10.0
    cooling = 0.99
    iterations = 20000
    
    # Precompute list of edges and non-edges for fast swapping?
    # Or just use matrix.
    
    for _ in range(iterations):
        if T < 0.1:
            break
            
        # Pick random edge to remove
        # Flatten A, find indices where A==1, pick one
        # Faster: maintain list of edges
        # Let's just use the matrix for simplicity, n=31 is small
        
        # Random edge
        while True:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and A[u][v] == 1:
                break
                
        # Random non-edge
        while True:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x != y and A[x][y] == 0:
                break
                
        # Perform swap
        A[u][v] = 0
        A[v][u] = 0
        A[x][y] = 1
        A[y][x] = 1
        
        new_c4 = count_c4(A)
        
        delta = new_c4 - current_c4
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_c4 = new_c4
        else:
            # Revert
            A[u][v] = 1
            A[v][u] = 1
            A[x][y] = 0
            A[y][x] = 0
            
        T *= cooling
        
    # Extract edges from final A
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] == 1:
                final_edges.append([i, j])
                
    return final_edges
