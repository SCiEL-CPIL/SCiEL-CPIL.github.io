import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 3, 9
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # Trivial rank-54 decomposition
    U_trivial = [[0] * 54 for _ in range(n * m)]
    V_trivial = [[0] * 54 for _ in range(m * p)]
    W_trivial = [[0] * 54 for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = (i * m + j) * p + k
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
    
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    best_rank = 54
    
    # For 2x3 * 3x9 matrix multiplication, we try to find lower rank decompositions
    # The tensor has dimensions (n*m) x (m*p) x (p*n) = 6 x 27 x 18
    # We'll use a structured approach to try to find lower rank
    
    # Try to exploit the structure: for each (i,j,k), we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to sum to 1
    # This is a bilinear problem in U,V,W for each tensor entry
    
    # Let's try rank r from 27 to 53 (since we need at least max(n*m, m*p, p*n) = 27)
    # Use randomized search with local optimization
    
    def try_rank(r, max_iters=1000):
        """Try to find rank-r decomposition using randomized local search"""
        if time.time() > deadline:
            return None
        
        # Initialize with random matrices
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Scale to avoid overflow
        U /= np.sqrt(r)
        V /= np.sqrt(r)
        W /= np.sqrt(r)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute error: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r_idx in range(r):
                            s += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                        err += (s - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
            
            # Local search: perturb one matrix
            matrix = random.choice([U, V, W])
            row = random.randint(0, matrix.shape[0] - 1)
            col = random.randint(0, r - 1)
            old_val = matrix[row, col]
            matrix[row, col] += random.gauss(0, 0.1)
            
            # Compute new error
            new_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r_idx in range(r):
                            s += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                        new_err += (s - 1.0) ** 2
            
            # Accept if better, or with some probability (simulated annealing)
            if new_err < err or random.random() < 0.1:
                err = new_err
            else:
                matrix[row, col] = old_val
        
        if best_err < 1e-10:
            return best_U, best_V, best_W
        return None
    
    # Try various ranks
    for r in range(27, 54):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iters=500)
        if result is not None:
            U, V, W = result
            # Convert to list format
            U_list = [[U[i][j] for j in range(r)] for i in range(n * m)]
            V_list = [[V[i][j] for j in range(r)] for i in range(m * p)]
            W_list = [[W[i][j] for j in range(r)] for i in range(p * n)]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = r
            break
    
    return best_result
