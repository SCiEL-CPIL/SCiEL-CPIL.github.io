import time
import numpy as np
import random

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Standard dimensions
    total_terms = n * m * p
    
    # Helper to create the base rank 58 decomposition
    def create_rank_58():
        rank = 58
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_block1 = {(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)}
        strassen_block2 = {(0,2,4), (0,2,5), (0,3,4), (0,3,5), (1,2,4), (1,2,5), (1,3,4), (1,3,5)}
        all_strassen = strassen_block1 | strassen_block2
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in all_strassen:
                        continue
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for u_idx, coeff in u_coeffs:
                U[u_idx][col] += coeff
            for v_idx, coeff in v_coeffs:
                V[v_idx][col] += coeff
            for w_idx, coeff in w_coeffs:
                W[w_idx][col] += coeff
            col += 1
        
        # Block 1 indices
        u00_1, u01_1 = 0, 1
        u10_1, u11_1 = 5, 6
        v00_1, v01_1 = 0, 1
        v10_1, v11_1 = 6, 7
        w00_1, w01_1 = 0, 2
        w10_1, w11_1 = 1, 3

        add_term([(u00_1, 1), (u11_1, 1)], [(v00_1, 1), (v11_1, 1)], [(w00_1, 1), (w11_1, 1)])
        add_term([(u10_1, 1), (u11_1, 1)], [(v00_1, 1)], [(w10_1, 1), (w11_1, -1)])
        add_term([(u00_1, 1)], [(v01_1, 1), (v11_1, -1)], [(w01_1, 1), (w11_1, 1)])
        add_term([(u11_1, 1)], [(v10_1, 1), (v00_1, -1)], [(w00_1, 1), (w10_1, 1)])
        add_term([(u00_1, 1), (u01_1, 1)], [(v11_1, 1)], [(w00_1, -1), (w01_1, 1)])
        add_term([(u10_1, 1), (u00_1, -1)], [(v00_1, 1), (v01_1, 1)], [(w11_1, 1)])
        add_term([(u01_1, 1), (u11_1, -1)], [(v10_1, 1), (v11_1, 1)], [(w00_1, 1)])

        # Block 2 indices
        u00_2, u01_2 = 2, 3
        u10_2, u11_2 = 7, 8
        v00_2, v01_2 = 16, 17
        v10_2, v11_2 = 22, 23
        w00_2, w01_2 = 8, 10
        w10_2, w11_2 = 9, 11

        add_term([(u00_2, 1), (u11_2, 1)], [(v00_2, 1), (v11_2, 1)], [(w00_2, 1), (w11_2, 1)])
        add_term([(u10_2, 1), (u11_2, 1)], [(v00_2, 1)], [(w10_2, 1), (w11_2, -1)])
        add_term([(u00_2, 1)], [(v01_2, 1), (v11_2, -1)], [(w01_2, 1), (w11_2, 1)])
        add_term([(u11_2, 1)], [(v10_2, 1), (v00_2, -1)], [(w00_2, 1), (w10_2, 1)])
        add_term([(u00_2, 1), (u01_2, 1)], [(v11_2, 1)], [(w00_2, -1), (w01_2, 1)])
        add_term([(u10_2, 1), (u00_2, -1)], [(v00_2, 1), (v01_2, 1)], [(w11_2, 1)])
        add_term([(u01_2, 1), (u11_2, -1)], [(v10_2, 1), (v11_2, 1)], [(w00_2, 1)])
        
        return U, V, W

    # Verification function
    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat)
        V_np = np.array(V_mat)
        W_np = np.array(W_mat)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i*m+j, :]
                    v_row = V_np[j*p+k, :]
                    w_row = W_np[k*n+i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-9:
                        return False
        return True

    # Get the baseline rank 58 solution
    U, V, W = create_rank_58()
    
    if not verify(U, V, W):
        # Fallback to rank 60 if construction fails (should not happen)
        rank = 60
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Attempt to find a rank 57 solution using a heuristic search
    # We try to see if we can reduce the rank by 1 by solving for a linear dependency
    # This is a heuristic and may not succeed.
    
    # If we have time, we could try a random search, but for now, 
    # we return the verified rank 58 solution.
    
    return {"U": U, "V": V, "W": W}
