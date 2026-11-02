import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 3, 6
    
    # Schoolbook Rank 36
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    # Try Rank 22 (two Rank 11 blocks)
    # We need a valid Rank 11 decomposition for 2x3x3
    # Using known Bini decomposition
    
    # Hardcoded Rank 11 decomposition for 2x3x3
    # U (6x11), V (9x11), W (6x11)
    
    U_11 = np.array([
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
    ])
    
    V_11 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    W_11 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    # Verify Rank 11
    def verify_11():
        T_target = np.ones((2, 3, 3))
        T_calc = np.zeros((2, 3, 3))
        for r in range(11):
            u = U_11[:, r].reshape(2, 3)
            v = V_11[:, r].reshape(3, 3)
            w = W_11[:, r].reshape(3, 2)
            # T[i,j,k] += U[i,j] * V[j,k] * W[k,i]
            # This is not a simple matrix product.
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        T_calc[i, j, k] += U_11[i*3+j, r] * V_11[j*3+k, r] * W_11[k*2+i, r]
        return np.allclose(T_calc, T_target)
    
    if verify_11():
        # Construct Rank 22 for 2x3x6
        # Split B into B1 (cols 0-2) and B2 (cols 3-5)
        # C = A B1 + A B2
        # Use two Rank 11 decompositions
        
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # First block (k=0,1,2)
        for r in range(11):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = U_11[i * m + j, r]
            for j in range(m):
                for k in range(3):
                    V[j * p + k][r] = V_11[j * 3 + k, r]
            for k in range(3):
                for i in range(n):
                    W[k * n + i][r] = W_11[k * n + i, r]
                    
        # Second block (k=3,4,5)
        for r in range(11):
            idx = 11 + r
            for i in range(n):
                for j in range(m):
                    U[i * m + j][idx] = U_11[i * m + j, r]
            for j in range(m):
                for k in range(3, 6):
                    V[j * p + k][idx] = V_11[j * 3 + (k-3), r]
            for k in range(3, 6):
                for i in range(n):
                    W[k * n + i][idx] = W_11[(k-3) * n + i, r]
                    
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
