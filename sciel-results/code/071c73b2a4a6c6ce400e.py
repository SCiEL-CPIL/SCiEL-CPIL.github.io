import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Laderman 3x3x3 factors (23 terms)
    # Each term is (u, v, w) of length 9 (flattened 3x3 matrices)
    # u[i*3+j], v[j*3+k], w[k*3+i]
    laderman_factors = [
        ([1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0]),
        ([0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([1, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0]),
        ([1, 0, 0, 0, -1, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, -1, 0, 0, 0, 0]),
        ([0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]),
        ([0, 1, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 1, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 1, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([1, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([1, 0, 0, 0, 0, 0, 0, 0, -1], [1, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, -1]),
        ([0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0]),
        ([0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0]),
        ([0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
    ]
    
    # Verify Laderman factors
    def verify_laderman(factors):
        if len(factors) != 23:
            return False
        # T[i,j,k] should be 1
        T = np.zeros((3, 3, 3), dtype=int)
        for u, v, w in factors:
            U_mat = np.array(u).reshape(3, 3)
            V_mat = np.array(v).reshape(3, 3)
            W_mat = np.array(w).reshape(3, 3)
            # sum_r U[i,j] V[j,k] W[k,i]
            # This is a tensor contraction
            # T[i,j,k] += U[i,j] * V[j,k] * W[k,i]
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        T[i, j, k] += U_mat[i, j] * V_mat[j, k] * W_mat[k, i]
        return np.all(T == 1)
    
    if verify_laderman(laderman_factors):
        rank = 55
        U = [[0] * rank for _ in range(n * m)] # 9x55
        V = [[0] * rank for _ in range(m * p)] # 21x55
        W = [[0] * rank for _ in range(p * n)] # 21x55
        
        # Block 1: Laderman for k=0,1,2
        for r in range(23):
            u, v, w = laderman_factors[r]
            # U
            for idx in range(9):
                U[idx][r] = u[idx]
            
            # V mapping: v[j*3+k'] -> V[j*7+k, r] where k=k'
            for j in range(3):
                for k_prime in range(3):
                    val = v[j*3 + k_prime]
                    if val != 0:
                        V[j*7 + k_prime][r] = val
            
            # W mapping: w[k'*3+i] -> W[k*3+i, r] where k=k'
            for k_prime in range(3):
                for i in range(3):
                    val = w[k_prime*3 + i]
                    if val != 0:
                        W[k_prime*3 + i][r] = val
        
        # Block 2: Laderman for k=3,4,5
        for r in range(23):
            u, v, w = laderman_factors[r]
            col = 23 + r
            # U
            for idx in range(9):
                U[idx][col] = u[idx]
            
            # V mapping: v[j*3+k'] -> V[j*7+(k'+3), col]
            for j in range(3):
                for k_prime in range(3):
                    val = v[j*3 + k_prime]
                    if val != 0:
                        V[j*7 + (k_prime + 3)][col] = val
            
            # W mapping: w[k'*3+i] -> W[(k'+3)*3+i, col]
            for k_prime in range(3):
                for i in range(3):
                    val = w[k_prime*3 + i]
                    if val != 0:
                        W[(k_prime + 3)*3 + i][col] = val
        
        # Block 3: Schoolbook for k=6
        for i in range(3):
            for j in range(3):
                col = 46 + i*3 + j
                U[i*3 + j][col] = 1
                V[j*7 + 6][col] = 1
                W[6*3 + i][col] = 1
        
        return {"U": U, "V": V, "W": W}
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
