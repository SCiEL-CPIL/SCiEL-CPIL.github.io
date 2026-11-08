import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition of T."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_approx = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_approx, T, atol=1e-8)

    def to_list(arr):
        return [[float(x) for x in row] for row in arr]

    # Fallback: Rank 28 solution
    def get_rank_28_solution():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # Part 1: A2 B2 (j=2, k=0..4)
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        # Part 2: A1 B13 (j=0,1, k=4)
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1
        r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1
        r += 1
        U[0 * m + 0][r] = 1
        V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1
        W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1
        r += 1
        U[1 * m + 1][r] = 1
        V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1
        W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1
        r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1
        r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
        V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
        V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1
        r += 1
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1
        r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1
        W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1
        r += 1
        U[0 * m + 0][r] = 1
        V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1
        W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1
        r += 1
        U[1 * m + 1][r] = 1
        V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1
        W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1
        r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1
        r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
        V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
        V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1
        r += 1
        
        return {"U": U, "V": V, "W": W}

    # Try to find a rank 27 solution using ALS with multiple restarts
    def try_als_for_rank(target_rank, max_iters=50, n_restarts=10):
        if time.time() > deadline:
            return None
        
        T_flat_U = T.reshape(n * m, -1)
        T_flat_V = T.reshape(m * p, -1)
        T_flat_W = T.reshape(p * n, -1)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                B_mat_T = np.zeros((m * p * p * n, target_rank))
                for r in range(target_rank):
                    B_mat_T[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                try:
                    P_U = np.linalg.pinv(B_mat_T)
                    U = T_flat_U @ P_U
                except:
                    break
                
                # Update V
                B_mat_T_V = np.zeros((n * m * p * n, target_rank))
                for r in range(target_rank):
                    B_mat_T_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                try:
                    P_V = np.linalg.pinv(B_mat_T_V)
                    V = T_flat_V @ P_V
                except:
                    break
                
                # Update W
                B_mat_T_W = np.zeros((n * m * m * p, target_rank))
                for r in range(target_rank):
                    B_mat_T_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                try:
                    P_W = np.linalg.pinv(B_mat_T_W)
                    W = T_flat_W @ P_W
                except:
                    break
                
                if verify(U, V, W):
                    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                    
        return None

    # Try rank 27
    sol_27 = try_als_for_rank(27)
    if sol_27:
        return sol_27
        
    # Return rank 28
    return get_rank_28_solution()

# EVOLVE-BLOCK-END
