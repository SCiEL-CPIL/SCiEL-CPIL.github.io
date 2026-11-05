import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (Rank 54)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    return {"U": U_school, "V": V_school, "W": W_school}

def verify_decomposition(U, V, W, n, m, p, num_tests=10):
    """
    Verify that sum_c U[:,c] * V[:,c] * W[:,c] == M_{n,m,p}
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    """
    rank = len(U[0])
    for _ in range(num_tests):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        
        # Compute via decomposition
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # Sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This should equal C[i, k]
        
        # Vectorized check
        # Reshape U to (n, m, rank)
        U_reshaped = np.array(U).reshape(n, m, rank)
        V_reshaped = np.array(V).reshape(m, p, rank)
        W_reshaped = np.array(W).reshape(p, n, rank)
        
        # C[i, k] = sum_j sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        # Note: W index is k*n + i, so W[k, i, c]
        
        C_comp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for c in range(rank):
                        s += U_reshaped[i, j, c] * V_reshaped[j, k, c] * W_reshaped[k, i, c]
                C_comp[i, k] = s
        
        if not np.allclose(C, C_comp, atol=1e-9):
            return False
    return True

def get_rank_23_3x3():
    """
    Returns U, V, W for 3x3 matrix multiplication with rank 23.
    If not available or invalid, returns None.
    """
    # Placeholder for rank 23 algorithm
    # I will try to use a known set of coefficients.
    # If I can't verify them, I won't use them.
    return None

# EVOLVE-BLOCK-END
