"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try to use the 23-term algorithm if available and valid
    # Otherwise fall back to schoolbook (rank 27)
    
    # Schoolbook fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Verification function
    def verify(U, V, W, rank):
        # Test with a few random matrices
        for _ in range(10):
            A = np.random.randint(-10, 10, (n, m)).astype(float)
            B = np.random.randint(-10, 10, (m, p)).astype(float)
            C_expected = A @ B
            
            C_calc = np.zeros((n, p))
            for r in range(rank):
                # u_i = sum_j A_ij U[i*m+j, r]
                u = np.array([sum(A[i, j] * U[i * m + j][r] for j in range(m)) for i in range(n)])
                # v_k = sum_j V[j*p+k, r] B_jk
                v = np.array([sum(V[j * p + k][r] * B[j, k] for j in range(m)) for k in range(p)])
                # C_ik += u_i * v_k * W[k*n+i, r]
                for i in range(n):
                    for k in range(p):
                        C_calc[i, k] += u[i] * v[k] * W[k * n + i][r]
                        
            if not np.allclose(C_expected, C_calc, atol=1e-9):
                return False
        return True

    # If we had a 23-term algorithm, we would test it here.
    # Since I don't have the exact 23-term coefficients hardcoded reliably, 
    # I will return the schoolbook algorithm.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
