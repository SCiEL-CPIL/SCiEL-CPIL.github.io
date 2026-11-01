import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    # A is n x m (2x2), B is m x p (2x10), C is n x p (2x10)
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Decomposition: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly computes matrix multiplication."""
        n, m, p = 2, 2, 10
        R = len(U[0])
        # Generate random test cases
        np.random.seed(42)
        for trial in range(10):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(R):
                # u_A: for each i, sum_j U[i*m+j, r] * A[i,j]
                u_A = np.zeros(n)
                for i in range(n):
                    for j in range(m):
                        u_A[i] += U[i*m + j][r] * A[i, j]
                
                # v_B: for each k, sum_j V[j*p+k, r] * B[j,k]
                v_B = np.zeros(p)
                for k in range(p):
                    for j in range(m):
                        v_B[k] += V[j*p + k][r] * B[j, k]
                
                # C[i,k] += W[k*n+i, r] * u_A[i] * v_B[k]
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += W[k*n + i][r] * u_A[i] * v_B[k]
            
            if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
                return False
        return True

    def build_schoolbook_solution():
        """
        Constructs a rank-40 schoolbook decomposition for 2x2 * 2x10 matrix multiplication.
        """
        n, m, p = 2, 2, 10
        R = n * m * p  # 40
        
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        
        return U, V, W

    U, V, W = build_schoolbook_solution()
    
    if not verify_decomposition(U, V, W):
        raise ValueError("Decomposition failed")

    return {"U": U, "V": V, "W": W}

