import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 118s for safety
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Try Bini rank-23 for 3x3x3 part + 9 schoolbook for 4th column = rank 32
    # Verify Bini decomposition at runtime
    
    def E(i, j):
        M = np.zeros((3, 3))
        M[i, j] = 1.0
        return M
    
    bini_terms = []
    bini_terms.append((E(0,0), E(0,0), E(0,0)))
    bini_terms.append((E(0,1), E(1,0), E(0,0)))
    bini_terms.append((E(0,2), E(2,0), E(0,0)))
    bini_terms.append((E(1,0), E(0,1), E(0,1)))
    bini_terms.append((E(1,1), E(1,1), E(0,1)))
    bini_terms.append((E(1,2), E(2,1), E(0,1)))
    bini_terms.append((E(2,0), E(0,2), E(0,2)))
    bini_terms.append((E(2,1), E(1,2), E(1,2)))
    bini_terms.append((E(2,2), E(2,2), E(1,2)))
    bini_terms.append((E(0,0)-E(1,0), E(0,0)-E(1,0), E(0,0)-E(1,0)))
    bini_terms.append((E(0,1)-E(1,1), E(1,0)-E(1,1), E(0,1)-E(1,1)))
    bini_terms.append((E(0,2)-E(1,2), E(2,0)-E(2,1), E(0,2)-E(1,2)))
    bini_terms.append((E(1,0)-E(2,0), E(0,0)-E(0,1), E(1,0)-E(2,0)))
    bini_terms.append((E(1,1)-E(2,1), E(1,0)-E(1,1), E(1,1)-E(2,1)))
    bini_terms.append((E(1,2)-E(2,2), E(2,0)-E(2,1), E(1,2)-E(2,2)))
    bini_terms.append((E(0,0)-E(0,1), E(0,0)-E(0,1), E(0,0)-E(0,1)))
    bini_terms.append((E(1,0)-E(1,1), E(1,0)-E(1,1), E(1,0)-E(1,1)))
    bini_terms.append((E(2,0)-E(2,1), E(2,0)-E(2,1), E(2,0)-E(2,1)))
    bini_terms.append((E(0,0)-E(0,2), E(0,0)-E(0,2), E(0,0)-E(0,2)))
    bini_terms.append((E(1,0)-E(1,2), E(1,0)-E(1,2), E(1,0)-E(1,2)))
    bini_terms.append((E(2,0)-E(2,2), E(2,0)-E(2,2), E(2,0)-E(2,2)))
    bini_terms.append((E(0,1)-E(0,2), E(1,0)-E(2,0), E(0,1)-E(0,2)))
    bini_terms.append((E(1,1)-E(1,2), E(1,1)-E(2,1), E(1,1)-E(1,2)))
    
    # Verify Bini decomposition
    def verify_bini(terms, num_tests=50):
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randn(3, 3)
            B = np.random.randn(3, 3)
            C_true = A @ B
            C_recon = np.zeros((3, 3))
            for U, V, W in terms:
                u_val = np.sum(U * A)
                v_val = np.sum(V * B)
                C_recon += u_val * v_val * W
            if not np.allclose(C_recon, C_true, atol=1e-10):
                return False
        return True
    
    bini_valid = verify_bini(bini_terms)
    
    if bini_valid:
        rank = 23 + 9
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Fill Bini terms (columns 0-22)
        for c, (U_mat, V_mat, W_mat) in enumerate(bini_terms):
            for i in range(3):
                for j in range(3):
                    U[i * m + j, c] = U_mat[i, j]
            for j in range(3):
                for k in range(3):
                    V[j * p + k, c] = V_mat[j, k]
            for k in range(3):
                for i in range(3):
                    W[k * n + i, c] = W_mat[k, i]
        
        # Fill schoolbook terms for 4th column (columns 23-31)
        col = 23
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = 1
                V[j * p + 3, col] = 1
                W[3 * n + i, col] = 1
                col += 1
        
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook rank 36
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
        return {"U": U, "V": V, "W": W}
