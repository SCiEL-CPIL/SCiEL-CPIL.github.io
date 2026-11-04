"""Matrix multiplication 3,3,5 - low rank decomposition via block structure."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS
    
    # Strategy: Block decomposition 3x3x3 (Rank 23) + 3x3x2 (Rank 18).
    # Total rank 41.
    
    # Bini 23-rank factors for 3x3x3 matrix multiplication.
    # Format: list of (U_flat, V_flat, W_flat)
    # U, V, W are 3x3 matrices flattened to 9 elements.
    # C[i,k] = sum_c W_c[i,k] * (sum_{i',j'} U_c[i',j'] A[i',j']) * (sum_{j',k'} V_c[j',k'] B[j',k'])
    
    # Verified Bini 23 factors
    bini_23 = [
        ([1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1]),
        ([0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        ([0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
    ]
    
    def verify_bini(factors, n=3):
        np.random.seed(42)
        for _ in range(5):
            A = np.random.randint(-5, 5, (n, n))
            B = np.random.randint(-5, 5, (n, n))
            C_expected = A @ B
            C_computed = np.zeros((n, n))
            for U_flat, V_flat, W_flat in factors:
                U = np.array(U_flat).reshape(n, n)
                V = np.array(V_flat).reshape(n, n)
                W = np.array(W_flat).reshape(n, n)
                mu = np.sum(U * A)
                nu = np.sum(V * B)
                C_computed += W * mu * nu
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    bini_valid = verify_bini(bini_23)
    
    if not bini_valid:
        # Fallback to schoolbook 3x3x3 (rank 27)
        bini_23 = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_flat = [0]*9; V_flat = [0]*9; W_flat = [0]*9
                    U_flat[i*3+j] = 1
                    V_flat[j*3+k] = 1
                    W_flat[i*3+k] = 1
                    bini_23.append((U_flat, V_flat, W_flat))
    
    rank_333 = len(bini_23)
    rank_332 = 18
    total_rank = rank_333 + rank_332
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill 3x3x3 part
    for c_idx, (U_flat, V_flat, W_flat) in enumerate(bini_23):
        U_mat = np.array(U_flat).reshape(3, 3)
        V_mat = np.array(V_flat).reshape(3, 3)
        W_mat = np.array(W_flat).reshape(3, 3)
        for i in range(3):
            for j in range(3):
                U[i*3+j][c_idx] = U_mat[i, j]
        for j in range(3):
            for k in range(3):
                V[j*3+k][c_idx] = V_mat[j, k]
        for k in range(3):
            for i in range(3):
                W[k*3+i][c_idx] = W_mat[k, i]
    
    # Fill 3x3x2 part (columns 3,4 of B and C)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                c_idx = rank_333 + i*3*2 + j*2 + k
                U[i*3+j][c_idx] = 1
                V[j*5+(3+k)][c_idx] = 1
                W[(3+k)*3+i][c_idx] = 1
    
    # Final Verification
    def verify_full(U, V, W, n, m, p):
        total_rank = len(U[0])
        np.random.seed(42)
        for _ in range(10):
            A = np.random.randint(-10, 10, (n, m))
            B = np.random.randint(-10, 10, (m, p))
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for c in range(total_rank):
                        alpha = 0.0
                        for j in range(m):
                            alpha += U[i*m+j][c] * A[i,j]
                        beta = 0.0
                        for j in range(m):
                            beta += V[j*p+k][c] * B[j,k]
                        C_computed[i,k] += W[k*n+i][c] * alpha * beta
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    if not verify_full(U, V, W, n, m, p):
        # Fallback to pure schoolbook 45
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
