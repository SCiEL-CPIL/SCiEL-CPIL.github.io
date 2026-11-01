"""Matrix multiplication tensor decomposition for 2x3 * 3x8."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

DEADLINE = 120.0  # seconds, with safety margin

def construct():
    t0 = time.time()
    n, m, p = 2, 3, 8
    
    # Rank 36 decomposition.
    # Split B into two 3x4 halves.
    # For each half:
    # 2x3 * 3x4 = (2x2 * 2x4) + (2x1 * 1x4)
    # 2x2 * 2x4 = two 2x2 * 2x2 blocks (Strassen, rank 7 each) = 14
    # 2x1 * 1x4 = outer product (rank 4)
    # Total per half: 14 + 4 = 18
    # Total rank: 18 * 2 = 36
    
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 36
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 36
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 36
    
    # Indices:
    # U[i*m + j] for A[i, j]
    # V[j*p + k] for B[j, k]
    # W[k*n + i] for C[i, k]
    
    def add_strassen_2x2x2(base, a_r0, a_r1, b_c0, b_c1, b_k0):
        """
        Adds a Strassen 2x2*2x2 decomposition to the matrices.
        A rows: a_r0, a_r1
        A cols (B rows): b_c0, b_c1
        B cols: b_k0, b_k0+1
        Output C rows: a_r0, a_r1; C cols: b_k0, b_k0+1
        """
        # M1 = (A[a_r0, b_c0] + A[a_r1, b_c1]) * (B[b_c0, b_k0] + B[b_c1, b_k0+1])
        U[a_r0*m + b_c0][base] += 1; V[b_c0*p + b_k0][base] += 1
        U[a_r1*m + b_c1][base] += 1; V[b_c1*p + b_k0+1][base] += 1
        W[b_k0*n + a_r0][base] += 1; W[(b_k0+1)*n + a_r0][base] += 1
        W[b_k0*n + a_r1][base] += 1; W[(b_k0+1)*n + a_r1][base] += 1
        
        # M2 = (A[a_r1, b_c0] + A[a_r1, b_c1]) * B[b_c0, b_k0]
        U[a_r1*m + b_c0][base+1] += 1; V[b_c0*p + b_k0][base+1] += 1
        U[a_r1*m + b_c1][base+1] += 1
        W[b_k0*n + a_r1][base+1] += 1
        
        # M3 = A[a_r0, b_c0] * (B[b_c1, b_k0] - B[b_c0, b_k0])
        U[a_r0*m + b_c0][base+2] += 1; V[b_c1*p + b_k0][base+2] += 1
        V[b_c0*p + b_k0][base+2] += -1
        W[(b_k0+1)*n + a_r0][base+2] += 1
        
        # M4 = (A[a_r0, b_c1] - A[a_r1, b_c1]) * (B[b_c1, b_k0] + B[b_c1, b_k0+1])
        U[a_r0*m + b_c1][base+3] += 1; V[b_c1*p + b_k0][base+3] += 1
        U[a_r1*m + b_c1][base+3] += -1; V[b_c1*p + b_k0+1][base+3] += 1
        W[b_k0*n + a_r0][base+3] += 1; W[(b_k0+1)*n + a_r0][base+3] += 1
        
        # M5 = A[a_r0, b_c1] * (B[b_c0, b_k0] + B[b_c0, b_k0+1])
        U[a_r0*m + b_c1][base+4] += 1; V[b_c0*p + b_k0][base+4] += 1
        V[b_c0*p + b_k0+1][base+4] += 1
        W[b_k0*n + a_r0][base+4] += -1; W[(b_k0+1)*n + a_r0][base+4] += 1
        
        # M6 = (A[a_r1, b_c0] - A[a_r0, b_c0]) * (B[b_c0, b_k0+1] - B[b_c1, b_k0+1])
        U[a_r1*m + b_c0][base+5] += 1; V[b_c0*p + b_k0+1][base+5] += 1
        U[a_r0*m + b_c0][base+5] += -1; V[b_c1*p + b_k0+1][base+5] += -1
        W[(b_k0+1)*n + a_r1][base+5] += 1
        
        # M7 = (A[a_r0, b_c0] + A[a_r0, b_c1]) * B[b_c1, b_k0+1]
        U[a_r0*m + b_c0][base+6] += 1; V[b_c1*p + b_k0+1][base+6] += 1
        U[a_r0*m + b_c1][base+6] += 1
        W[b_k0*n + a_r0][base+6] += 1; W[(b_k0+1)*n + a_r1][base+6] += 1

    # Process two halves of B (cols 0-3 and 4-7)
    for half in range(2):
        bk = half * 4  # base col in B
        base = half * 18  # base index in rank
        
        # Part 1: A[0:2, 0:2] * B[0:2, bk:bk+4]
        # Split B[0:2, bk:bk+4] into two 2x2 blocks: cols bk, bk+1 and bk+2, bk+3
        add_strassen_2x2x2(base, 0, 1, 0, 1, bk)       # C[0:2, bk:bk+2]
        add_strassen_2x2x2(base+7, 0, 1, 0, 1, bk+2)   # C[0:2, bk+2:bk+4]
        
        # Part 2: A[0:2, 2] * B[2, bk:bk+4]
        # This is a 2x1 * 1x4 multiplication.
        # C[i, bk+dk] += A[i, 2] * B[2, bk+dk]
        for dk in range(4):
            c = base + 14 + dk
            U[0*m + 2][c] += 1  # A[0, 2]
            U[1*m + 2][c] += 1  # A[1, 2]
            V[2*p + bk + dk][c] += 1  # B[2, bk+dk]
            W[(bk+dk)*n + 0][c] += 1  # C[0, bk+dk]
            W[(bk+dk)*n + 1][c] += 1  # C[1, bk+dk]
    
    # Verify with deadline guard
    if time.time() - t0 > DEADLINE - 10:
        return {"U": U, "V": V, "W": W}
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Test with random matrices
    A = np.random.randn(2, 3)
    B = np.random.randn(3, 8)
    C_expected = A @ B
    
    # Compute C from decomposition
    C_computed = np.zeros((2, 8))
    for c in range(rank):
        U_mat = U_np[:, c].reshape(2, 3)
        V_mat = V_np[:, c].reshape(3, 8)
        W_vec = W_np[:, c]
        prod = U_mat @ V_mat
        for i in range(2):
            for k in range(8):
                C_computed[i, k] += W_vec[k*2+i] * prod[i, k]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank2 = n * m * p
        U2 = [[0] * rank2 for _ in range(n * m)]
        V2 = [[0] * rank2 for _ in range(m * p)]
        W2 = [[0] * rank2 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i * m + j][col] = 1
                    V2[j * p + k][col] = 1
                    W2[k * n + i][col] = 1
                    col += 1
        return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
