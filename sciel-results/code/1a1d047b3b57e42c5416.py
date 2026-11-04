import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _build_strassen_2x2():
    """
    Build Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
    U indexed by (i,k), V by (k,j), W by (j,i)
    T[i,k,j] = sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build Strassen-based rank-49 decomposition for 4x4 x 4x4.
    Uses the 7 big multiplications, each expanded to 7 small Strassen products.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    big_mults = [
        {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
        {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)], 'C': [((0,1), 1), ((1,1), -1)]},
        {'A': [((0,0), 1)], 'B': [((0,1), 1), ((1,1), -1)], 'C': [((1,0), 1), ((1,1), 1)]},
        {'A': [((1,1), 1)], 'B': [((1,0), 1), ((0,0), -1)], 'C': [((0,0), 1), ((0,1), 1)]},
        {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)], 'C': [((0,0), -1), ((1,0), 1)]},
        {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)], 'C': [((1,1), 1)]},
        {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)], 'C': [((0,0), 1)]},
    ]
    
    for big_idx, big in enumerate(big_mults):
        base_r = big_idx * 7
        for small_idx in range(7):
            r = base_r + small_idx
            for (bi, bk), sign_A in big['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2 * bi + i2
                        k = 2 * bk + k2
                        U[i * 4 + k, r] += sign_A * U2[i2 * 2 + k2, small_idx]
            for (bk, bj), sign_B in big['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2 * bk + k2
                        j = 2 * bj + j2
                        V[k * 4 + j, r] += sign_B * V2[k2 * 2 + j2, small_idx]
            for (bj, bi), sign_C in big['C']:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2 * bj + j2
                        i = 2 * bi + i2
                        W[j * 4 + i, r] += sign_C * W2[j2 * 2 + i2, small_idx]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Build the 4x4 x 4x4 Strassen decomposition (rank 49)
    U44, V44, W44 = _build_4x4x4_strassen()
    
    # Build the 4x1 x 4x1 trivial decomposition (rank 16)
    # This multiplies a 4x1 matrix (column vector) by a 4x4 matrix to get 4x1
    # Wait, we need A (4x4) * B_col (4x1) = C_col (4x1)
    # So it's 4x4 times 4x1. Standard multiplication uses 16 terms.
    # U[i,k] * V[k,0] * W[0,i]
    U41 = np.zeros((16, 16))
    V41 = np.zeros((4, 16))
    W41 = np.zeros((4, 16))
    
    for i in range(4):
        for k in range(4):
            r = i * 4 + k
            U41[i * 4 + k, r] = 1
            V41[k * 1 + 0, r] = 1
            W41[0 * 4 + i, r] = 1
    
    # Total rank: 49 (for first 4 cols) + 49 (for next 4 cols) + 16 (for last 1 col) = 114
    total_rank = 49 + 49 + 16
    
    # Allocate full matrices
    # U shape: (n*m, rank) = (16, 114)
    # V shape: (m*p, rank) = (36, 114)
    # W shape: (p*n, rank) = (36, 114)
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Part 1: A * B1 (B1 is 4x4, cols 0-3 of B)
    # U part: same as U44
    U[:, 0:49] = U44
    # V part: V44 placed at cols 0-3 of B (which are cols 0-3 of the 4x9 matrix)
    # V is indexed by (k, j) where j is 0..8 for the 4x9 matrix
    # For the first 4 columns, j=0..3
    for k in range(4):
        for j in range(4):
            V[k * 9 + j, 0:49] = V44[k * 4 + j, :]
    # W part: W44 for output cols 0-3
    for j in range(4):
        for i in range(4):
            W[j * 4 + i, 0:49] = W44[j * 4 + i, :]
    
    # Part 2: A * B2 (B2 is 4x4, cols 4-7 of B)
    # U part: same as U44 (reusing the same A structure)
    U[:, 49:98] = U44
    # V part: V44 placed at cols 4-7 of B
    for k in range(4):
        for j in range(4):
            V[k * 9 + (j + 4), 49:98] = V44[k * 4 + j, :]
    # W part: W44 for output cols 4-7
    for j in range(4):
        for i in range(4):
            W[(j + 4) * 4 + i, 49:98] = W44[j * 4 + i, :]
    
    # Part 3: A * B3 (B3 is 4x1, col 8 of B)
    # U part: same as U41 (standard multiplication)
    U[:, 98:114] = U41
    # V part: V41 placed at col 8 of B
    for k in range(4):
        V[k * 9 + 8, 98:114] = V41[k, :]
    # W part: W41 for output col 8
    for i in range(4):
        W[8 * 4 + i, 98:114] = W41[i, :]
        
    # Verify correctness with a random test
    if time.time() < deadline - 10:
        A = np.random.randn(4, 4)
        B = np.random.randn(4, 9)
        C_expected = A @ B
        
        # Compute C_decomposed
        C_decomposed = np.zeros((4, 9))
        for r in range(total_rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            # u_col is 16x1, v_col is 36x1, w_col is 36x1
            # Reshape u_col to 4x4, v_col to 4x9, w_col to 9x4
            U_r = u_col.reshape(4, 4)
            V_r = v_col.reshape(4, 9)
            W_r = w_col.reshape(9, 4)
            # C contribution: sum over k of A[i,k] * B[k,j]
            # In tensor form: T[i,j,k] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
            # So C[i,j] = sum_k sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
            # Let's compute: C_decomposed += (A * U_r.T) @ (B * V_r.T) * W_r.T ? No.
            # C[i,j] = sum_k A[i,k] * B[k,j]
            # = sum_k sum_r U[i,k,r] * V[k,j,r] * W[j,i,r] * A[i,k] * B[k,j] ... wait
            # The tensor T[i,k,j] = sum_r U[i*4+k, r] * V[k*9+j, r] * W[j*4+i, r]
            # should equal A[i,k] * B[k,j]
            # So C[i,j] = sum_k A[i,k] * B[k,j] = sum_k sum_r U[i*4+k,r] * V[k*9+j,r] * W[j*4+i,r] * A[i,k] * B[k,j] ... no
            # Actually, the identity is: A[i,k] * B[k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
            # So C[i,j] = sum_k A[i,k] * B[k,j] = sum_k sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
            # Let's compute this directly
            for i in range(4):
                for j in range(9):
                    for k in range(4):
                        C_decomposed[i, j] += U[i * 4 + k, r] * V[k * 9 + j, r] * W[j * 4 + i, r]
        
        # Check
        diff = np.max(np.abs(C_decomposed - C_expected))
        if diff > 1e-10:
            print(f"Warning: decomposition error {diff}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
