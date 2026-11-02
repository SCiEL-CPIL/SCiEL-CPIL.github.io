"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = start_time + 130.0

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2x2():
        """Strassen's 2x2x2 rank-7 decomposition."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1
        W[3, 0] = 1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        W[3, 1] = -1
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        W[3, 2] = 1
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1
        W[2, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1
        W[1, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def verify(U, V, W, n, m, p):
        """Verify the decomposition."""
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            M_r = np.zeros(n)
            N_r = np.zeros(p)
            for i in range(n):
                for j in range(m):
                    M_r[i] += U[i*m+j, r] * A[i, j]
            for k in range(p):
                for j in range(m):
                    N_r[k] += V[j*p+k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += W[k*n+i, r] * M_r[i] * N_r[k]
        
        return np.allclose(C_decomp, C_expected, atol=1e-10)

    def embed_block(U_block, V_block, W_block, n_sub, m_sub, p_sub, 
                    i_offset, j_offset, k_offset, n, m, p, U_full, V_full, W_full, col_start):
        """Embed a sub-block decomposition into the full matrices."""
        rank_sub = U_block.shape[1]
        for r in range(rank_sub):
            for i in range(n_sub):
                for j in range(m_sub):
                    U_full[(i_offset + i) * m + (j_offset + j), col_start + r] = U_block[i * m_sub + j, r]
            for j in range(m_sub):
                for k in range(p_sub):
                    V_full[(j_offset + j) * p + (k_offset + k), col_start + r] = V_block[j * p_sub + k, r]
            for k in range(p_sub):
                for i in range(n_sub):
                    W_full[(k_offset + k) * n + (i_offset + i), col_start + r] = W_block[k * p_sub + i, r]
        return col_start + rank_sub

    # Strategy: Split 3x5x6 as n=2+1, m=4+1, p=4+2
    # Use Strassen for 2x2x2 sub-blocks within 2x4x4 and 2x4x2
    
    def make_2x4x4():
        """2x4x4 using 4 Strassen 2x2x2 blocks."""
        U = np.zeros((2*4, 28))
        V = np.zeros((4*4, 28))
        W = np.zeros((4*2, 28))
        col = 0
        for jb in range(2):  # m split into 2+2
            for kb in range(2):  # p split into 2+2
                Us, Vs, Ws = strassen_2x2x2()
                for r in range(7):
                    for i in range(2):
                        for j in range(2):
                            U[i*4 + jb*2 + j, col+r] = Us[i*2+j, r]
                    for j in range(2):
                        for k in range(2):
                            V[(jb*2+j)*4 + kb*2 + k, col+r] = Vs[j*2+k, r]
                    for k in range(2):
                        for i in range(2):
                            W[(kb*2+k)*2 + i, col+r] = Ws[k*2+i, r]
                col += 7
        return U, V, W

    def make_2x4x2():
        """2x4x2 using 2 Strassen 2x2x2 blocks."""
        U = np.zeros((2*4, 14))
        V = np.zeros((4*2, 14))
        W = np.zeros((2*2, 14))
        col = 0
        for jb in range(2):  # m split into 2+2
            Us, Vs, Ws = strassen_2x2x2()
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i*4 + jb*2 + j, col+r] = Us[i*2+j, r]
                for j in range(2):
                    for k in range(2):
                        V[(jb*2+j)*2 + k, col+r] = Vs[j*2+k, r]
                for k in range(2):
                    for i in range(2):
                        W[k*2 + i, col+r] = Ws[k*2+i, r]
            col += 7
        return U, V, W

    # Build full 3x5x6
    U_full = np.zeros((15, 100))
    V_full = np.zeros((30, 100))
    W_full = np.zeros((18, 100))
    col = 0
    
    # Block 1: 2x4x4 (i=0:2, j=0:4, k=0:4)
    U1, V1, W1 = make_2x4x4()
    col = embed_block(U1, V1, W1, 2, 4, 4, 0, 0, 0, 3, 5, 6, U_full, V_full, W_full, col)
    
    # Block 2: 2x4x2 (i=0:2, j=0:4, k=4:6)
    U2, V2, W2 = make_2x4x2()
    col = embed_block(U2, V2, W2, 2, 4, 2, 0, 0, 4, 3, 5, 6, U_full, V_full, W_full, col)
    
    # Block 3: 2x1x4 (i=0:2, j=4:5, k=0:4) schoolbook
    U3, V3, W3 = schoolbook(2, 1, 4)
    col = embed_block(U3, V3, W3, 2, 1, 4, 0, 4, 0, 3, 5, 6, U_full, V_full, W_full, col)
    
    # Block 4: 2x1x2 (i=0:2, j=4:5, k=4:6) schoolbook
    U4, V4, W4 = schoolbook(2, 1, 2)
    col = embed_block(U4, V4, W4, 2, 1, 2, 0, 4, 4, 3, 5, 6, U_full, V_full, W_full, col)
    
    # Block 5: 1x4x4 (i=2:3, j=0:4, k=0:4) schoolbook
    U5, V5, W5 = schoolbook(1, 4, 4)
    col = embed_block(U5, V5, W5, 1, 4, 4, 2, 0, 0, 3, 5, 6, U_full, V_full, W_full, col)
    
    # Block 6: 1x4x2 (i=2:3, j=0:4, k=4:6) schoolbook
    U6, V6, W6 = schoolbook(1, 4, 2)
    col = embed_block(U6, V6, W6, 1, 4, 2, 2, 0, 4, 3, 5, 6, U_full, V_full, W_full, col)
    
    # Block 7: 1x1x4 (i=2:3, j=4:5, k=0:4) schoolbook
    U7, V7, W7 = schoolbook(1, 1, 4)
    col = embed_block(U7, V7, W7, 1, 1, 4, 2, 4, 0, 3, 5, 6, U_full, V_full, W_full, col)
    
    # Block 8: 1x1x2 (i=2:3, j=4:5, k=4:6) schoolbook
    U8, V8, W8 = schoolbook(1, 1, 2)
    col = embed_block(U8, V8, W8, 1, 1, 2, 2, 4, 4, 3, 5, 6, U_full, V_full, W_full, col)
    
    U = U_full[:, :col]
    V = V_full[:, :col]
    W = W_full[:, :col]
    
    rank = col
    print(f"Constructed rank: {rank}")
    
    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
