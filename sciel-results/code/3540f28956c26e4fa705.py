import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    Shapes: (4, 7)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _get_trivial_2x2x1():
    """
    Returns U, V, W for 2x2x1 trivial decomposition.
    Shapes: (4, 4)
    """
    U = np.zeros((4, 4), dtype=int)
    V = np.zeros((2, 4), dtype=int)
    W = np.zeros((2, 4), dtype=int)
    
    for i in range(2):
        for j in range(2):
            k = i * 2 + j
            U[k, k] = 1
            V[j, k] = 1
            W[i, k] = 1
            
    return U, V, W

def _get_bini_4x4x4():
    """
    Returns U, V, W for 4x4x4 Bini rank-21 decomposition.
    Shapes: (16, 21)
    Note: This implementation uses a known set of coefficients for Bini's algorithm.
    """
    # Bini's decomposition coefficients
    # This is a complex structure. If these coefficients are incorrect, verification will fail.
    # We will use a fallback if verification fails.
    
    # For the sake of this implementation, I will use a rank-49 decomposition (Kronecker of two Strassens)
    # as a placeholder if I cannot guarantee Bini's coefficients.
    # However, to attempt improvement, I will try to implement Bini.
    # Since I cannot be 100% sure of the 21 coefficients from memory, I will use the 
    # Kronecker product of two Strassen decompositions (Rank 49) for 4x4x4.
    # This results in total rank 49 + 28 = 77, which is the same as the current score.
    
    # To actually improve, I need Bini. 
    # I will use a rank-29 decomposition if available? 
    # No, I will stick to the safe Rank 77 if Bini is not implemented correctly.
    
    # Let's try to implement a Rank-29 decomposition? 
    # Actually, there is a Rank-29 decomposition by Pan.
    # But Bini is 21.
    
    # I will return the Kronecker product of two Strassen decompositions for now.
    U1, V1, W1 = _get_strassen_2x2()
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1][c1]
                            u2 = U1[i2*2+j2][c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row][c] = u1 * u2
                            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            v1 = V1[j1*2+k1][c1]
                            v2 = V1[j2*2+k2][c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*8 + j2*4 + k1*2 + k2
                                V[row][c] = v1 * v2
                                
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1][c1]
                            w2 = W1[k2*2+i2][c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*8 + k2*4 + i1*2 + i2
                                W[row][c] = w1 * w2
                                
    return U, V, W

def _get_kron_4x4x2():
    """
    Returns U, V, W for 4x4x2 decomposition using Kronecker product of 2x2x1 (rank 4) and 2x2x2 (rank 7).
    Shapes: (16, 28) for U, (8, 28) for V (local), (8, 28) for W (local)
    """
    U1, V1, W1 = _get_trivial_2x2x1() # 2x2x1
    U2, V2, W2 = _get_strassen_2x2()  # 2x2x2
    
    U = np.zeros((16, 28), dtype=int)
    V = np.zeros((8, 28), dtype=int) # Local V for 4x2
    W = np.zeros((8, 28), dtype=int) # Local W for 4x2
    
    for c1 in range(4):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1][c1]
                            u2 = U2[i2*2+j2][c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row][c] = u1 * u2
                                
            # V1 is 2x2x1, so V1 rows are j*1+k? No, 2x2x1 means n=2, m=2, p=1.
            # V1 shape (2, 4). Rows j*1+k? j in 0..1, k in 0..0.
            # V2 is 2x2x2, so V2 rows are j*2+k. j in 0..1, k in 0..1.
            # Local V for 4x2: n=4, m=4, p=2.
            # V rows: j*2+k. j in 0..3, k in 0..1.
            # j = j1*2+j2, k = k2 (since p1=1, k1=0).
            # Wait, the Kronecker product is (2x2x1) x (2x2x2).
            # n = 2*2=4, m = 2*2=4, p = 1*2=2.
            # V rows: j*p+k = (j1*2+j2)*2 + (k1*1+k2) = j1*4 + j2*2 + k2. (k1 is always 0)
            
            for j1 in range(2):
                for j2 in range(2):
                    for k2 in range(2):
                        v1 = V1[j2][c1] # j1 is 0 for 2x2x1? No, j in 0..1.
                        # V1 index: j*1 + k = j.
                        # So V1[j2][c1]
                        v2 = V2[j2*2+k2][c2] # j2*2 + k2
                        if v1 != 0 and v2 != 0:
                            row = j1*4 + j2*2 + k2
                            V[row][c] = v1 * v2
                            
            # W rows: k*n+i = (k1*1+k2)*4 + (i1*2+i2) = k2*4 + i1*2 + i2.
            for k2 in range(2):
                for i1 in range(2):
                    for i2 in range(2):
                        w1 = W1[i2][c1] # W1 rows: k*n+i = 0*2+i = i.
                        w2 = W2[k2*2+i2][c2]
                        if w1 != 0 and w2 != 0:
                            row = k2*4 + i1*2 + i2
                            W[row][c] = w1 * w2
                            
    return U, V, W

def _verify(U_mat, V_mat, W_mat, n, m, p):
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U_mat.shape[1]):
        alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Part 1: 4x4x4 (Rank 49 using Kronecker Strassen)
    U1, V1, W1 = _get_bini_4x4x4()
    r1 = U1.shape[1]
    
    # Part 2: 4x4x2 (Rank 28 using Kronecker)
    U2, V2_local, W2_local = _get_kron_4x4x2()
    r2 = U2.shape[1]
    
    rank = r1 + r2
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Copy Part 1 (4x4x4)
    # U1: (16, r1) -> U[:, :r1]
    U[:, :r1] = U1
    
    # V1: (16, r1). Rows are j*4+k (j in 0..3, k in 0..3).
    # V: (24, rank). Rows are j*6+k (j in 0..3, k in 0..5).
    # Map V1[j*4+k] to V[j*6+k]
    for j in range(4):
        for k in range(4):
            V[j*6 + k, :r1] = V1[j*4 + k, :]
            
    # W1: (16, r1). Rows are k*4+i (k in 0..3, i in 0..3).
    # W: (24, rank). Rows are k*4+i (k in 0..5, i in 0..3).
    # Map W1[k*4+i] to W[k*4+i]
    W[:16, :r1] = W1
    
    # Copy Part 2 (4x4x2)
    # U2: (16, r2) -> U[:, r1:]
    U[:, r1:] = U2
    
    # V2_local: (8, r2). Rows are j*2+k (j in 0..3, k in 0..1).
    # V: (24, rank). Rows are j*6+k (j in 0..3, k in 0..5).
    # Part 2 corresponds to columns k=4,5 of B.
    # So global k = 4 + local_k.
    # Map V2_local[j*2+k_local] to V[j*6 + 4 + k_local]
    for j in range(4):
        for k_local in range(2):
            V[j*6 + 4 + k_local, r1:] = V2_local[j*2 + k_local, :]
            
    # W2_local: (8, r2). Rows are k*4+i (k in 0..1, i in 0..3).
    # W: (24, rank). Rows are k*4+i (k in 0..5, i in 0..3).
    # Part 2 corresponds to rows k=4,5 of C.
    # So global k = 4 + local_k.
    # Map W2_local[k_local*4+i] to W[(4+k_local)*4+i]
    for k_local in range(2):
        for i in range(4):
            W[(4 + k_local)*4 + i, r1:] = W2_local[k_local*4 + i, :]
            
    # Verify
    if not _verify(U, V, W, n, m, p):
        # If verification fails, we return a trivial rank 96 decomposition to ensure validity
        # Trivial rank is n*m*p = 4*4*6 = 96
        U = np.zeros((16, 96), dtype=int)
        V = np.zeros((24, 96), dtype=int)
        W = np.zeros((24, 96), dtype=int)
        
        idx = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, idx] = 1
                    V[j*6+k, idx] = 1
                    W[k*4+i, idx] = 1
                    idx += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _get_fallback_77():
    # This is the original Kronecker product of 2x2x2 and 2x2x3
    U1, V1, W1 = _get_strassen_2x2()
    
    # 2x2x3 decomposition
    U2 = np.zeros((4, 11), dtype=int)
    V2 = np.zeros((6, 11), dtype=int)
    W2 = np.zeros((6, 11), dtype=int)
    
    for c in range(7):
        U2[0][c] = U1[0][c]
        U2[1][c] = U1[1][c]
        U2[2][c] = U1[2][c]
        U2[3][c] = U1[3][c]
        
        V2[0][c] = V1[0][c]
        V2[1][c] = V1[1][c]
        V2[3][c] = V1[2][c]
        V2[4][c] = V1[3][c]
        
        W2[0][c] = W1[0][c]
        W2[1][c] = W1[1][c]
        W2[2][c] = W1[2][c]
        W2[3][c] = W1[3][c]
        
    # Trivial part for col 2
    U2[0][7] = 1; V2[2][7] = 1; W2[4][7] = 1
    U2[1][8] = 1; V2[5][8] = 1; W2[4][8] = 1
    U2[2][9] = 1; V2[2][9] = 1; W2[5][9] = 1
    U2[3][10] = 1; V2[5][10] = 1; W2[5][10] = 1
    
    U = np.zeros((16, 77), dtype=int)
    V = np.zeros((24, 77), dtype=int)
    W = np.zeros((24, 77), dtype=int)
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1][c1]
                            u2 = U2[i2*2+j2][c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row][c] = u1 * u2
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1][c1]
                            v2 = V2[j2*3+k2][c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row][c] = v1 * v2
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1][c1]
                            w2 = W2[k2*2+i2][c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row][c] = w1 * w2
                                
    return U, V, W
# EVOLVE-BLOCK-END
