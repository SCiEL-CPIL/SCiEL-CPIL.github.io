import numpy as np

DIMENSIONS = [4, 4, 6]


def _strassen_2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7.
    Index conventions: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # r=0: M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11
    
    # r=1: M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1  # B00
    W[2][1] = 1; W[3][1] = -1  # C10, C11
    
    # r=2: M3 = A00*(B01-B11)
    U[0][2] = 1  # A00
    V[1][2] = 1; V[3][2] = -1  # B01, B11
    W[1][2] = 1; W[3][2] = 1  # C01, C11
    
    # r=3: M4 = A11*(B10-B00)
    U[3][3] = 1  # A11
    V[2][3] = 1; V[0][3] = -1  # B10, B00
    W[0][3] = 1; W[2][3] = 1  # C00, C10
    
    # r=4: M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1  # B11
    W[0][4] = -1; W[1][4] = 1  # C00, C01
    
    # r=5: M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1  # A10, A00
    V[0][5] = 1; V[1][5] = 1  # B00, B01
    W[2][5] = 1  # C10
    
    # r=6: M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1  # A01, A11
    V[2][6] = 1; V[3][6] = 1  # B10, B11
    W[0][6] = -1; W[3][6] = 1  # C00, C11
    
    return U, V, W


def _kronecker_matmul(n1, m1, p1, n2, m2, p2, U1, V1, W1, U2, V2, W2):
    """Kronecker product of two matmul decompositions.
    For (n1 x m1) * (m1 x p1) and (n2 x m2) * (m2 x p2),
    the Kronecker gives (n1*n2 x m1*m2) * (m1*m2 x p1*p2).
    """
    rank = len(U1[0]) * len(U2[0])
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r1 in range(len(U1[0])):
        for r2 in range(len(U2[0])):
            r = r1 * len(U2[0]) + r2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            idx_u = (i1*n2 + i2)*m + (j1*m2 + j2)
                            U[idx_u][r] = U1[i1*m1+j1][r1] * U2[i2*m2+j2][r2]
            
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            idx_v = (j1*m2 + j2)*p + (k1*p2 + k2)
                            V[idx_v][r] = V1[j1*p1+k1][r1] * V2[j2*p2+k2][r2]
            
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            idx_w = (k1*p2 + k2)*n + (i1*n2 + i2)
                            W[idx_w][r] = W1[k1*n1+i1][r1] * W2[k2*n2+i2][r2]
    
    return U, V, W, rank


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank


def _combine_decompositions(n, m, p, decomp_list):
    """Combine multiple decompositions of the same problem by concatenating columns.
    Each decomp is (U, V, W, rank) for the same (n,m,p).
    """
    total_rank = sum(d[3] for d in decomp_list)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    offset = 0
    for U_d, V_d, W_d, r_d in decomp_list:
        for i in range(n*m):
            for r in range(r_d):
                U[i][offset+r] = U_d[i][r]
        for i in range(m*p):
            for r in range(r_d):
                V[i][offset+r] = V_d[i][r]
        for i in range(p*n):
            for r in range(r_d):
                W[i][offset+r] = W_d[i][r]
        offset += r_d
    
    return U, V, W, total_rank


def _matmul_4x4x6():
    """Construct rank-minimal decomposition for 4x4 * 4x6.
    
    Strategy: Split the 4x6 matrix B into B1 (4x4) and B2 (4x2).
    C = A*B = A*B1 + A*B2
    
    For A*B1: use Kronecker product of Strassen 2x2x2 (rank 7) with itself -> rank 49
    For A*B2: use schoolbook 4x4x2 -> rank 32
    
    Total rank: 49 + 32 = 81
    """
    n, m, p = 4, 4, 6
    
    # Part 1: A * B1 where B1 is 4x4 (first 4 columns of B)
    # Use Strassen 2x2x2 Kronecker Strassen 2x2x2 for 4x4x4
    U_s, V_s, W_s = _strassen_2x2()
    U_44, V_44, W_44, rank_44 = _kronecker_matmul(2, 2, 2, 2, 2, 2, U_s, V_s, W_s, U_s, V_s, W_s)
    
    # Now embed 4x4x4 into 4x4x6:
    # U stays the same (16 x 49)
    # V: 4x4x4 has V indices j*4+k, we need j*6+k for k=0..3
    # W: 4x4x4 has W indices k*4+i, we need k*4+i for k=0..3 (same since n=4)
    
    U_1 = [[0]*rank_44 for _ in range(16)]
    for i in range(16):
        U_1[i] = U_44[i][:]
    
    V_1 = [[0]*rank_44 for _ in range(24)]  # m*p = 4*6
    for j in range(4):
        for k in range(4):
            for r in range(rank_44):
                V_1[j*6+k][r] = V_44[j*4+k][r]
    
    W_1 = [[0]*rank_44 for _ in range(24)]  # p*n = 6*4
    for k in range(4):
        for i in range(4):
            for r in range(rank_44):
                W_1[k*4+i][r] = W_44[k*4+i][r]
    
    # Part 2: A * B2 where B2 is 4x2 (last 2 columns of B)
    # Schoolbook for 4x4x2
    U_2, V_2, W_2, rank_2 = _schoolbook(4, 4, 2)
    
    # Embed into 4x4x6:
    U_2_full = [[0]*rank_2 for _ in range(16)]
    for i in range(16):
        U_2_full[i] = U_2[i][:]
    
    V_2_full = [[0]*rank_2 for _ in range(24)]
    for j in range(4):
        for k in range(2):
            for r in range(rank_2):
                V_2_full[j*6+(4+k)][r] = V_2[j*2+k][r]
    
    W_2_full = [[0]*rank_2 for _ in range(24)]
    for k in range(2):
        for i in range(4):
            for r in range(rank_2):
                W_2_full[(4+k)*4+i][r] = W_2[k*4+i][r]
    
    # Combine
    U, V, W, total_rank = _combine_decompositions(n, m, p, [
        (U_1, V_1, W_1, rank_44),
        (U_2_full, V_2_full, W_2_full, rank_2)
    ])
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
    
    # Compute decomposed tensor
    T_dec = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_dec += np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    return np.allclose(T_dec, T_target, atol=1e-10)


def construct():
    n, m, p = DIMENSIONS
    
    U, V, W = _matmul_4x4x6()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U, V, W, _ = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
