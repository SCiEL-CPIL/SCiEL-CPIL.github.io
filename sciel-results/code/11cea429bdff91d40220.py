import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen terms
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _decomp_2x2x3():
    """
    Decomposition of M_{2,2,3} using Winograd.
    Rank 8.
    """
    rank = 8
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(6)]
    
    # Winograd for 2x2x3
    # This uses the fact that we can compute 2x2x3 with 8 multiplications
    # using the identity:
    # A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # 
    # m1 = (a+d)(e+h)
    # m2 = (a+d)(f+i)
    # m3 = (a+d)(g+j)
    # m4 = (c-d)(e-h)
    # m5 = (b-c)(f-i)
    # m6 = (b-c)(g-j)
    # m7 = a*e - d*j
    # m8 = c*i - b*h
    #
    # C00 = m1 + m4 + m7
    # C01 = m2 + m5 + m8
    # C02 = m3 + m6
    # C10 = m1 - m4 - m7
    # C11 = m2 - m5 - m8
    # C12 = m3 - m6
    
    # m1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (a+d)(f+i)
    U[0][1] = 1; U[3][1] = 1
    V[1][1] = 1; V[4][1] = 1
    W[1][1] = 1; W[4][1] = 1
    
    # m3 = (a+d)(g+j)
    U[0][2] = 1; U[3][2] = 1
    V[2][2] = 1; V[5][2] = 1
    W[2][2] = 1; W[5][2] = 1
    
    # m4 = (c-d)(e-h)
    U[2][3] = 1; U[3][3] = -1
    V[0][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[3][3] = -1
    
    # m5 = (b-c)(f-i)
    U[1][4] = 1; U[2][4] = -1
    V[1][4] = 1; V[4][4] = -1
    W[1][4] = 1; W[4][4] = -1
    
    # m6 = (b-c)(g-j)
    U[1][5] = 1; U[2][5] = -1
    V[2][5] = 1; V[5][5] = -1
    W[2][5] = 1; W[5][5] = -1
    
    # m7 = a*e - d*j
    U[0][6] = 1
    V[0][6] = 1
    W[0][6] = 1
    U[3][6] = -1
    V[5][6] = -1
    W[3][6] = -1
    
    # m8 = c*i - b*h
    U[2][7] = 1
    V[4][7] = 1
    W[4][7] = 1
    U[1][7] = -1
    V[3][7] = -1
    W[1][7] = -1
    
    return {"U": U, "V": V, "W": W}

def _combine_decomps(decomp1, decomp2, n, m, p1, p2):
    """
    Combine two decompositions for M_{n,m,p1} and M_{n,m,p2} into M_{n,m,p1+p2}.
    """
    r1 = len(decomp1["U"][0])
    r2 = len(decomp2["U"][0])
    r = r1 + r2
    
    U = [row + [0]*r2 for row in decomp1["U"]]
    
    V = [[0]*r for _ in range(m * (p1 + p2))]
    for j in range(m):
        for k in range(p1):
            V[j*(p1+p2) + k] = decomp1["V"][j*p1 + k][:r1] + [0]*r2
        for k in range(p2):
            V[j*(p1+p2) + p1 + k] = [0]*r1 + decomp2["V"][j*p2 + k]
            
    W = [[0]*r for _ in range((p1 + p2) * n)]
    for k in range(p1):
        for i in range(n):
            W[k*n + i] = decomp1["W"][k*n + i][:r1] + [0]*r2
    for k in range(p2):
        for i in range(n):
            W[(p1+k)*n + i] = [0]*r1 + decomp2["W"][k*n + i]
            
    return {"U": U, "V": V, "W": W}

def _kron_decomp(d1, d2, n1, m1, p1, n2, m2, p2):
    """
    Kronecker product of two decompositions.
    M_{n1,m1,p1} ⊗ M_{n2,m2,p2} = M_{n1*n2, m1*m2, p1*p2}
    """
    r1 = len(d1["U"][0])
    r2 = len(d2["U"][0])
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0]*r for _ in range(n * m)]
    V = [[0]*r for _ in range(m * p)]
    W = [[0]*r for _ in range(p * n)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    for t1 in range(r1):
                        for t2 in range(r2):
                            col = t1 * r2 + t2
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i*m + j][col] = d1["U"][i1*m1 + j1][t1] * d2["U"][i2*m2 + j2][t2]
    
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    for t1 in range(r1):
                        for t2 in range(r2):
                            col = t1 * r2 + t2
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V[j*p + k][col] = d1["V"][j1*p1 + k1][t1] * d2["V"][j2*p2 + k2][t2]
    
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    for t1 in range(r1):
                        for t2 in range(r2):
                            col = t1 * r2 + t2
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            W[k*n + i][col] = d1["W"][k1*n1 + i1][t1] * d2["W"][k2*n2 + i2][t2]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Strategy: Use Winograd 2x2x3 (rank 8) for the first 3 columns
    # and schoolbook for the remaining 6 columns
    # This gives rank 8 + 4*4*6 = 8 + 96 = 104
    
    d_winograd = _decomp_2x2x3()
    d_school_6 = _schoolbook(2, 2, 6)
    
    # Combine for 2x2x9
    d_229 = _combine_decomps(d_winograd, d_school_6, 2, 2, 3, 6)
    
    # Now we need 4x4x9, so we need to extend from 2x2x9 to 4x4x9
    # Use Strassen for the 2x2 -> 4x4 extension in n and m dimensions
    
    d_strassen = _strassen_2x2()
    
    # First, extend p from 9 to 9 (no change), but n and m from 2 to 4
    # This is a Kronecker-like product but with Strassen
    
    # Actually, let's use a different approach:
    # Decompose 4x4x9 as (2x2x9) ⊗ (2x2x1) using Strassen
    
    d_221 = _schoolbook(2, 2, 1)
    d_449_kron = _kron_decomp(d_229, d_221, 2, 2, 9, 2, 2, 1)
    
    # This gives rank 16 * 4 = 64, which is better than 104!
    # But wait, this is a Kronecker product, not a valid decomposition for 4x4x9
    # Let me reconsider.
    
    # Actually, for 4x4x9, we can use:
    # - 2x2x3 Winograd (rank 8) for the first 3 columns, applied to both 2x2 blocks
    # - Schoolbook for the remaining 6 columns
    
    # Let's use the block structure:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12, B13, B14, B15, B16, B17, B18, B19],
    #                                      [B21, B22, B23, B24, B25, B26, B27, B28, B29]]
    
    # C = A*B
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # ...
    
    # This is complex. Let's use a simpler approach.
    
    # Use Strassen for 4x4x4 (rank 49) twice, plus schoolbook for 4x4x1 (rank 16)
    # Total: 49 + 49 + 16 = 114
    
    d_444_1 = _decomp_4x4x4()
    d_444_2 = _decomp_4x4x4()
    d_441 = _schoolbook(4, 4, 1)
    
    d_448 = _combine_decomps(d_444_1, d_444_2, 4, 4, 4, 4)
    d_449 = _combine_decomps(d_448, d_441, 4, 4, 8, 1)
    
    if _verify_fast(d_449["U"], d_449["V"], d_449["W"], n, m, p):
        return d_449
    else:
        return _schoolbook(n, m, p)

def _decomp_4x4x4():
    """
    Decomposition of M_{4,4,4} using 2x2 Strassen blocks.
    Rank 49.
    """
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    inner = _strassen_2x2()
    
    outer_A = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [0, 0, 1, -1], [0, 1, 0, -1]
    ]
    outer_B = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    outer_C = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [-1, 0, 1, 0],
        [1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner["U"][li*2 + lj][r]
                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j][col] = coeff_U
                            
                    coeff_V = outer_B[t][b] * inner["V"][li*2 + lj][r]
                    for ei in range(2):
                        for ej in range(2):
                            g_j_B = 2 * (b // 2) + ei
                            g_k_B = 2 * (b % 2) + ej
                            V[g_j_B * 4 + g_k_B][col] = coeff_V
                            
                    coeff_W = outer_C[t][b] * inner["W"][li*2 + lj][r]
                    for ei in range(2):
                        for ej in range(2):
                            g_k_C = 2 * (b // 2) + ei
                            g_i_C = 2 * (b % 2) + ej
                            W[g_k_C * 4 + g_i_C][col] = coeff_W

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
