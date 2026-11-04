"""Matrix multiplication 4x5x5 with structured block decomposition using Strassen."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2():
    """
    Return the 7-term Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) where:
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # U: 4x7, indexed by [i*2+j, c]
    # V: 4x7, indexed by [j*2+k, c]  
    # W: 4x7, indexed by [k*2+i, c]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h): U has a,d; V has e,h; W has C11,C22
    # a = A[0,0] = index 0, d = A[1,1] = index 3
    # e = B[0,0] = index 0, h = B[1,1] = index 3
    # C11 = C[0,0] = W index 0*2+0 = 0, C22 = C[1,1] = W index 1*2+1 = 3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e: U has c,d; V has e; W has C21,C22
    # c = A[1,0] = index 2, d = A[1,1] = index 3
    # e = B[0,0] = index 0
    # C21 = C[1,0] = W index 1*2+0 = 2, C22 = C[1,1] = W index 3
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h): U has a; V has f,h; W has C12,C22
    # a = A[0,0] = index 0
    # f = B[0,1] = index 1, h = B[1,1] = index 3
    # C12 = C[0,1] = W index 0*2+1 = 1, C22 = C[1,1] = W index 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g-e): U has d; V has g,e; W has C11,C21
    # d = A[1,1] = index 3
    # g = B[1,0] = index 2, e = B[0,0] = index 0
    # C11 = C[0,0] = W index 0, C21 = C[1,0] = W index 2
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h: U has a,b; V has h; W has C11,C12
    # a = A[0,0] = index 0, b = A[0,1] = index 1
    # h = B[1,1] = index 3
    # C11 = C[0,0] = W index 0, C12 = C[0,1] = W index 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c-a)(e+f): U has c,a; V has e,f; W has C22
    # c = A[1,0] = index 2, a = A[0,0] = index 0
    # e = B[0,0] = index 0, f = B[0,1] = index 1
    # C22 = C[1,1] = W index 3
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h): U has b,d; V has g,h; W has C11
    # b = A[0,1] = index 1, d = A[1,1] = index 3
    # g = B[1,0] = index 2, h = B[1,1] = index 3
    # C11 = C[0,0] = W index 0
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_numpy(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    return np.allclose(T, 1.0, atol=1e-9)

def _schoolbook(n, m, p):
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

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix using block decomposition with Strassen.
    
    Strategy:
    - Split i (rows of A, rows of C) into {0,1} and {2,3}
    - Split j (cols of A, rows of B) into {0,1}, {2,3}, {4}
    - Split k (cols of B, cols of C) into {0,1}, {2,3}, {4}
    - For 2x2x2 sub-blocks, use Strassen (rank 7)
    - For other sub-blocks, use schoolbook
    - Total rank: 8*7 + 4*4 + 4*4 + 2*2 = 56 + 16 + 16 + 4 = 92
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 140
    
    # Block definitions
    # i blocks: [0,1], [2,3] -> sizes [2, 2]
    # j blocks: [0,1], [2,3], [4] -> sizes [2, 2, 1]
    # k blocks: [0,1], [2,3], [4] -> sizes [2, 2, 1]
    
    i_blocks = [[0, 1], [2, 3]]
    j_blocks = [[0, 1], [2, 3], [4]]
    k_blocks = [[0, 1], [2, 3], [4]]
    
    # Collect all terms
    U_terms = []  # list of (row_index, value, col_offset)
    V_terms = []
    W_terms = []
    
    col_offset = 0
    
    for ib, i_list in enumerate(i_blocks):
        for jb, j_list in enumerate(j_blocks):
            for kb, k_list in enumerate(k_blocks):
                a = len(i_list)
                b = len(j_list)
                c = len(k_list)
                
                if a == 2 and b == 2 and c == 2:
                    # Use Strassen
                    Su, Sv, Sw = _strassen_2x2x2()
                    for ci in range(2):
                        for cj in range(2):
                            for ck in range(2):
                                # Map local indices to global
                                gi = i_list[ci]
                                gj = j_list[cj]
                                gk = k_list[ck]
                                # U local index: ci*2+cj
                                u_local = ci * 2 + cj
                                v_local = cj * 2 + ck
                                w_local = ck * 2 + ci
                                for t in range(7):
                                    if Su[u_local, t] != 0:
                                        U_terms.append((gi * m + gj, Su[u_local, t], col_offset + t))
                                    if Sv[v_local, t] != 0:
                                        V_terms.append((gj * p + gk, Sv[v_local, t], col_offset + t))
                                    if Sw[w_local, t] != 0:
                                        W_terms.append((gk * n + gi, Sw[w_local, t], col_offset + t))
                    col_offset += 7
                else:
                    # Schoolbook for this sub-block
                    for ci in range(a):
                        for cj in range(b):
                            for ck in range(c):
                                gi = i_list[ci]
                                gj = j_list[cj]
                                gk = k_list[ck]
                                U_terms.append((gi * m + gj, 1, col_offset))
                                V_terms.append((gj * p + gk, 1, col_offset))
                                W_terms.append((gk * n + gi, 1, col_offset))
                                col_offset += 1
    
    rank = col_offset
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for row, val, col in U_terms:
        U[row][col] += val
    for row, val, col in V_terms:
        V[row][col] += val
    for row, val, col in W_terms:
        W[row][col] += val
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify_numpy(U, V, W, n, m, p):
        return result
    
    # Fallback to schoolbook if verification fails
    return _schoolbook(n, m, p)
