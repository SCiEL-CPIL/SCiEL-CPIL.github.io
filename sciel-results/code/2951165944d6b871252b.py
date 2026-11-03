"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125

    result = _build_decomposition(n, m, p)

    if result is not None:
        U, V, W, rank = result
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def _get_strassen_2x2x2():
    # Standard Strassen 2x2x2 rank-7 decomposition
    # U rows: (i,j) pairs: 00, 01, 10, 11
    # V rows: (j,k) pairs: 00, 01, 10, 11
    # W rows: (k,i) pairs: 00, 01, 10, 11
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a11-a22)(b21-b22)  -- wait, let me use the standard formulation
    # Standard Strassen:
    # m1 = (A11-A22)*(B21-B22)  -> no, let me be careful
    
    # Actually the standard Strassen 7 products are:
    # m1 = (a11-a22)*(b21-b22)
    # m2 = (a11+a12)*b22
    # m3 = a11*(b11-b22)
    # m4 = a22*(b22-b11)
    # m5 = (a11+a22)*b11
    # m6 = (a12-a22)*(b11+b21)
    # m7 = (a11-a12)*(b12+b22)
    
    # C11 = m1 + m3 - m4 + m5
    # C12 = m2 + m4
    # C21 = m1 + m3
    # C22 = m2 - m1 + m5 + m7
    
    # U factorizes A into: for each product index r, U[*, r] gives coefficients for A entries
    # V factorizes B similarly
    # W factorizes C = sum_r U * V * W
    
    # Let me define: A has entries indexed by (i,j) where i in {0,1}, j in {0,1}
    # B has entries indexed by (j,k) where j in {0,1}, k in {0,1}
    # C has entries indexed by (k,i) where k in {0,1}, i in {0,1}
    
    # U[i,j,r] = coefficient of A[i,j] in product r
    # V[j,k,r] = coefficient of B[j,k] in product r
    # W[k,i,r] = coefficient in C[k,i] from product r
    
    # m1 = (A00 - A11)*(B10 - B11)
    U[0,0] = 1;  U[1,0] = -1
    V[0,0] = -1; V[1,0] = 1
    W[0,0] = 1;  W[1,0] = 1;  W[2,0] = 1;  W[3,0] = -1
    
    # m2 = (A00 + A01)*B11
    U[0,1] = 1;  U[1,1] = 1
    V[1,1] = 1
    W[1,1] = 1;  W[3,1] = 1
    
    # m3 = A00*(B00 - B11)
    U[0,2] = 1
    V[0,2] = 1;  V[1,2] = -1
    W[0,2] = 1;  W[1,2] = 1
    
    # m4 = A11*(B11 - B00)
    U[1,3] = 1
    V[0,3] = -1; V[1,3] = 1
    W[0,3] = -1; W[1,3] = 1; W[3,3] = 1
    
    # m5 = (A00 + A11)*B00
    U[0,4] = 1;  U[1,4] = 1
    V[0,4] = 1
    W[0,4] = 1;  W[3,4] = 1
    
    # m6 = (A01 - A11)*(B00 + B10)
    U[0,5] = 1;  U[1,5] = -1
    V[0,5] = 1;  V[1,5] = 1
    W[2,5] = 1
    
    # m7 = (A00 - A01)*(B01 + B11)
    U[0,6] = 1;  U[1,6] = -1
    V[0,6] = 1;  V[1,6] = 1
    W[3,6] = 1
    
    return U, V, W

def _build_decomposition(n, m, p):
    sU, sV, sW = _get_strassen_2x2x2()
    
    # Partition indices
    # i: {0,1} top (2 rows), {2} bottom (1 row)
    # j: {0,1}, {2,3}, {4}  -- 3 blocks of widths 2, 2, 1
    # k: {0,1}, {2,3}, {4,5} -- 3 blocks of widths 2, 2, 2
    
    rank = 84
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Top 2 rows (i in {0,1})
    for j_block in range(3):
        j0 = j_block * 2
        j1 = min(j0 + 2, m)
        j_width = j1 - j0
        
        for k_block in range(3):
            k0 = k_block * 2
            k1 = min(k0 + 2, p)
            k_width = k1 - k0
            
            if j_width == 2 and k_width == 2:
                for c in range(7):
                    # U: rows are i*m + j, for i in {0,1}, j in {j0, j0+1}
                    U[0*5 + j0, col] = sU[0, c]
                    U[0*5 + j0+1, col] = sU[1, c]
                    U[1*5 + j0, col] = sU[2, c]
                    U[1*5 + j0+1, col] = sU[3, c]
                    
                    # V: rows are j*p + k, for j in {j0, j0+1}, k in {k0, k0+1}
                    V[j0*6 + k0, col] = sV[0, c]
                    V[j0*6 + k0+1, col] = sV[1, c]
                    V[(j0+1)*6 + k0, col] = sV[2, c]
                    V[(j0+1)*6 + k0+1, col] = sV[3, c]
                    
                    # W: rows are k*n + i, for k in {k0, k0+1}, i in {0,1}
                    W[k0*3 + 0, col] = sW[0, c]
                    W[k0*3 + 1, col] = sW[1, c]
                    W[(k0+1)*3 + 0, col] = sW[2, c]
                    W[(k0+1)*3 + 1, col] = sW[3, c]
                    col += 1
            elif j_width == 1 and k_width == 2:
                # j0 = 4, single j column
                for i in range(2):
                    for kk in range(2):
                        U[i*5 + j0, col] = 1
                        V[j0*6 + k0 + kk, col] = 1
                        W[(k0 + kk)*3 + i, col] = 1
                        col += 1
            else:
                raise ValueError(f"Unexpected top block: {j_width}x{k_width}")
    
    # Bottom 1 row (i = 2)
    for j_block in range(3):
        j0 = j_block * 2
        j1 = min(j0 + 2, m)
        j_width = j1 - j0
        
        for k_block in range(3):
            k0 = k_block * 2
            k1 = min(k0 + 2, p)
            k_width = k1 - k0
            
            if j_width == 2 and k_width == 2:
                for jj in range(2):
                    for kk in range(2):
                        U[10 + j0 + jj, col] = 1
                        V[(j0 + jj)*6 + k0 + kk, col] = 1
                        W[(k0 + kk)*3 + 2, col] = 1
                        col += 1
            elif j_width == 1 and k_width == 2:
                for kk in range(2):
                    U[10 + j0, col] = 1
                    V[j0*6 + k0 + kk, col] = 1
                    W[(k0 + kk)*3 + 2, col] = 1
                    col += 1
            else:
                raise ValueError(f"Unexpected bottom block: {j_width}x{k_width}")
    
    if col != rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        rank = col
        
    return U, V, W, rank

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

# EVOLVE-BLOCK-END
