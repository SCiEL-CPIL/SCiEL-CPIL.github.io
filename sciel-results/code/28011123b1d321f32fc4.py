"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the tensor decomposition of 4x5 by 5x6 matrix multiplication.
    Uses block decomposition with Strassen for 2x2x2 sub-blocks to achieve rank 118.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Block structure: n=4=2+2, m=5=2+3, p=6=2+4
    # A = [[A11(2x2), A12(2x3)],
    #      [A21(2x2), A22(2x3)]]
    # B = [[B11(2x2), B12(2x4)],
    #      [B21(3x2), B22(3x4)]]
    #
    # C11 = A11*B11 + A12*B21  (2x2 output)
    #   A11(2x2)*B11(2x2): Strassen rank 7
    #   A12(2x3)*B21(3x2): schoolbook rank 12
    # C12 = A11*B12 + A12*B22  (2x4 output)
    #   A11(2x2)*B12(2x4): schoolbook rank 16
    #   A12(2x3)*B22(3x4): schoolbook rank 24
    # C21 = A21*B11 + A22*B21  (2x2 output)
    #   A21(2x2)*B11(2x2): Strassen rank 7
    #   A22(2x3)*B21(3x2): schoolbook rank 12
    # C22 = A21*B12 + A22*B22  (2x4 output)
    #   A21(2x2)*B12(2x4): schoolbook rank 16
    #   A22(2x3)*B22(3x4): schoolbook rank 24
    # Total: 19 + 40 + 19 + 40 = 118
    
    total_rank = 7 + 12 + 16 + 24 + 7 + 12 + 16 + 24
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    col = 0
    
    def add_schoolbook(U, V, W, col, i_off, j_off, k_off, ni, mj, pk):
        """Add schoolbook for (ni x mj) * (mj x pk) block at offsets."""
        for ii in range(ni):
            for jj in range(mj):
                for kk in range(pk):
                    ig = i_off + ii
                    jg = j_off + jj
                    kg = k_off + kk
                    U[ig*m + jg, col] = 1.0
                    V[jg*p + kg, col] = 1.0
                    W[kg*n + ig, col] = 1.0
                    col += 1
        return col
    
    def add_strassen_222(U, V, W, col, i_off, j_off, k_off):
        """Add Strassen rank-7 decomposition for 2x2 * 2x2 -> 2x2 block.
        
        The 7 rank-1 terms for the 2x2x2 multiplication tensor.
        Indices: i,j in {0,1} for A, j,k in {0,1} for B, k,i in {0,1} for C.
        Global: i_g = i_off + i, j_g = j_off + j, k_g = k_off + k
        U index: i_g * m + j_g
        V index: j_g * p + k_g  
        W index: k_g * n + i_g
        """
        # Strassen 7-term decomposition of 2x2x2 tensor
        # Each term: (u_coeffs for A[2x2], v_coeffs for B[2x2], w_coeffs for C[2x2])
        # u_coeffs[i,j], v_coeffs[j,k], w_coeffs[k,i]
        
        # The 7 terms (u, v, w) where each is a 2x2 matrix:
        # Term 0: u=[[1,0],[1,0]], v=[[1,1],[0,0]], w=[[1,0],[0,0]]  -> M2 part
        # Let me use the standard Strassen decomposition properly.
        
        # Standard Strassen for 2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)(b00)
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)(b11)
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        # c00 = M1+M4-M5+M7
        # c01 = M3+M5
        # c10 = M2+M4
        # c11 = M1-M2+M3+M6
        
        # In tensor form, each Ms contributes to multiple c entries.
        # The rank-1 decomposition has 7 terms where term s has:
        # u_s[i,j] = coefficient of a_ij in the A-linear-combo for Ms
        # v_s[j,k] = coefficient of b_jk in the B-linear-combo for Ms
        # w_s[k,i] = sum over c entries where Ms contributes, with sign
        
        # Let me enumerate:
        # M1: A_combo = a00+a11, B_combo = b00+b11
        #   u1 = [[1,0],[0,1]], v1 = [[1,0],[0,1]]
        #   M1 contributes to: c00(+1), c11(+1)
        #   w1[k,i]: c00 is (k=0,i=0), c11 is (k=1,i=1)
        #   w1 = [[1,0],[0,1]]
        
        # M2: A_combo = a10+a11, B_combo = b00
        #   u2 = [[0,0],[1,1]], v2 = [[1,0],[0,0]]
        #   M2 contributes to: c10(+1), c11(-1)
        #   c10: k=0,i=1; c11: k=1,i=1
        #   w2[k,i]: w2[0,1]=1, w2[1,1]=-1 => w2 = [[0,0],[0,0]]... 
        #   w2 as 2x2: row k, col i. w2[0,1]=1, w2[1,1]=-1
        #   w2 = [[0,1],[0,-1]]
        
        # M3: A_combo = a00, B_combo = b01-b11
        #   u3 = [[1,0],[0,0]], v3 = [[0,1],[0,-1]]
        #   M3 contributes to: c01(+1), c11(+1)
        #   c01: k=1,i=0; c11: k=1,i=1
        #   w3[1,0]=1, w3[1,1]=1 => w3 = [[0,0],[1,1]]
        
        # M4: A_combo = a11, B_combo = b10-b00
        #   u4 = [[0,0],[0,1]], v4 = [[-1,0],[1,0]]
        #   M4 contributes to: c00(+1), c10(+1)
        #   c00: k=0,i=0; c10: k=0,i=1
        #   w4[0,0]=1, w4[0,1]=1 => w4 = [[1,1],[0,0]]
        
        # M5: A_combo = a00+a01, B_combo = b11
        #   u5 = [[1,1],[0,0]], v5 = [[0,0],[0,1]]
        #   M5 contributes to: c00(-1), c01(+1)
        #   c00: k=0,i=0; c01: k=1,i=0
        #   w5[0,0]=-1, w5[1,0]=1 => w5 = [[-1,0],[1,0]]
        
        # M6: A_combo = a10-a00, B_combo = b00+b01
        #   u6 = [[-1,0],[1,0]], v6 = [[1,1],[0,0]]
        #   M6 contributes to: c11(+1)
        #   c11: k=1,i=1
        #   w6[1,1]=1 => w6 = [[0,0],[0,1]]
        
        # M7: A_combo = a01-a11, B_combo = b10+b11
        #   u7 = [[0,1],[0,-1]], v7 = [[0,0],[1,1]]
        #   M7 contributes to: c00(+1)
        #   c00: k=0,i=0
        #   w7[0,0]=1 => w7 = [[1,0],[0,0]]
        
        terms = [
            # (u_2x2, v_2x2, w_2x2)
            ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),      # M1
            ([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,-1]]),     # M2
            ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,0],[1,1]]),     # M3
            ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,1],[0,0]]),     # M4
            ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[1,0]]),     # M5
            ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),     # M6
            ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),     # M7
        ]
        
        for u_mat, v_mat, w_mat in terms:
            for ii in range(2):
                for jj in range(2):
                    if u_mat[ii][jj] != 0:
                        ig = i_off + ii
                        jg = j_off + jj
                        U[ig*m + jg, col] += u_mat[ii][jj]
            for jj in range(2):
                for kk in range(2):
                    if v_mat[jj][kk] != 0:
                        jg = j_off + jj
                        kg = k_off + kk
                        V[jg*p + kg, col] += v_mat[jj][kk]
            for kk in range(2):
                for ii in range(2):
                    if w_mat[kk][ii] != 0:
                        kg = k_off + kk
                        ig = i_off + ii
                        W[kg*n + ig, col] += w_mat[kk][ii]
            col += 1
        return col
    
    # Block offsets:
    # A11: i_off=0, j_off=0, 2x2
    # A12: i_off=0, j_off=2, 2x3
    # A21: i_off=2, j_off=0, 2x2
    # A22: i_off=2, j_off=2, 2x3
    # B11: j_off=0, k_off=0, 2x2
    # B12: j_off=0, k_off=2, 2x4
    # B21: j_off=2, k_off=0, 3x2
    # B22: j_off=2, k_off=2, 3x4
    
    # C11 = A11*B11 + A12*B21 (2x2 output)
    # A11*B11: Strassen 2x2x2
    col = add_strassen_222(U, V, W, col, i_off=0, j_off=0, k_off=0)
    # A12*B21: schoolbook (2x3)*(3x2) -> 2x2
    col = add_schoolbook(U, V, W, col, i_off=0, j_off=2, k_off=0, ni=2, mj=3, pk=2)
    
    # C12 = A11*B12 + A12*B22 (2x4 output)
    # A11*B12: schoolbook (2x2)*(2x4) -> 2x4
    col = add_schoolbook(U, V, W, col, i_off=0, j_off=0, k_off=2, ni=2, mj=2, pk=4)
    # A12*B22: schoolbook (2x3)*(3x4) -> 2x4
    col = add_schoolbook(U, V, W, col, i_off=0, j_off=2, k_off=2, ni=2, mj=3, pk=4)
    
    # C21 = A21*B11 + A22*B21 (2x2 output)
    # A21*B11: Strassen 2x2x2
    col = add_strassen_222(U, V, W, col, i_off=2, j_off=0, k_off=0)
    # A22*B21: schoolbook (2x3)*(3x2) -> 2x2
    col = add_schoolbook(U, V, W, col, i_off=2, j_off=2, k_off=0, ni=2, mj=3, pk=2)
    
    # C22 = A21*B12 + A22*B22 (2x4 output)
    # A21*B12: schoolbook (2x2)*(2x4) -> 2x4
    col = add_schoolbook(U, V, W, col, i_off=2, j_off=0, k_off=2, ni=2, mj=2, pk=4)
    # A22*B22: schoolbook (2x3)*(3x4) -> 2x4
    col = add_schoolbook(U, V, W, col, i_off=2, j_off=2, k_off=2, ni=2, mj=3, pk=4)
    
    # Verify col matches total_rank
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    # Verify the decomposition is correct
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For the multiplication tensor, T[i,j,k] = 1 for all valid i,j,k
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(total_rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)
    
    if error > 1e-6:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = np.zeros((nm, rank_school), dtype=float)
        V_school = np.zeros((mp, rank_school), dtype=float)
        W_school = np.zeros((pn, rank_school), dtype=float)
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col2] = 1.0
                    V_school[j*p + k, col2] = 1.0
                    W_school[k*n + i, col2] = 1.0
                    col2 += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Convert to lists
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
