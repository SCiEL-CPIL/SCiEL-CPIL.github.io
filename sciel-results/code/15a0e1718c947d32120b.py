"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a rank-54 exact tensor decomposition for 2x5 by 5x6 matrix multiplication.
    
    Split m=5 into [2, 2, 1].
    Block 1: A1(2,2) @ B1(2,6) -> 3 Strassen 2x2x2 sub-blocks, rank 21
    Block 2: A2(2,2) @ B2(2,6) -> 3 Strassen 2x2x2 sub-blocks, rank 21
    Block 3: A3(2,1) @ B3(1,6) -> 12 scalar multiplications
    Total rank = 54.
    """
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    total_rank = 54
    U_mat = np.zeros((n * m, total_rank))  # (10, 54)
    V_mat = np.zeros((m * p, total_rank))  # (30, 54)
    W_mat = np.zeros((p * n, total_rank))  # (12, 54)
    
    col = 0
    
    # Strassen 2x2x2 decomposition.
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = [[c00, c01], [c10, c11]]
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b11)
    # m4 = a11(b01-b10)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # For a 2x2x2 block, the 7 rank-1 terms (u, v, w) where:
    # u is coefficients for (a00, a01, a10, a11)
    # v is coefficients for (b00, b01, b10, b11)
    # w is coefficients for (c00, c01, c10, c11)
    strassen_terms = [
        # m1: (a00+a11)(b00+b11) -> contributes to c00, c11
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        # m2: (a10+a11)b00 -> contributes to c10, c11 (with -1 for c11)
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),
        # m3: a00(b10-b11) -> contributes to c01, c11
        (np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1]), np.array([0, 1, 0, 1])),
        # m4: a11(b01-b10) -> contributes to c00, c10
        (np.array([0, 0, 0, 1]), np.array([0, 1, -1, 0]), np.array([1, 0, 1, 0])),
        # m5: (a00+a01)b11 -> contributes to c00 (with -1), c01
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
        # m6: (a10-a00)(b00+b10) -> contributes to c11
        (np.array([-1, 0, 1, 0]), np.array([1, 0, 1, 0]), np.array([0, 0, 0, 1])),
        # m7: (a01-a11)(b01+b11) -> contributes to c00
        (np.array([0, 1, 0, -1]), np.array([0, 1, 0, 1]), np.array([1, 0, 0, 0])),
    ]
    
    def add_strassen_block(j_start, col_start):
        """Add 21 rank-1 terms for a 2x2x6 block (3 Strassen 2x2x2 sub-blocks).
        
        j_start: starting index in the middle dimension (0 or 2)
        col_start: starting column index
        Returns: new column index
        """
        col = col_start
        
        # U row indices for A block:
        # A00: i=0, j=j_start -> U row 0*m + j_start
        # A01: i=0, j=j_start+1 -> U row 0*m + j_start + 1
        # A10: i=1, j=j_start -> U row 1*m + j_start
        # A11: i=1, j=j_start+1 -> U row 1*m + j_start + 1
        u_rows = [0 * m + j_start, 0 * m + j_start + 1, 1 * m + j_start, 1 * m + j_start + 1]
        
        # For each of 3 sub-blocks (k groups of 2)
        for k_sub in range(3):
            k0 = k_sub * 2  # starting k index
            
            # V row indices for B block (2x2 sub-block at k0, k0+1):
            # B00: j=j_start, k=k0 -> V row j_start*p + k0
            # B01: j=j_start, k=k0+1 -> V row j_start*p + k0 + 1
            # B10: j=j_start+1, k=k0 -> V row (j_start+1)*p + k0
            # B11: j=j_start+1, k=k0+1 -> V row (j_start+1)*p + k0 + 1
            v_rows = [j_start * p + k0, j_start * p + k0 + 1, (j_start + 1) * p + k0, (j_start + 1) * p + k0 + 1]
            
            # W row indices for C block (2x2 sub-block at k0, k0+1):
            # C00: i=0, k=k0 -> W row k0*n + 0
            # C01: i=0, k=k0+1 -> W row (k0+1)*n + 0
            # C10: i=1, k=k0 -> W row k0*n + 1
            # C11: i=1, k=k0+1 -> W row (k0+1)*n + 1
            w_rows = [k0 * n + 0, (k0 + 1) * n + 0, k0 * n + 1, (k0 + 1) * n + 1]
            
            # Add 7 Strassen terms
            for u_coeff, v_coeff, w_coeff in strassen_terms:
                U_mat[u_rows[0], col] += u_coeff[0]
                U_mat[u_rows[1], col] += u_coeff[1]
                U_mat[u_rows[2], col] += u_coeff[2]
                U_mat[u_rows[3], col] += u_coeff[3]
                
                V_mat[v_rows[0], col] += v_coeff[0]
                V_mat[v_rows[1], col] += v_coeff[1]
                V_mat[v_rows[2], col] += v_coeff[2]
                V_mat[v_rows[3], col] += v_coeff[3]
                
                W_mat[w_rows[0], col] += w_coeff[0]
                W_mat[w_rows[1], col] += w_coeff[1]
                W_mat[w_rows[2], col] += w_coeff[2]
                W_mat[w_rows[3], col] += w_coeff[3]
                
                col += 1
        
        return col
    
    # Block 1: j_start = 0
    col = add_strassen_block(0, col)
    
    # Block 2: j_start = 2
    col = add_strassen_block(2, col)
    
    # Block 3: A3(2,1) @ B3(1,6) -> 12 scalar multiplications
    # A3[i, 4] * B3[4, k] -> C[i, k]
    # U row for A3[i,4]: i*m + 4 = i*5 + 4
    # V row for B3[4,k]: 4*p + k = 4*6 + k = 24 + k
    # W row for C[i,k]: k*n + i = k*2 + i
    for i in range(n):
        for k in range(p):
            U_mat[i * m + 4, col] = 1
            V_mat[4 * p + k, col] = 1
            W_mat[k * n + i, col] = 1
            col += 1
    
    # Verify correctness
    # T[i, j, k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i, j, k] = 1 (all entries are 1)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to trivial rank-60 decomposition if verification fails
        rank = n * m * p
        U_final = np.zeros((n * m, rank))
        V_final = np.zeros((m * p, rank))
        W_final = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_final[i * m + j, col] = 1
                    V_final[j * p + k, col] = 1
                    W_final[k * n + i, col] = 1
                    col += 1
        
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
# EVOLVE-BLOCK-END
