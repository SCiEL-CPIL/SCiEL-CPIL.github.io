import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build a rank-44 decomposition using block structure.
    # A is 2x5, B is 5x5.
    # Split A into A1 (2x2), A2 (2x2), A3 (2x1).
    # Split B into B1 (2x5), B2 (2x5), B3 (1x5).
    # C = A1 B1 + A2 B2 + A3 B3.
    
    # We need to map the global indices to local indices for each block.
    # Global U index: i*m + j
    # Global V index: j*p + k
    # Global W index: k*n + i
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block decomposition
    # local_i, local_j, local_k are indices within the block
    # global_i, global_j, global_k are indices in the full matrix
    
    def add_block(u_coeffs, v_coeffs, w_coeffs, i_offset, j_offset, k_offset):
        nonlocal col
        for r in range(len(u_coeffs)):
            for li in range(2): # n=2
                for lj in range(2): # m_block
                    u_val = u_coeffs[r][li][lj]
                    if u_val != 0:
                        gi = i_offset + li
                        gj = j_offset + lj
                        U[gi * m + gj][col + r] += u_val
            for lj in range(2): # m_block
                for lk in range(5): # p_block
                    v_val = v_coeffs[r][lj][lk]
                    if v_val != 0:
                        gj = j_offset + lj
                        gk = k_offset + lk
                        V[gj * p + gk][col + r] += v_val
            for lk in range(5): # p_block
                for li in range(2): # n_block
                    w_val = w_coeffs[r][li][lk]
                    if w_val != 0:
                        gk = k_offset + lk
                        gi = i_offset + li
                        W[gk * n + gi][col + r] += w_val
        col += len(u_coeffs)
        
    # This helper is too slow/complex for arbitrary blocks. 
    # I will hardcode the 44-rank construction.
    
    # Reset
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: A1 (rows 0,1; cols 0,1) * B1 (rows 0,1; cols 0,4)
    # This is M_{2,2,5}. Split B1 into B11 (2x2), B12 (2x2), B13 (2x1).
    # B11: cols 0,1. B12: cols 2,3. B13: col 4.
    
    # Sub-block 1.1: M_{2,2,2} (Strassen)
    # A11 (2x2), B11 (2x2)
    # Indices: i in 0,1; j in 0,1; k in 0,1
    # Strassen 7 mults
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for U (A), V (B), W (C)
    # U is 2x2, V is 2x2, W is 2x2 (local)
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # 7 columns
        for r in range(7):
            col_idx = col + r
            # U
            # M1
            if r == 0:
                U[i_off*m + j_off][col_idx] += 1
                U[i_off*m + j_off+1][col_idx] += 1
                U[(i_off+1)*m + j_off][col_idx] += 1
                U[(i_off+1)*m + j_off+1][col_idx] += 1
                V[j_off*p + k_off][col_idx] += 1
                V[(j_off+1)*p + k_off][col_idx] += 1
                V[j_off*p + (k_off+1)][col_idx] += 1
                V[(j_off+1)*p + (k_off+1)][col_idx] += 1
                W[k_off*n + i_off][col_idx] += 1
                W[(k_off+1)*n + i_off][col_idx] += 1
                W[k_off*n + (i_off+1)][col_idx] += 1
                W[(k_off+1)*n + (i_off+1)][col_idx] += 1
            # M2
            elif r == 1:
                U[(i_off+1)*m + j_off][col_idx] += 1
                U[(i_off+1)*m + j_off+1][col_idx] += 1
                V[j_off*p + k_off][col_idx] += 1
                W[k_off*n + (i_off+1)][col_idx] += 1
                W[(k_off+1)*n + (i_off+1)][col_idx] += 1
            # M3
            elif r == 2:
                U[i_off*m + j_off][col_idx] += 1
                V[(j_off+1)*p + k_off][col_idx] += 1
                V[(j_off+1)*p + (k_off+1)][col_idx] -= 1
                W[k_off*n + i_off][col_idx] += 1
                W[(k_off+1)*n + i_off][col_idx] -= 1
            # M4
            elif r == 3:
                U[(i_off+1)*m + j_off+1][col_idx] += 1
                V[j_off*p + (k_off+1)][col_idx] += 1
                V[j_off*p + k_off][col_idx] -= 1
                W[(k_off+1)*n + (i_off+1)][col_idx] += 1
                W[k_off*n + (i_off+1)][col_idx] -= 1
            # M5
            elif r == 4:
                U[i_off*m + j_off][col_idx] += 1
                U[i_off*m + (j_off+1)][col_idx] += 1
                V[(j_off+1)*p + (k_off+1)][col_idx] += 1
                W[(k_off+1)*n + i_off][col_idx] += 1
                W[(k_off+1)*n + (i_off+1)][col_idx] += 1
            # M6
            elif r == 5:
                U[(i_off+1)*m + j_off][col_idx] += 1
                U[i_off*m + j_off][col_idx] -= 1
                V[j_off*p + k_off][col_idx] += 1
                V[j_off*p + (k_off+1)][col_idx] += 1
                W[k_off*n + (i_off+1)][col_idx] += 1
                W[(k_off+1)*n + (i_off+1)][col_idx] += 1
            # M7
            elif r == 6:
                U[i_off*m + (j_off+1)][col_idx] += 1
                U[(i_off+1)*m + (j_off+1)][col_idx] -= 1
                V[(j_off+1)*p + k_off][col_idx] += 1
                V[(j_off+1)*p + (k_off+1)][col_idx] += 1
                W[k_off*n + i_off][col_idx] += 1
                W[(k_off+1)*n + i_off][col_idx] += 1
        col += 7

    # Sub-block 1.2: M_{2,2,2} (Strassen) for B12 (cols 2,3)
    add_strassen(0, 0, 2) # A1 (cols 0,1), B12 (cols 2,3) -> Wait, j_off is for A cols, k_off is for B cols.
    # A1 is cols 0,1. So j_off=0.
    # B12 is cols 2,3. So k_off=2.
    # But I already called add_strassen(0,0,0) for B11?
    # I need to call it for B11 (k=0,1) and B12 (k=2,3).
    
    # Reset col and matrices to be safe
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: A1 (j=0,1) * B1 (j=0,1)
    # B1 cols: 0,1 (Strassen), 2,3 (Strassen), 4 (Winograd 2x2x1)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    
    # Sub-block 1.3: M_{2,2,1} (Winograd) for B13 (col 4)
    # A1 (2x2), B13 (2x1)
    # C13 = A1 B13
    # Winograd rank 3
    # M1 = A00 B00
    # M2 = A11 B11
    # M3 = (A00+A11)(B00+B11)
    # C0 = M1 + M2
    # C1 = M3 - M1 - M2
    # Note: B13 is 2x1, so B00 and B11 are B[0,4] and B[1,4].
    # C13 is 2x1, so C0 and C1 are C[0,4] and C[1,4].
    
    def add_winograd_221(i_off, j_off, k_off):
        nonlocal col
        for r in range(3):
            col_idx = col + r
            if r == 0:
                U[i_off*m + j_off][col_idx] += 1
                V[j_off*p + k_off][col_idx] += 1
                W[k_off*n + i_off][col_idx] += 1
            elif r == 1:
                U[(i_off+1)*m + (j_off+1)][col_idx] += 1
                V[(j_off+1)*p + k_off][col_idx] += 1
                W[k_off*n + (i_off+1)][col_idx] += 1
            elif r == 2:
                U[i_off*m + j_off][col_idx] += 1
                U[(i_off+1)*m + (j_off+1)][col_idx] += 1
                V[j_off*p + k_off][col_idx] += 1
                V[(j_off+1)*p + k_off][col_idx] += 1
                W[k_off*n + i_off][col_idx] -= 1
                W[k_off*n + (i_off+1)][col_idx] += 1
        col += 3

    add_winograd_221(0, 0, 4)
    
    # Block 2: A2 (j=2,3) * B2 (j=2,3)
    # B2 cols: 0,1 (Strassen), 2,3 (Strassen), 4 (Winograd)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_winograd_221(0, 2, 4)
    
    # Block 3: A3 (j=4) * B3 (j=4)
    # A3 is 2x1, B3 is 1x5.
    # C3 = A3 B3
    # C[i,k] = A[i,4] B[4,k]
    # This is M_{2,1,5}. Rank 10.
    # For each k, we have 2 multiplications?
    # C[0,k] = A[0,4] B[4,k]
    # C[1,k] = A[1,4] B[4,k]
    # We can use 2 mults per k. Total 10.
    for k in range(5):
        # M1 = A[0,4] B[4,k]
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        # M2 = A[1,4] B[4,k]
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
