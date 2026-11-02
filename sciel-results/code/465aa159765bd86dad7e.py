"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16
    
    # Strassen decomposition for 2x2 matrix multiplication
    # C = A * B where A, B are 2x2
    # Rank 7 decomposition:
    # M1 = (A00) * (B01 + B11)
    # M2 = (A10 + A11) * (B00)
    # M3 = (A00 + A01) * (B10)
    # M4 = (A11) * (B10 + B00)
    # M5 = (A00 + A10) * (B01)
    # M6 = (A01 - A11) * (B10 + B11)
    # M7 = (A10 - A00) * (B00 + B01)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We want to compute C (2x8) = A (2x4) * B (4x8)
    # Split A into A_left (2x2) and A_right (2x2)
    # A = [A_left, A_right]
    # A_left = [[A00, A01], [A10, A11]]
    # A_right = [[A20, A21], [A30, A31]]
    #
    # Split B into 4 blocks B0, B1, B2, B3, each 2x2
    # B = [B0, B1, B2, B3] vertically? No, B is 4x8.
    # Rows of B correspond to columns of A.
    # B = [[B00, B01, B02, B03],  (Row 0 of B, 8 cols)
    #      [B10, B11, B12, B13],  (Row 1 of B)
    #      [B20, B21, B22, B23],  (Row 2 of B)
    #      [B30, B31, B32, B33]]  (Row 3 of B)
    #
    # We can group the 4 rows of B into two 2x4 blocks:
    # B_top = [[B00, B01, B02, B03], [B10, B11, B12, B13]]
    # B_bot = [[B20, B21, B22, B23], [B30, B31, B32, B33]]
    #
    # Then C = A_left * B_top + A_right * B_bot
    #
    # Now, B_top is 2x4. We can split B_top into two 2x2 blocks: B_top_left, B_top_right.
    # B_top_left = [[B00, B01], [B10, B11]]
    # B_top_right = [[B02, B03], [B12, B13]]
    # Similarly for B_bot.
    #
    # So C = A_left * [B_tl, B_tr] + A_right * [B_bl, B_br]
    # C = [A_left * B_tl + A_right * B_bl, A_left * B_tr + A_right * B_br]
    #
    # Let C1 = A_left * B_tl + A_right * B_bl (2x2)
    # Let C2 = A_left * B_tr + A_right * B_br (2x2)
    #
    # Each term is a sum of two 2x2 matrix products.
    # A_left * B_tl can be done in rank 7.
    # A_right * B_bl can be done in rank 7.
    # So C1 requires rank 14.
    # C2 requires rank 14.
    # Total rank = 28.
    
    # Let's build the matrices U, V, W for this rank 28 decomposition.
    # The 28 terms are:
    # 7 terms for A_left * B_tl
    # 7 terms for A_right * B_bl
    # 7 terms for A_left * B_tr
    # 7 terms for A_right * B_br
    
    rank = 28
    U = np.zeros((U_dim, rank))
    V = np.zeros((V_dim, rank))
    W = np.zeros((W_dim, rank))
    
    # Indices for A (2x4):
    # A[0,0]=A00, A[0,1]=A01, A[0,2]=A20, A[0,3]=A21
    # A[1,0]=A10, A[1,1]=A11, A[1,2]=A30, A[1,3]=A31
    # U index: i*m + j
    # A00 -> 0*4+0 = 0
    # A01 -> 0*4+1 = 1
    # A20 -> 0*4+2 = 2
    # A21 -> 0*4+3 = 3
    # A10 -> 1*4+0 = 4
    # A11 -> 1*4+1 = 5
    # A30 -> 1*4+2 = 6
    # A31 -> 1*4+3 = 7
    
    # Indices for B (4x8):
    # B[j, k] -> j*8 + k
    # B_tl:
    # B00 -> 0*8+0 = 0
    # B01 -> 0*8+1 = 1
    # B10 -> 1*8+0 = 8
    # B11 -> 1*8+1 = 9
    # B_bl:
    # B20 -> 2*8+0 = 16
    # B21 -> 2*8+1 = 17
    # B30 -> 3*8+0 = 24
    # B31 -> 3*8+1 = 25
    # B_tr:
    # B02 -> 0*8+2 = 2
    # B03 -> 0*8+3 = 3
    # B12 -> 1*8+2 = 10
    # B13 -> 1*8+3 = 11
    # B_br:
    # B22 -> 2*8+2 = 18
    # B23 -> 2*8+3 = 19
    # B32 -> 3*8+2 = 26
    # B33 -> 3*8+3 = 27
    
    # Indices for C (2x8):
    # C[i, k] -> k*n + i
    # C[0,0]=C00 -> 0*2+0=0
    # C[0,1]=C01 -> 1*2+0=2
    # C[0,2]=C02 -> 2*2+0=4
    # C[0,3]=C03 -> 3*2+0=6
    # C[1,0]=C10 -> 0*2+1=1
    # C[1,1]=C11 -> 1*2+1=3
    # C[1,2]=C12 -> 2*2+1=5
    # C[1,3]=C13 -> 3*2+1=7
    
    # Strassen factors for 2x2:
    # M1 = A00 * (B01 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = (A00 + A01) * B10
    # M4 = A11 * (B10 + B00)
    # M5 = (A00 + A10) * B01
    # M6 = (A01 - A11) * (B10 + B11)
    # M7 = (A10 - A00) * (B00 + B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_term(base_col, A_idx_map, B_idx_map, C_idx_map):
        """
        Adds 7 columns for one 2x2 Strassen multiplication.
        A_idx_map: dict mapping 'A00', 'A01', 'A10', 'A11' to U indices
        B_idx_map: dict mapping 'B00', 'B01', 'B10', 'B11' to V indices
        C_idx_map: dict mapping 'C00', 'C01', 'C10', 'C11' to W indices
        """
        # M1
        col = base_col
        U[A_idx_map['A00'], col] = 1
        V[B_idx_map['B01'], col] = 1
        V[B_idx_map['B11'], col] = 1
        W[C_idx_map['C00'], col] = 1
        W[C_idx_map['C11'], col] = 1
        
        # M2
        col = base_col + 1
        U[A_idx_map['A10'], col] = 1
        U[A_idx_map['A11'], col] = 1
        V[B_idx_map['B00'], col] = 1
        W[C_idx_map['C10'], col] = 1
        W[C_idx_map['C11'], col] = -1
        
        # M3
        col = base_col + 2
        U[A_idx_map['A00'], col] = 1
        U[A_idx_map['A01'], col] = 1
        V[B_idx_map['B10'], col] = 1
        W[C_idx_map['C01'], col] = 1
        W[C_idx_map['C11'], col] = 1
        
        # M4
        col = base_col + 3
        U[A_idx_map['A11'], col] = 1
        V[B_idx_map['B10'], col] = 1
        V[B_idx_map['B00'], col] = 1
        W[C_idx_map['C00'], col] = 1
        W[C_idx_map['C10'], col] = 1
        
        # M5
        col = base_col + 4
        U[A_idx_map['A00'], col] = 1
        U[A_idx_map['A10'], col] = 1
        V[B_idx_map['B01'], col] = 1
        W[C_idx_map['C00'], col] = -1
        W[C_idx_map['C01'], col] = 1
        
        # M6
        col = base_col + 5
        U[A_idx_map['A01'], col] = 1
        U[A_idx_map['A11'], col] = -1
        V[B_idx_map['B10'], col] = 1
        V[B_idx_map['B11'], col] = 1
        W[C_idx_map['C11'], col] = 1
        
        # M7
        col = base_col + 6
        U[A_idx_map['A10'], col] = 1
        U[A_idx_map['A00'], col] = -1
        V[B_idx_map['B00'], col] = 1
        V[B_idx_map['B01'], col] = 1
        W[C_idx_map['C00'], col] = 1

    # Term 1: A_left * B_tl
    # A_left: A00(0), A01(1), A10(4), A11(5)
    # B_tl: B00(0), B01(1), B10(8), B11(9)
    # C1 (left part of C): C00(0), C01(2), C10(1), C11(3)
    add_strassen_term(0, 
                      {'A00': 0, 'A01': 1, 'A10': 4, 'A11': 5},
                      {'B00': 0, 'B01': 1, 'B10': 8, 'B11': 9},
                      {'C00': 0, 'C01': 2, 'C10': 1, 'C11': 3})
    
    # Term 2: A_right * B_bl
    # A_right: A20(2), A21(3), A30(6), A31(7)
    # B_bl: B20(16), B21(17), B30(24), B31(25)
    # C1 (left part of C): C00(0), C01(2), C10(1), C11(3)
    add_strassen_term(7, 
                      {'A00': 2, 'A01': 3, 'A10': 6, 'A11': 7},
                      {'B00': 16, 'B01': 17, 'B10': 24, 'B11': 25},
                      {'C00': 0, 'C01': 2, 'C10': 1, 'C11': 3})
    
    # Term 3: A_left * B_tr
    # A_left: A00(0), A01(1), A10(4), A11(5)
    # B_tr: B02(2), B03(3), B12(10), B13(11)
    # C2 (right part of C): C02(4), C03(6), C12(5), C13(7)
    add_strassen_term(14, 
                      {'A00': 0, 'A01': 1, 'A10': 4, 'A11': 5},
                      {'B00': 2, 'B01': 3, 'B10': 10, 'B11': 11},
                      {'C00': 4, 'C01': 6, 'C10': 5, 'C11': 7})
    
    # Term 4: A_right * B_br
    # A_right: A20(2), A21(3), A30(6), A31(7)
    # B_br: B22(18), B23(19), B32(26), B33(27)
    # C2 (right part of C): C02(4), C03(6), C12(5), C13(7)
    add_strassen_term(21, 
                      {'A00': 2, 'A01': 3, 'A10': 6, 'A11': 7},
                      {'B00': 18, 'B01': 19, 'B10': 26, 'B11': 27},
                      {'C00': 4, 'C01': 6, 'C10': 5, 'C11': 7})
    
    # Verify decomposition
    def check_decomp(U, V, W, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Ur = U[i*m+j, :]
                    Vr = V[j*p+k, :]
                    Wr = W[k*n+i, :]
                    val = np.dot(Ur * Vr, Wr)
                    if abs(val - 1.0) > tol:
                        return False
        return True
        
    if not check_decomp(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
