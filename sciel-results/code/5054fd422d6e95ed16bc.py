"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We'll build a decomposition using Strassen's algorithm on 2x2 blocks.
    # A is 4x4, B is 4x8, C = A*B is 4x8.
    # 
    # Partition A into 2x2 blocks of 2x2:
    #   A = [[A11, A12],
    #        [A21, A22]]
    # Partition B into top/bottom halves (each 2x8):
    #   B = [B_top; B_bot]  where B_top is 2x8, B_bot is 2x8
    # C = [C_top; C_bot] where C_top = A11*B_top + A12*B_bot (2x8), 
    #                          C_bot = A21*B_top + A22*B_bot (2x8)
    #
    # For each 2x2 * 2x8 product, we use a Strassen-like decomposition.
    # Actually, let's use a simpler approach: treat each 2x2 * 2x8 as 
    # two 2x2 * 2x4 blocks, and apply Strassen to each 2x2*2x2 sub-block.
    #
    # Better: use the known rank-98 construction.
    # 
    # Let me build it systematically using Strassen at 2x2 level.
    # 
    # For multiplying X (2x2) by Y (2x8), write Y = [Y1 Y2] where Y1,Y2 are 2x4.
    # Then X*Y = [X*Y1, X*Y2]. Each X*Yi is 2x2 * 2x4.
    # For 2x2 * 2x4, write Yi = [Yi1 Yi2] (2x2 each). X*Yi = [X*Yi1, X*Yi2].
    # Each X*Yij is 2x2 * 2x2, which Strassen does in 7 mults.
    # So 2x2 * 2x8 = 4 * (2x2*2x2) = 4*7 = 28 mults.
    #
    # Total: We have 4 such 2x2*2x8 products (A11*B_top, A12*B_bot, A21*B_top, A22*B_bot)
    # But we can share! Using Strassen at the 4x4 level:
    #
    # Standard Strassen for 4x4 * 4x4 uses 7 recursive calls of 2x2*2x2.
    # For 4x4 * 4x8, we can do:
    #   C = A * B where B is 4x8. Split B = [B1 | B2] (4x4 each).
    #   C = A*B1 | A*B2
    #   Each A*Bj is 4x4*4x4, using Strassen rank 7 (each sub-call is 2x2*2x2 rank 7)
    #   So each 4x4*4x4 has rank 7*7=49? No, Strassen 4x4 uses 7 calls of 2x2*2x2.
    #   Each 2x2*2x2 uses 7 scalar mults. So 4x4*4x4 rank = 7*7 = 49.
    #   Total for 4x4*4x8 = 2 * 49 = 98.
    #
    # That gives rank 98! Let me construct this.
    
    def strassen_2x2_2x2(U, V, W, offset):
        """Add Strassen 2x2*2x2 factors. U,V,W are lists of lists, offset is column offset."""
        # U rows: 0,1 for X11,X12,X21,X22 (4 rows for 2x2)
        # Actually U has 4 rows (2x2), V has 4 rows (2x2), W has 4 rows (2x2)
        # Strassen:
        # M1 = (X11+X22)*(Y11+Y22)
        # M2 = (X21+X22)*Y11
        # M3 = X11*(Y12-Y22)
        # M4 = X22*(Y21-Y11)
        # M5 = (X11+X12)*Y22
        # M6 = (X21-X11)*(Y11+Y12)
        # M7 = (X12-X22)*(Y21+Y22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # U rows indexed 0-3: [0,0],[0,1],[1,0],[1,1] for X
        # V rows indexed 0-3: [0,0],[0,1],[1,0],[1,1] for Y  
        # W rows indexed 0-3: [0,0],[0,1],[1,0],[1,1] for C
        
        # M1
        U[0][offset] += 1; U[3][offset] += 1
        V[0][offset] += 1; V[3][offset] += 1
        W[0][offset] += 1; W[1][offset] += 1; W[2][offset] += 1; W[3][offset] += 1
        offset += 1
        # M2
        U[2][offset] += 1; U[3][offset] += 1
        V[0][offset] += 1
        W[2][offset] += 1; W[3][offset] += 1
        offset += 1
        # M3
        U[0][offset] += 1
        V[1][offset] += 1; V[3][offset] -= 1
        W[1][offset] += 1; W[3][offset] += 1
        offset += 1
        # M4
        U[3][offset] += 1
        V[2][offset] += 1; V[0][offset] -= 1
        W[0][offset] += 1; W[2][offset] += 1
        offset += 1
        # M5
        U[0][offset] += 1; U[1][offset] += 1
        V[3][offset] += 1
        W[0][offset] -= 1; W[1][offset] += 1
        offset += 1
        # M6
        U[2][offset] += 1; U[0][offset] -= 1
        V[0][offset] += 1; V[1][offset] += 1
        W[3][offset] += 1
        offset += 1
        # M7
        U[1][offset] += 1; U[3][offset] -= 1
        V[2][offset] += 1; V[3][offset] += 1
        W[0][offset] += 1; W[3][offset] += 1
        offset += 1
        
        return offset
    
    def strassen_4x4_4x4(U, V, W, u_off, v_off, w_off):
        """Apply Strassen 4x4*4x4 using 7 calls to 2x2*2x2.
        U rows 0-15 for 4x4, V rows 0-15 for 4x4, W rows 0-15 for 4x4.
        Returns new offset."""
        # 4x4 blocks:
        # X = [[X11, X12], [X21, X22]] each 2x2
        # X11 rows: 0,1; X12 rows: 0,1 cols 2,3 -> in U: rows 0-1 (X11), 2-3 (X12), 4-5 (X21), 6-7 (X22)
        # Wait, U is n*m = 4*4 = 16 rows. Row index = i*4+j for A[i,j].
        # X11 = A[0:2, 0:2] -> U rows 0,1,4,5
        # X12 = A[0:2, 2:4] -> U rows 2,3,6,7
        # X21 = A[2:4, 0:2] -> U rows 8,9,12,13
        # X22 = A[2:4, 2:4] -> U rows 10,11,14,15
        
        # Y = B[0:4, 0:4] -> V rows 0,1,4,5 (Y11), 2,3,6,7 (Y12), 8,9,12,13 (Y21), 10,11,14,15 (Y22)
        # C = result 4x4 -> W rows 0,1,4,5 (C11), 2,3,6,7 (C12), 8,9,12,13 (C21), 10,11,14,15 (C22)
        
        # Strassen:
        # M1 = (X11+X22)*(Y11+Y22) -> C11 += M1, C12 += M1, C21 += M1, C22 += M1
        # M2 = (X21+X22)*Y11 -> C21 += M2, C22 += M2
        # M3 = X11*(Y12-Y22) -> C12 += M3, C22 += M3
        # M4 = X22*(Y21-Y11) -> C11 += M4, C21 += M4
        # M5 = (X11+X12)*Y22 -> C11 -= M5, C12 += M5
        # M6 = (X21-X11)*(Y11+Y12) -> C22 += M6
        # M7 = (X12-X22)*(Y21+Y22) -> C11 += M7, C22 += M7
        
        # For each Strassen factor, we need to add to U/V/W the appropriate 2x2 sub-matrix contributions.
        # Since U/V/W are flat, we add coefficients to the specific rows.
        
        X11 = [0,1,4,5]
        X12 = [2,3,6,7]
        X21 = [8,9,12,13]
        X22 = [10,11,14,15]
        Y11 = [0,1,4,5]
        Y12 = [2,3,6,7]
        Y21 = [8,9,12,13]
        Y22 = [10,11,14,15]
        C11 = [0,1,4,5]
        C12 = [2,3,6,7]
        C21 = [8,9,12,13]
        C22 = [10,11,14,15]
        
        # M1: (X11+X22) * (Y11+Y22)
        for r in X11: U[r][offset] += 1
        for r in X22: U[r][offset] += 1
        for r in Y11: V[r][offset] += 1
        for r in Y22: V[r][offset] += 1
        for r in C11: W[r][offset] += 1
        for r in C12: W[r][offset] += 1
        for r in C21: W[r][offset] += 1
        for r in C22: W[r][offset] += 1
        offset += 1
        
        # M2: (X21+X22) * Y11
        for r in X21: U[r][offset] += 1
        for r in X22: U[r][offset] += 1
        for r in Y11: V[r][offset] += 1
        for r in C21: W[r][offset] += 1
        for r in C22: W[r][offset] += 1
        offset += 1
        
        # M3: X11 * (Y12-Y22)
        for r in X11: U[r][offset] += 1
        for r in Y12: V[r][offset] += 1
        for r in Y22: V[r][offset] -= 1
        for r in C12: W[r][offset] += 1
        for r in C22: W[r][offset] += 1
        offset += 1
        
        # M4: X22 * (Y21-Y11)
        for r in X22: U[r][offset] += 1
        for r in Y21: V[r][offset] += 1
        for r in Y11: V[r][offset] -= 1
        for r in C11: W[r][offset] += 1
        for r in C21: W[r][offset] += 1
        offset += 1
        
        # M5: (X11+X12) * Y22
        for r in X11: U[r][offset] += 1
        for r in X12: U[r][offset] += 1
        for r in Y22: V[r][offset] += 1
        for r in C11: W[r][offset] -= 1
        for r in C12: W[r][offset] += 1
        offset += 1
        
        # M6: (X21-X11) * (Y11+Y12)
        for r in X21: U[r][offset] += 1
        for r in X11: U[r][offset] -= 1
        for r in Y11: V[r][offset] += 1
        for r in Y12: V[r][offset] += 1
        for r in C22: W[r][offset] += 1
        offset += 1
        
        # M7: (X12-X22) * (Y21+Y22)
        for r in X12: U[r][offset] += 1
        for r in X22: U[r][offset] -= 1
        for r in Y21: V[r][offset] += 1
        for r in Y22: V[r][offset] += 1
        for r in C11: W[r][offset] += 1
        for r in C22: W[r][offset] += 1
        offset += 1
        
        return offset
    
    # Now for 4x4 * 4x8:
    # A is 4x4, B is 4x8. Split B = [B1 | B2] where B1 is 4x4, B2 is 4x4.
    # C = A*B1 | A*B2
    # Each A*Bj is 4x4*4x4, using Strassen with 7 factors of 2x2*2x2 (rank 7 each).
    # Total rank = 2 * 7 = 14? No, each 4x4*4x4 Strassen has 7 factors, each factor is a 2x2*2x2 mult.
    # But in tensor decomposition, each 2x2*2x2 factor contributes 7 to the rank.
    # So each 4x4*4x4 contributes 7*7 = 49 to rank? No!
    
    # Wait, I'm confusing myself. In the tensor decomposition:
    # A 2x2*2x2 multiplication tensor has rank 7 (Strassen).
    # A 4x4*4x4 multiplication can be done with 7 copies of 2x2*2x2, so rank = 7*7 = 49.
    # A 4x4*4x8 = two 4x4*4x4, so rank = 2*49 = 98.
    
    # But in my construction above, strassen_4x4_4x4 adds 7 columns, each of which is a scalar multiplication.
    # That's wrong! Each "factor" in Strassen 4x4*4x4 is itself a 2x2*2x2 multiplication, which needs 7 scalar mults.
    
    # Let me redo this properly. I need to compose the decompositions.
    
    # Approach: Build the full decomposition directly.
    # For 4x4 * 4x8:
    # Split B into B1 (4x4) and B2 (4x4).
    # C1 = A * B1 (4x4), C2 = A * B2 (4x4).
    # For each, use Strassen 4x4*4x4 which decomposes into 7 sub-problems of 2x2*2x2.
    # Each 2x2*2x2 uses Strassen with 7 scalar multiplications.
    # Total: 2 * 7 * 7 = 98 scalar multiplications.
    
    rank = 98
    U = [[0] * rank for _ in range(16)]  # 4*4
    V = [[0] * rank for _ in range(32)]  # 4*8
    W = [[0] * rank for _ in range(32)]  # 8*4 (p*n = 8*4)
    
    # W index: W[k*n+i, c] where k is column of B (0-7), i is row of A (0-3).
    # So W row index = k*4 + i.
    # C[i,k] = sum_c U[i*4+j, c] * V[j*4+k, c] ... wait no.
    # 
    # The tensor decomposition: C[i,k] = sum_{j,c} A[i,j] * B[j,k] * T[i,j,k,c]
    # In the factor form: C[i,k] = sum_c U[i*4 + j... ] 
    # 
    # Actually the standard form: 
    # C[i,k] = sum_c U[i*m + ?, c] ... 
    # 
    # Let me re-read the convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # C[i,k] = sum_{j,c} A[i,j] * B[j,k] * W[k*n+i, c] ... no.
    # 
    # The decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c where T[i,j,k] = A[i,j]*B[j,k]
    # U_c[i*m+j] = coefficient for A[i,j]
    # V_c[j*p+k] = coefficient for B[j,k]  
    # W_c[k*n+i] = coefficient for C[i,k]
    # 
    # So C[i,k] = sum_c (sum_j U_c[i*m+j] * A[i,j]) * (sum_j V_c[j*p+k] * B[j,k]) * W_c[k*n+i]
    # 
    # Wait, that's not right either. The correct form is:
    # C[i,k] = sum_c U_c[i*m+j] ... no, it's a tensor product.
    # T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[i,k]
    # where U_c is n×m, V_c is m×p, W_c is p×n (but indexed as k*n+i).
    # 
    # In flat form: U_flat[i*m+j, c], V_flat[j*p+k, c], W_flat[k*n+i, c]
    # C[i,k] = sum_c sum_j U_flat[i*m+j, c] * A[i,j] ... no.
    # 
    # Actually: T[i,j,k] = sum_c U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
    # And T[i,j,k] should equal A[i,j]*B[j,k] for the multiplication tensor.
    # 
    # So we need: A[i,j]*B[j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # for all i,j,k.
    
    # Let me build this by composing Strassen decompositions.
    
    # For 4x4*4x4 (B1 part, k=0..3):
    # Strassen 4x4*4x4:
    # M1 = (X11+X22)*(Y11+Y22), C11+=M1, C12+=M1, C21+=M1, C22+=M1
    # etc.
    # Each Mq is a 2x2*2x2 product.
    
    # Let me define helper to add a 2x2*2x2 Strassen decomposition at given block positions.
    
    def add_2x2_strassen(U, V, W, offset, 
                         u_rows, v_rows, w_rows):
        """Add Strassen 2x2*2x2.
        u_rows: list of 4 U row indices for the 2x2 X matrix (row-major: X11r1, X11r2, X21r1, X21r2)
                 Actually for 2x2, rows are [r0, r1, r2, r3] corresponding to (0,0),(0,1),(1,0),(1,1)
        v_rows: list of 4 V row indices for the 2x2 Y matrix
        w_rows: list of 4 W row indices for the 2x2 C matrix
        """
        # u_rows[0]=X11, u_rows[1]=X12, u_rows[2]=X21, u_rows[3]=X22
        # v_rows[0]=Y11, v_rows[1]=Y12, v_rows[2]=Y21, v_rows[3]=Y22
        # w_rows[0]=C11, w_rows[1]=C12, w_rows[2]=C21, w_rows[3]=C22
        
        # M1 = (X11+X22)*(Y11+Y22)
        U[u_rows[0]][offset] += 1; U[u_rows[3]][offset] += 1
        V[v_rows[0]][offset] += 1; V[v_rows[3]][offset] += 1
        W[w_rows[0]][offset] += 1; W[w_rows[1]][offset] += 1
        W[w_rows[2]][offset] += 1; W[w_rows[3]][offset] += 1
        offset += 1
        # M2 = (X21+X22)*Y11
        U[u_rows[2]][offset] += 1; U[u_rows[3]][offset] += 1
        V[v_rows[0]][offset] += 1
        W[w_rows[2]][offset] += 1; W[w_rows[3]][offset] += 1
        offset += 1
        # M3 = X11*(Y12-Y22)
        U[u_rows[0]][offset] += 1
        V[v_rows[1]][offset] += 1; V[v_rows[3]][offset] -= 1
        W[w_rows[1]][offset] += 1; W[w_rows[3]][offset] += 1
        offset += 1
        # M4 = X22*(Y21-Y11)
        U[u_rows[3]][offset] += 1
        V[v_rows[2]][offset] += 1; V[v_rows[0]][offset] -= 1
        W[w_rows[0]][offset] += 1; W[w_rows[2]][offset] += 1
        offset += 1
        # M5 = (X11+X12)*Y22
        U[u_rows[0]][offset] += 1; U[u_rows[1]][offset] += 1
        V[v_rows[3]][offset] += 1
        W[w_rows[0]][offset] -= 1; W[w_rows[1]][offset] += 1
        offset += 1
        # M6 = (X21-X11)*(Y11+Y12)
        U[u_rows[2]][offset] += 1; U[u_rows[0]][offset] -= 1
        V[v_rows[0]][offset] += 1; V[v_rows[1]][offset] += 1
        W[w_rows[3]][offset] += 1
        offset += 1
        # M7 = (X12-X22)*(Y21+Y22)
        U[u_rows[1]][offset] += 1; U[u_rows[3]][offset] -= 1
        V[v_rows[2]][offset] += 1; V[v_rows[3]][offset] += 1
        W[w_rows[0]][offset] += 1; W[w_rows[3]][offset] += 1
        offset += 1
        
        return offset
    
    # Now build the full 4x4*4x8 decomposition.
    # A is 4x4: U row for A[i,j] is i*4+j.
    # B is 4x8: V row for B[j,k] is j*8+k.
    # C is 4x8: W row for C[i,k] is k*4+i.
    
    offset = 0
    
    # Split B into B1 (cols 0-3) and B2 (cols 4-7).
    # For each half, do Strassen 4x4*4x4.
    
    for b_half in range(2):
        k_offset = b_half * 4  # column offset in B and C
        
        # X = A (4x4), Y = B[:, k_offset:k_offset+4] (4x4)
        # X11 = A[0:2, 0:2], X12 = A[0:2, 2:4], X21 = A[2:4, 0:2], X22 = A[2:4, 2:4]
        # Y11 = B[0:2, k_off:k_off+2], Y12 = B[0:2, k_off+2:k_off+4], Y21 = B[2:4, k_off:k_off+2], Y22 = B[2:4, k_off+2:k_off+4]
        # C11 = C[0:2, k_off:k_off+2], C12 = C[0:2, k_off+2:k_off+4], C21 = C[2:4, k_off:k_off+2], C22 = C[2:4, k_off+2:k_off+4]
        
        # For each Strassen factor, we have a 2x2*2x2 sub-problem.
        # X block rows in U: for X[bi, bj] where bi,bj in 0,1: U row = (2*bi)*4 + (2*bj) ... 
        # Actually X is a 2x2 block of A. X[0,0] = A[0,0], X[0,1] = A[0,1], X[1,0] = A[1,0], X[1,1] = A[1,1]
        # But X is 2x2, so X[bi,bj] for bi,bj in {0,1}. In A, this is A[2*top+bi, 2*left+bj].
        
        # For Strassen 4x4*4x4, we have 7 factors. Each factor involves a 2x2*2x2 product.
        # Let me enumerate the 7 factors and their 2x2 sub-blocks.
        
        # Factor 1: (X11+X22) * (Y11+Y22)
        # X11 = A[0:2, 0:2] -> U rows: 0,1,4,5 (A[0,0],A[0,1],A[1,0],A[1,1])
        # X22 = A[2:4, 2:4] -> U rows: 10,11,14,15
        # Y11 = B[0:2, k: k+2] -> V rows: 0*8+k, 1*8+k, 0*8+k+1, 1*8+k+1 = k, k+1, 8+k, 8+k+1
        # Y22 = B[2:4, k+2: k+4] -> V rows: 2*8+k+2, 3*8+k+2, 2*8+k+3, 3*8+k+3 = 18+k, 18+k+1, 26+k, 26+k+1
        # C11 = C[0:2, k:k+2] -> W rows: k*4+0, k*4+1, (k+1)*4+0, (k+1)*4+1
        # C12 = C[0:2, k+2:k+4] -> W rows: (k+2)*4+0, (k+2)*4+1, (k+3)*4+0, (k+3)*4+1
        # C21 = C[2:4, k:k+2] -> W rows: k*4+2, k*4+3, (k+1)*4+2, (k+1)*4+3
        # C22 = C[2:4, k+2:k+4] -> W rows: (k+2)*4+2, (k+2)*4+3, (k+3)*4+2, (k+3)*4+3
        
        # This is getting complex. Let me just directly compute the U, V, W rows for each sub-block.
        
        # For a 2x2 block of A at top-left (ti, tj): rows are [ti*4+tj, ti*4+tj+1, (ti+1)*4+tj, (ti+1)*4+tj+1]
        # For a 2x2 block of B at (ti, kj): rows are [ti*8+kj, ti*8+kj+1, (ti+1)*8+kj, (ti+1)*8+kj+1]
        # For a 2x2 block of C at (ti, kj): rows are [kj*4+ti, kj*4+ti+1, (kj+1)*4+ti, (kj+1)*4+ti+1]
        
        def a_block(ti, tj):
            return [ti*4+tj, ti*4+tj+1, (ti+1)*4+tj, (ti+1)*4+tj+1]
        
        def b_block(ti, kj):
            return [ti*8+kj, ti*8+kj+1, (ti+1)*8+kj, (ti+1)*8+kj+1]
        
        def c_block(ti, kj):
            return [kj*4+ti, kj*4+ti+1, (kj+1)*4+ti, (kj+1)*4+ti+1]
        
        # Strassen 4x4*4x4 factors:
        # M1 = (X11+X22)*(Y11+Y22)
        #   X part: X11 and X22, Y part: Y11 and Y22
        #   C parts: C11, C12, C21, C22 all get +1
        #   But this is a 2x2*2x2 where the "X" is X11+X22 (not a single 2x2 block!)
        
        # I see the issue. In Strassen 4x4*4x4, each factor like (X11+X22)*(Y11+Y22) is NOT a single 2x2*2x2 multiplication.
        # It's the product of a sum of two 2x2 matrices with a sum of two 2x2 matrices.
        # To decompose this into scalar multiplications, I need to expand:
        # (X11+X22)*(Y11+Y22) = X11*Y11 + X11*Y22 + X22*Y11 + X22*Y22
        # That's 4 separate 2x2*2x2 products, each needing 7 scalar mults = 28 mults.
        # But Strassen says the whole 4x4*4x4 needs only 7*7=49 mults, not 7*28=196.
        
        # The key insight: (X11+X22)*(Y11+Y22) is itself a 2x2*2x2 multiplication where the "matrices" are 2x2 blocks.
        # So I apply Strassen 2x2*2x2 to this, where the elements of the 2x2 "matrix" are themselves 2x2 blocks.
        # But in the tensor decomposition, I need scalar coefficients.
        
        # Alternative approach: just apply Strassen 2x2*2x2 directly where the 2x2 matrices have scalar entries.
        # For 4x4*4x4, I can view it as 2x2 blocks and apply Strassen, but each "scalar" in the Strassen formula
        # is actually a 2x2*2x2 multiplication.
        
        # So the total rank for 4x4*4x4 is 7 (Strassen factors) * 7 (each factor is 2x2*2x2 Strassen) = 49.
        # And for 4x4*4x8 it's 2 * 49 = 98.
        
        # Let me just build it by nested Strassen.
        
        # For factor M1 = (X11+X22)*(Y11+Y22):
        # This is a 2x2*2x2 multiplication where:
        #   The "2x2 matrix" P has entries P[0,0]=X11, P[0,1]=0, P[1,0]=0, P[1,1]=X22 (i.e., P = X11+X22 in block sense)
        #   Actually P = [[X11, 0], [0, X22]] + [[0,0],[0,0]] ... no.
        #   P = X11 + X22 means P[0,0]=X11[0,0]+X22[0,0], etc. So P is a 2x2 matrix where each entry is a 2x2 block sum.
        
        # I think the cleanest way is to just directly construct the 98-rank decomposition by iterating over
        # all the Strassen factors at both levels.
        
        # Let me define the 7 Strassen 4x4*4x4 factors, each specifying which 2x2 blocks of A and B are combined,
        # and then for each, apply Strassen 2x2*2x2.
        
        # Actually, let me just hardcode a known correct rank-98 decomposition.
        # I'll use the approach: for each of the 7 Strassen 4x4*4x4 factors, and for each of the 7 Strassen 2x2*2x2
        # factors within it, add the appropriate coefficients.
        
        # This is getting very complex to code correctly. Let me verify my understanding with a simpler approach.
        
        # Alternative: Use the fact that 4x4*4x8 can be seen as 4x4*4x4 twice.
        # I'll just call strassen_4x4_4x4 twice, but I need to fix it to properly compose 2x2*2x2 Strassen.
        
        # Let me restart with a cleaner implementation.
        pass
    
    # Let me just build it directly with explicit indices.
    # I'll create a helper that adds one 2x2*2x2 Strassen decomposition.
    
    offset = 0
    
    def add_2x2_strassen_at(U, V, W, offset, a_ti, a_tj, b_ti, b_kj, c_ti, c_kj, signs_a, signs_b, signs_c):
        """Add a 2x2*2x2 Strassen decomposition.
        a_ti, a_tj: top-left of 2x2 block in A
        b_ti, b_kj: top-left of 2x2 block in B  
        c_ti, c_kj: top-left of 2x2 block in C
        signs_a: list of 4 signs for A block entries [a00, a01, a10, a11]
        signs_b: list of 4 signs for B block entries
        signs_c: list of 4 signs for C block entries (applied to W)
        """
        a_rows = a_block(a_ti, a_tj)
        b_rows = b_block(b_ti, b_kj)
        c_rows = c_block(c_ti, c_kj)
        
        # Apply Strassen 2x2*2x2 with given signs
        # M1 = (A00+A11)*(B00+B11)
        U[a_rows[0]][offset] += signs_a[0]; U[a_rows[3]][offset] += signs_a[3]
        V[b_rows[0]][offset] += signs_b[0]; V[b_rows[3]][offset] += signs_b[3]
        W[c_rows[0]][offset] += signs_c[0]; W[c_rows[1]][offset] += signs_c[1]
        W[c_rows[2]][offset] += signs_c[2]; W[c_rows[3]][offset] += signs_c[3]
        offset += 1
        # M2 = (A10+A11)*B00
        U[a_rows[2]][offset] += signs_a[2]; U[a_rows[3]][offset] += signs_a[3]
        V[b_rows[0]][offset] += signs_b[0]
        W[c_rows[2]][offset] += signs_c[2]; W[c_rows[3]][offset] += signs_c[3]
        offset += 1
        # M3 = A00*(B01-B11)
        U[a_rows[0]][offset] += signs_a[0]
        V[b_rows[1]][offset] += signs_b[1]; V[b_rows[3]][offset] -= signs_b[3]
        W[c_rows[1]][offset] += signs_c[1]; W[c_rows[3]][offset] += signs_c[3]
        offset += 1
        # M4 = A11*(B10-B00)
        U[a_rows[3]][offset] += signs_a[3]
        V[b_rows[2]][offset] += signs_b[2]; V[b_rows[0]][offset] -= signs_b[0]
        W[c_rows[0]][offset] += signs_c[0]; W[c_rows[2]][offset] += signs_c[2]
        offset += 1
        # M5 = (A00+A01)*B11
        U[a_rows[0]][offset] += signs_a[0]; U[a_rows[1]][offset] += signs_a[1]
        V[b_rows[3]][offset] += signs_b[3]
        W[c_rows[0]][offset] -= signs_c[0]; W[c_rows[1]][offset] += signs_c[1]
        offset += 1
        # M6 = (A10-A00)*(B00+B01)
        U[a_rows[2]][offset] += signs_a[2]; U[a_rows[0]][offset] -= signs_a[0]
        V[b_rows[0]][offset] += signs_b[0]; V[b_rows[1]][offset] += signs_b[1]
        W[c_rows[3]][offset] += signs_c[3]
        offset += 1
        # M7 = (A01-A11)*(B10+B11)
        U[a_rows[1]][offset] += signs_a[1]; U[a_rows[3]][offset] -= signs_a[3]
        V[b_rows[2]][offset] += signs_b[2]; V[b_rows[3]][offset] += signs_b[3]
        W[c_rows[0]][offset] += signs_c[0]; W[c_rows[3]][offset] += signs_c[3]
        offset += 1
        
        return offset
    
    # Now, for Strassen 4x4*4x4, the 7 factors are:
    # M1 = (X11+X22)*(Y11+Y22) -> C11+=M1, C12+=M1, C21+=M1, C22+=M1
    # M2 = (X21+X22)*Y11 -> C21+=M2, C22+=M2
    # M3 = X11*(Y12-Y22) -> C12+=M3, C22+=M3
    # M4 = X22*(Y21-Y11) -> C11+=M4, C21+=M4
    # M5 = (X11+X12)*Y22 -> C11-=M5, C12+=M5
    # M6 = (X21-X11)*(Y11+Y12) -> C22+=M6
    # M7 = (X12-X22)*(Y21+Y22) -> C11+=M7, C22+=M7
    
    # Each Mq is a 2x2*2x2 product. To decompose Mq, I apply Strassen 2x2*2x2.
    # But the "matrices" in Mq are sums/differences of 2x2 blocks.
    # 
    # For example, M1 = (X11+X22)*(Y11+Y22).
    # This is NOT a single 2x2*2x2 with simple signs. It's the product of two 2x2 matrices where
    # the first has X11 in position (0,0), X22 in position (1,1), and the second has Y11 in (0,0), Y22 in (1,1).
    # 
    # Wait, that's exactly a 2x2*2x2 where:
    # P = [[X11, 0], [0, X22]] (but X11, X22 are 2x2 blocks, so P is 4x4)
    # 
    # I think I'm overcomplicating this. Let me just verify: does the rank-98 decomposition actually work?
    # Let me test it numerically.
    
    # For now, let me just return the schoolbook decomposition to ensure correctness,
    # and add a TODO for the Strassen optimization.
    
    # Actually, let me just try to build and verify the rank-98 decomposition.
    
    # Clear and rebuild
    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    # Define the 7 Strassen 4x4*4x4 factors with their block structure.
    # Each factor specifies: (a_blocks, b_blocks, c_blocks, signs)
    # where a_blocks is a list of (ti, tj, sign) for A blocks, etc.
    
    # For factor M1 = (X11+X22)*(Y11+Y22):
    # A part: X11 (ti=0,tj=0,sign=+1), X22 (ti=1,tj=1,sign=+1)
    # B part: Y11 (ti=0,kj=0,sign=+1), Y22 (ti=1,kj=1,sign=+1)
    # C parts: C11 (ti=0,kj=0,sign=+1), C12 (ti=0,kj=1,sign=+1), C21 (ti=1,kj=0,sign=+1), C22 (ti=1,kj=1,sign=+1)
    
    # This is a 2x2*2x2 where the "2x2 matrices" are sparse block matrices.
    # To decompose into scalar mults, I need to expand the block multiplication.
    
    # I think the correct approach is: for each Strassen 4x4*4x4 factor, the product is a 4x4*4x4
    # multiplication of specific block combinations. I should just apply Strassen 2x2*2x2 to each
    # 2x2 sub-block pair.
    
    # Let me try a completely different, simpler approach: just use the known fact that
    # rank(4x4*4x4) <= 49 via Strassen, and construct it by direct enumeration.
    
    # For now, return schoolbook to be safe.
    rank = 128
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    column = 0
    for i in range(4):
        for j in range(4):
            for k in range(8):
                U[i * 4 + j][column] = 1
                V[j * 8 + k][column] = 1
                W[k * 4 + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
