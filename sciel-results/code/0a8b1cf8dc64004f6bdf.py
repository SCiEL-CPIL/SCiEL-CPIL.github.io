import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    terms = []

    def strassen_2x2_sum(i_off, j_off, k_off, k_width):
        """
        Compute C_block(2 x k_width) = A_block(2 x 2) * B_block1(2 x k_width) + A_block2(2 x 2) * B_block2(2 x k_width)
        where A_block is at rows [i_off, i_off+1], cols [j_off, j_off+1]
              A_block2 is at rows [i_off, i_off+1], cols [j_off+2, j_off+3]
              B_block1 is at rows [j_off, j_off+1], cols [k_off, k_off+k_width-1]
              B_block2 is at rows [j_off+2, j_off+3], cols [k_off, k_off+k_width-1]
              C_block is at rows [i_off, i_off+1], cols [k_off, k_off+k_width-1]
        
        Uses 7 Strassen terms for the 2x2xk_width case.
        """
        result = []
        
        if k_width == 2:
            # Standard 2x2x2 Strassen
            # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
            u = [0] * (n * m)
            u[i_off * m + j_off] = 1
            u[(i_off+1) * m + (j_off+1)] = 1
            v = [0] * (m * p)
            v[j_off * p + k_off] = 1
            v[(j_off+1) * p + (k_off+1)] = 1
            w = [0] * (p * n)
            w[k_off * n + i_off] = 1
            w[(k_off+1) * n + (i_off+1)] = 1
            result.append((u[:], v[:], w[:]))
            
            # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
            u = [0] * (n * m)
            u[(i_off+1) * m + j_off] = 1
            u[(i_off+1) * m + (j_off+1)] = 1
            v = [0] * (m * p)
            v[j_off * p + k_off] = 1
            w = [0] * (p * n)
            w[k_off * n + (i_off+1)] = 1
            w[(k_off+1) * n + (i_off+1)] = -1
            result.append((u[:], v[:], w[:]))
            
            # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
            u = [0] * (n * m)
            u[i_off * m + j_off] = 1
            v = [0] * (m * p)
            v[j_off * p + (k_off+1)] = 1
            v[(j_off+1) * p + (k_off+1)] = -1
            w = [0] * (p * n)
            w[(k_off+1) * n + i_off] = 1
            w[(k_off+1) * n + (i_off+1)] = 1
            result.append((u[:], v[:], w[:]))
            
            # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
            u = [0] * (n * m)
            u[(i_off+1) * m + (j_off+1)] = 1
            v = [0] * (m * p)
            v[(j_off+1) * p + k_off] = 1
            v[j_off * p + k_off] = -1
            w = [0] * (p * n)
            w[k_off * n + i_off] = 1
            w[k_off * n + (i_off+1)] = 1
            result.append((u[:], v[:], w[:]))
            
            # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
            u = [0] * (n * m)
            u[i_off * m + j_off] = 1
            u[i_off * m + (j_off+1)] = 1
            v = [0] * (m * p)
            v[(j_off+1) * p + (k_off+1)] = 1
            w = [0] * (p * n)
            w[(k_off+1) * n + i_off] = 1
            w[k_off * n + i_off] = -1
            result.append((u[:], v[:], w[:]))
            
            # M6 = (A21-A11)*(B11+B12) -> C22 += M6
            u = [0] * (n * m)
            u[(i_off+1) * m + j_off] = 1
            u[i_off * m + j_off] = -1
            v = [0] * (m * p)
            v[j_off * p + k_off] = 1
            v[j_off * p + (k_off+1)] = 1
            w = [0] * (p * n)
            w[(k_off+1) * n + (i_off+1)] = 1
            result.append((u[:], v[:], w[:]))
            
            # M7 = (A12-A22)*(B21+B22) -> C11 += M7
            u = [0] * (n * m)
            u[i_off * m + (j_off+1)] = 1
            u[(i_off+1) * m + (j_off+1)] = -1
            v = [0] * (m * p)
            v[(j_off+1) * p + k_off] = 1
            v[(j_off+1) * p + (k_off+1)] = 1
            w = [0] * (p * n)
            w[k_off * n + i_off] = 1
            result.append((u[:], v[:], w[:]))
            
        elif k_width == 1:
            # 2x2x1: C(2x2) = A(2x2) * B(2x1)
            # This has rank 4 (schoolbook is optimal)
            # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
            # C[0,1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
            # Wait, that's not right. Let me re-examine.
            # 
            # Actually for 2x2x1: A is 2x2, B is 2x1, C is 2x1
            # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
            # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
            # Rank is 4.
            
            # But wait, in my formulation, I have C(2 x k_width) = A1(2x2)*B1(2 x k_width) + A2(2x2)*B2(2 x k_width)
            # For k_width=1: C(2x1) = A1(2x2)*B1(2x1) + A2(2x2)*B2(2x1)
            # C[0] = A1[0,0]*B1[0] + A1[0,1]*B1[1] + A2[0,0]*B2[0] + A2[0,1]*B2[1]
            # C[1] = A1[1,0]*B1[0] + A1[1,1]*B1[1] + A2[1,0]*B2[0] + A2[1,1]*B2[1]
            # This is 8 products, rank at most 8.
            
            # Hmm, I think I need to reconsider my block decomposition.
            pass
        
        return result

    # Let me reconsider the approach.
    # For 4x4x4: use recursive Strassen = 28 terms
    # For 4x4x3: split into 4x4x2 (14 terms) + 4x4x1 (16 terms) = 30 terms
    # Total: 58 terms
    
    # Let me implement this properly.
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u[:], v[:], w[:]))
    
    # Part 1: 4x4x4 (columns 0-3 of B/C)
    # Use 2x2 Strassen at block level
    # C_top(2x4) = A_top(2x4) * B_top(4x4)
    # C_bot(2x4) = A_bot(2x4) * B_bot(4x4)
    # 
    # C_top = A1(2x2)*B11(2x2) + A2(2x2)*B21(2x2) for cols 0-1
    #         + A1(2x2)*B12(2x2) + A2(2x2)*B22(2x2) for cols 2-3
    # 
    # Wait, I need to be more careful with the block structure.
    # 
    # A is 4x4: split rows into top(0-1) and bot(2-3), cols into left(0-1) and right(2-3)
    #   A = [[A11, A12], [A21, A22]] where each is 2x2
    # B is 4x4: split rows into top(0-1) and bot(2-3), cols into left(0-1) and right(2-3)
    #   B = [[B11, B12], [B21, B22]] where each is 2x2
    # C is 4x4: C = [[C11, C12], [C21, C22]] where each is 2x2
    #
    # C11 = A11*B11 + A21*B21
    # C12 = A11*B12 + A21*B22
    # C21 = A12*B11 + A22*B21
    # C22 = A12*B12 + A22*B22
    #
    # Each Ci_j is a 2x2x2 matrix multiplication (sum of two 2x2 products), rank 7.
    
    def strassen_2x2x2(i_off, j1_off, j2_off, k_off):
        """
        Compute C(2x2) at rows [i_off, i_off+1], cols [k_off, k_off+1]
        = A_block1(2x2) * B_block1(2x2) + A_block2(2x2) * B_block2(2x2)
        where A_block1 is at rows [i_off, i_off+1], cols [j1_off, j1_off+1]
              A_block2 is at rows [i_off, i_off+1], cols [j2_off, j2_off+1]
              B_block1 is at rows [j1_off, j1_off+1], cols [k_off, k_off+1]
              B_block2 is at rows [j2_off, j2_off+1], cols [k_off, k_off+1]
        """
        result = []
        
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = [0] * (n * m)
        u[i_off * m + j1_off] = 1       # A11[0,0]
        u[(i_off+1) * m + (j2_off+1)] = 1  # A22[1,1]
        v = [0] * (m * p)
        v[j1_off * p + k_off] = 1       # B11[0,0]
        v[(j2_off+1) * p + (k_off+1)] = 1  # B22[1,1]
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))
        
        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j1_off] = 1       # A21[1,0]
        u[(i_off+1) * m + (j2_off+1)] = 1   # A22[1,1]
        v = [0] * (m * p)
        v[j1_off * p + k_off] = 1           # B11[0,0]
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = -1
        result.append((u[:], v[:], w[:]))
        
        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j1_off] = 1           # A11[0,0]
        v = [0] * (m * p)
        v[j1_off * p + (k_off+1)] = 1       # B11[0,1]
        v[(j2_off+1) * p + (k_off+1)] = -1  # B22[1,1]
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))
        
        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j2_off+1)] = 1   # A22[1,1]
        v = [0] * (m * p)
        v[(j2_off+1) * p + k_off] = 1       # B22[1,0]
        v[j1_off * p + k_off] = -1          # B11[0,0]
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[k_off * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))
        
        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[i_off * m + j1_off] = 1           # A11[0,0]
        u[i_off * m + (j2_off+1)] = 1       # A12[0,1]
        v = [0] * (m * p)
        v[(j2_off+1) * p + (k_off+1)] = 1   # B22[1,1]
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[k_off * n + i_off] = -1
        result.append((u[:], v[:], w[:]))
        
        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j1_off] = 1       # A21[1,0]
        u[i_off * m + j1_off] = -1          # A11[0,0]
        v = [0] * (m * p)
        v[j1_off * p + k_off] = 1           # B11[0,0]
        v[j1_off * p + (k_off+1)] = 1       # B11[0,1]
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))
        
        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j2_off+1)] = 1       # A12[0,1]
        u[(i_off+1) * m + (j2_off+1)] = -1  # A22[1,1]
        v = [0] * (m * p)
        v[(j2_off+1) * p + k_off] = 1       # B22[1,0]
        v[(j2_off+1) * p + (k_off+1)] = 1   # B22[1,1]
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        result.append((u[:], v[:], w[:]))
        
        return result
    
    # 4x4x4 part: 4 blocks of 2x2x2 Strassen
    # C11 (rows 0-1, cols 0-1) = A11*B11 + A21*B21
    terms.extend(strassen_2x2x2(0, 0, 2, 0))
    # C12 (rows 0-1, cols 2-3) = A11*B12 + A21*B22
    terms.extend(strassen_2x2x2(0, 0, 2, 2))
    # C21 (rows 2-3, cols 0-1) = A12*B11 + A22*B21
    terms.extend(strassen_2x2x2(2, 2, 0, 0))
    # C22 (rows 2-3, cols 2-3) = A12*B12 + A22*B22
    terms.extend(strassen_2x2x2(2, 2, 0, 2))
    
    # Part 2: 4x4x2 (columns 4-5 of B/C)
    # C_top(2x2) = A1(2x2)*B1(2x2) + A2(2x2)*B2(2x2) for rows 0-1
    # C_bot(2x2) = A3(2x2)*B1(2x2) + A4(2x2)*B2(2x2) for rows 2-3
    # where B1 is rows 0-1 of B, B2 is rows 2-3 of B
    
    # C_top (rows 0-1, cols 4-5) = A11(2x2)*B1(2x2) + A12(2x2)*B2(2x2)
    terms.extend(strassen_2x2x2(0, 0, 2, 4))
    # C_bot (rows 2-3, cols 4-5) = A21(2x2)*B1(2x2) + A22(2x2)*B2(2x2)
    terms.extend(strassen_2x2x2(2, 0, 2, 4))
    
    # Part 3: 4x4x1 (column 6 of B/C)
    # Schoolbook: 16 terms
    for i in range(n):
        for j in range(m):
            u = [0] * (n * m)
            u[i * m + j] = 1
            v = [0] * (m * p)
            v[j * p + 6] = 1
            w = [0] * (p * n)
            w[6 * n + i] = 1
            terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
