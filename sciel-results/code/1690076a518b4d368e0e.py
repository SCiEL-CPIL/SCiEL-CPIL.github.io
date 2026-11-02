"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition:
    # Split B (3x7) into B1 (3x3), B2 (3x2), B3 (3x2)
    # M_233 rank 17, M_232 rank 11, M_232 rank 11
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        if coeff != 0:
            U[u_idx][col] += coeff
            V[v_idx][col] += coeff
            W[w_idx][col] += coeff
        col += 1

    # Indices for A (2x3): u_idx = i*3 + j
    # Indices for B (3x7): v_idx = j*7 + k
    # Indices for C (2x7): w_idx = k*2 + i
    
    # --- Part 1: A (2x3) x B1 (3x3) ---
    # B1 uses columns 0, 1, 2 of B.
    # Decompose into A1 (2x2) x B11 (2x2) [Strassen] + A1 (2x2) x b12 (2x1) [MatVec] + a3 (2x1) x b3 (1x3) [Outer]
    
    # 1. A1 (2x2) x B11 (2x2) using Strassen (Rank 7)
    # A1: rows 0,1; cols 0,1. Indices: 0,1,3,4
    # B11: rows 0,1; cols 0,1. Indices: 0,1,7,8
    # C: cols 0,1. Rows 0,1. Indices: 0,1,2,3
    
    # M1 = (a00+a11)(b00+b11)
    # a00=0, a11=4, b00=0, b11=8
    # (0+4) is not a single index. We need to expand.
    # This approach of adding terms one by one with coefficients is tricky for Strassen because the factors are sums.
    # We must expand Strassen into rank-1 terms? No, Strassen is already a decomposition.
    # The U, V, W matrices for Strassen have entries that are sums of A/B entries.
    # So U and V will have multiple non-zeros in a column.
    
    # Let's reset the helper to handle columns properly.
    # I will manually construct the columns for the 39 terms.
    
    # It is safer to build the U, V, W by iterating through the logical terms.
    
    # Re-initialize col
    col = 0
    
    # --- Helper to add a Strassen block ---
    def add_strassen_222(a_base, b_base, c_base_col_offset):
        # a_base: (i, j) base for 2x2 A
        # b_base: (j, k) base for 2x2 B
        # c_base_col_offset: k offset for C
        nonlocal col
        # A indices: (a_base_i, a_base_j), (a_base_i, a_base_j+1), (a_base_i+1, a_base_j), (a_base_i+1, a_base_j+1)
        # B indices: (b_base_i, b_base_k), ...
        
        a00 = a_base[0]*m + a_base[1]
        a01 = a_base[0]*m + a_base[1] + 1
        a10 = (a_base[0]+1)*m + a_base[1]
        a11 = (a_base[0]+1)*m + a_base[1] + 1
        
        b00 = b_base[0]*p + b_base[1]
        b01 = b_base[0]*p + b_base[1] + 1
        b10 = (b_base[0]+1)*p + b_base[1]
        b11 = (b_base[0]+1)*p + b_base[1] + 1
        
        c00 = (b_base[1] + c_base_col_offset)*n + a_base[0]
        c01 = (b_base[1] + c_base_col_offset + 1)*n + a_base[0]
        c10 = (b_base[1] + c_base_col_offset)*n + a_base[0] + 1
        c11 = (b_base[1] + c_base_col_offset + 1)*n + a_base[0] + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c01][col] = 1; W[c10][col] = 1; W[c11][col] = 1
        # Correction: W coefficients for M1
        # C00 += M1, C01 += 0, C10 += 0, C11 += M1
        # Wait, Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M1:
        W[c00][col] = 1
        W[c01][col] = 0
        W[c10][col] = 0
        W[c11][col] = 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c00][col] = 0
        W[c01][col] = 0
        W[c10][col] = 1
        W[c11][col] = -1
        col += 1
        
        # M3 = a00(b01-b11)
        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c00][col] = 0
        W[c01][col] = 1
        W[c10][col] = 0
        W[c11][col] = 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[c00][col] = 1
        W[c01][col] = 0
        W[c10][col] = 1
        W[c11][col] = 0
        col += 1
        
        # M5 = (a00+a01)b11
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1
        W[c01][col] = 1
        W[c10][col] = 0
        W[c11][col] = 0
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c00][col] = 0
        W[c01][col] = 0
        W[c10][col] = 0
        W[c11][col] = 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c00][col] = 1
        W[c01][col] = 0
        W[c10][col] = 0
        W[c11][col] = 0
        col += 1

    # --- Part 1: A (2x3) x B1 (3x3) ---
    # A1 is A[:, 0:2], B11 is B[0:2, 0:2]
    add_strassen_222((0, 0), (0, 0), 0)
    
    # A1 (2x2) x b12 (2x1) where b12 is B[0:2, 2]
    # This is MatVec. 4 terms.
    # C col 2 (index 2 in B, so k=2)
    # C0,2 = a00*b02 + a01*b12
    # C1,2 = a10*b02 + a11*b12
    # b02 index: 0*7+2 = 2
    # b12 index: 1*7+2 = 9
    
    # Term 1: a00 * b02
    U[0][col] = 1; V[2][col] = 1; W[2*2+0][col] = 1; col += 1
    # Term 2: a01 * b12
    U[1][col] = 1; V[9][col] = 1; W[2*2+0][col] = 1; col += 1
    # Term 3: a10 * b02
    U[3][col] = 1; V[2][col] = 1; W[2*2+1][col] = 1; col += 1
    # Term 4: a11 * b12
    U[4][col] = 1; V[9][col] = 1; W[2*2+1][col] = 1; col += 1
    
    # a3 (2x1) x b3 (1x3) where a3 is A[:, 2], b3 is B[2, 0:3]
    # a3 indices: 2, 5
    # b3 indices: 14, 15, 16
    # C cols 0,1,2
    
    # Term 5: a02 * b20
    U[2][col] = 1; V[14][col] = 1; W[0*2+0][col] = 1; col += 1
    # Term 6: a02 * b21
    U[2][col] = 1; V[15][col] = 1; W[1*2+0][col] = 1; col += 1
    # Term 7: a02 * b22
    U[2][col] = 1; V[16][col] = 1; W[2*2+0][col] = 1; col += 1
    # Term 8: a12 * b20
    U[5][col] = 1; V[14][col] = 1; W[0*2+1][col] = 1; col += 1
    # Term 9: a12 * b21
    U[5][col] = 1; V[15][col] = 1; W[1*2+1][col] = 1; col += 1
    # Term 10: a12 * b22
    U[5][col] = 1; V[16][col] = 1; W[2*2+1][col] = 1; col += 1
    
    # --- Part 2: A (2x3) x B2 (3x2) ---
    # B2 is B[:, 3:5]
    # Decompose into A1 (2x2) x B21 (2x2) [Strassen] + a3 (2x1) x b3 (1x2) [Outer]
    
    # A1 is A[:, 0:2], B21 is B[0:2, 3:5]
    add_strassen_222((0, 0), (0, 3), 3)
    
    # a3 (2x1) x b3 (1x2) where a3 is A[:, 2], b3 is B[2, 3:5]
    # a3 indices: 2, 5
    # b3 indices: 21, 22
    
    # Term 11: a02 * b23
    U[2][col] = 1; V[21][col] = 1; W[3*2+0][col] = 1; col += 1
    # Term 12: a02 * b24
    U[2][col] = 1; V[22][col] = 1; W[4*2+0][col] = 1; col += 1
    # Term 13: a12 * b23
    U[5][col] = 1; V[21][col] = 1; W[3*2+1][col] = 1; col += 1
    # Term 14: a12 * b24
    U[5][col] = 1; V[22][col] = 1; W[4*2+1][col] = 1; col += 1
    
    # --- Part 3: A (2x3) x B3 (3x2) ---
    # B3 is B[:, 5:7]
    # Decompose into A1 (2x2) x B31 (2x2) [Strassen] + a3 (2x1) x b3 (1x2) [Outer]
    
    # A1 is A[:, 0:2], B31 is B[0:2, 5:7]
    add_strassen_222((0, 0), (0, 5), 5)
    
    # a3 (2x1) x b3 (1x2) where a3 is A[:, 2], b3 is B[2, 5:7]
    # a3 indices: 2, 5
    # b3 indices: 19, 20
    
    # Term 15: a02 * b25
    U[2][col] = 1; V[19][col] = 1; W[5*2+0][col] = 1; col += 1
    # Term 16: a02 * b26
    U[2][col] = 1; V[20][col] = 1; W[6*2+0][col] = 1; col += 1
    # Term 17: a12 * b25
    U[5][col] = 1; V[19][col] = 1; W[5*2+1][col] = 1; col += 1
    # Term 18: a12 * b26
    U[5][col] = 1; V[20][col] = 1; W[6*2+1][col] = 1; col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
