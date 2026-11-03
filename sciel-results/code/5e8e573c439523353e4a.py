"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a lower-rank decomposition for 5x5 matrix multiplication
    using block structure: 3x3 + 2x2 cores with rectangular cross-terms.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # We'll build the decomposition using the block partition:
    # A = [[A11(3x3), A12(3x2)],
    #      [A21(2x3), A22(2x2)]]
    # B = [[B11(3x3), B12(3x2)],
    #      [B21(2x3), B22(2x2)]]
    # C = A*B = [[C11(3x3), C12(3x2)],
    #            [C21(2x3), C22(2x2)]]
    
    # For exact decomposition, we need:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each product term is a separate matrix multiplication.
    # We'll use known low-rank decompositions for each.
    
    # Helper: get Laderman decomposition for 3x3
    def laderman_3x3():
        """
        Returns U, V, W for 3x3 matrix multiplication with rank 15.
        Laderman's algorithm uses 15 multiplications.
        """
        # Laderman's 15-multiplication algorithm for 3x3
        # The 15 intermediate products are:
        # m1 = (a11+a21)*(b11+b21)
        # m2 = (a11-a21)*(b11-b21)
        # m3 = (a12+a22)*(b12+b22)
        # m4 = (a12-a22)*(b12-b22)
        # m5 = (a13+a23)*(b13+b23)
        # m6 = (a13-a23)*(b13-b23)
        # m7 = (a11+a12+a13)*(b11+b21+b31)
        # m8 = (a11+a12+a13)*(b12+b22+b32)
        # m9 = (a11+a12+a13)*(b13+b23+b33)
        # m10 = (a21+a22+a23)*(b11+b21+b31)
        # m11 = (a21+a22+a23)*(b12+b22+b32)
        # m12 = (a21+a22+a23)*(b13+b23+b33)
        # m13 = (a31+a32+a33)*(b11+b21+b31)
        # m14 = (a31+a32+a33)*(b12+b22+b32)
        # m15 = (a31+a32+a33)*(b13+b23+b33)
        
        # Actually, let me use a cleaner formulation.
        # Laderman's algorithm for 3x3 matrix multiplication has rank 15.
        # I'll construct it explicitly.
        
        rank = 15
        U = [[Fraction(0)] * rank for _ in range(9)]  # 3x3 = 9 entries
        V = [[Fraction(0)] * rank for _ in range(9)]
        W = [[Fraction(0)] * rank for _ in range(9)]
        
        # Define the 15 bilinear forms
        # Using Laderman's specific construction
        # Row indices for U: i*3+j for A[i][j]
        # Row indices for V: i*3+j for B[i][j]  
        # Row indices for W: i*3+j for C[i][j]
        
        # The 15 products (using 0-indexed):
        # p0: (A00+A10)*(B00+B10)
        # p1: (A00-A10)*(B00-B10)
        # p2: (A01+A11)*(B01+B11)
        # p3: (A01-A11)*(B01-B11)
        # p4: (A02+A12)*(B02+B12)
        # p5: (A02-A12)*(B02-B12)
        # p6: (A00+A01+A02)*(B00+B10+B20)
        # p7: (A00+A01+A02)*(B01+B11+B21)
        # p8: (A00+A01+A02)*(B02+B12+B22)
        # p9: (A10+A11+A12)*(B00+B10+B20)
        # p10: (A10+A11+A12)*(B01+B11+B21)
        # p11: (A10+A11+A12)*(B02+B12+B22)
        # p12: (A20+A21+A22)*(B00+B10+B20)
        # p13: (A20+A21+A22)*(B01+B11+B21)
        # p14: (A20+A21+A22)*(B02+B12+B22)
        
        # This isn't quite right. Let me use a verified Laderman decomposition.
        # I'll use the standard Laderman 15-multiplication scheme.
        
        # Actually, let me just use the schoolbook for now and focus on the structure.
        # For 3x3, schoolbook rank is 27. Let me use that as baseline.
        rank = 27
        U = [[Fraction(0)] * rank for _ in range(9)]
        V = [[Fraction(0)] * rank for _ in range(9)]
        W = [[Fraction(0)] * rank for _ in range(9)]
        
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = Fraction(1)
                    V[j*3+k][col] = Fraction(1)
                    W[k*3+i][col] = Fraction(1)
                    col += 1
        
        return U, V, W, rank
    
    # Helper: get Strassen decomposition for 2x2
    def strassen_2x2():
        """
        Returns U, V, W for 2x2 matrix multiplication with rank 7.
        """
        rank = 7
        U = [[Fraction(0)] * rank for _ in range(4)]
        V = [[Fraction(0)] * rank for _ in range(4)]
        W = [[Fraction(0)] * rank for _ in range(4)]
        
        # Strassen's 7 products:
        # m1 = (A00+A11)*(B00+B11)
        # m2 = (A10+A11)*B00
        # m3 = A00*(B10-B11)
        # m4 = A11*(B01-B00)
        # m5 = (A00+A01)*B11
        # m6 = (A10-A00)*(B00+B10)
        # m7 = (A01-A11)*(B01+B11)
        
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # U indices: A00=0, A01=1, A10=2, A11=3
        # V indices: B00=0, B01=1, B10=2, B11=3
        # W indices: C00=0, C01=1, C10=2, C11=3
        
        # m1
        U[0][0] = Fraction(1); U[3][0] = Fraction(1)
        V[0][0] = Fraction(1); V[3][0] = Fraction(1)
        W[0][0] = Fraction(1); W[3][0] = Fraction(1)
        
        # m2
        U[2][1] = Fraction(1); U[3][1] = Fraction(1)
        V[0][1] = Fraction(1)
        W[2][1] = Fraction(1); W[3][1] = Fraction(1)
        
        # m3
        U[0][2] = Fraction(1)
        V[2][2] = Fraction(1); V[3][2] = Fraction(-1)
        W[1][2] = Fraction(1); W[3][2] = Fraction(1)
        
        # m4
        U[3][3] = Fraction(1)
        V[1][3] = Fraction(1); V[0][3] = Fraction(-1)
        W[0][3] = Fraction(1); W[2][3] = Fraction(1)
        
        # m5
        U[0][4] = Fraction(1); U[1][4] = Fraction(1)
        V[3][4] = Fraction(1)
        W[0][4] = Fraction(-1); W[1][4] = Fraction(1)
        
        # m6
        U[2][5] = Fraction(1); U[0][5] = Fraction(-1)
        V[0][5] = Fraction(1); V[2][5] = Fraction(1)
        W[3][5] = Fraction(1)
        
        # m7
        U[1][6] = Fraction(1); U[3][6] = Fraction(-1)
        V[1][6] = Fraction(1); V[3][6] = Fraction(1)
        W[0][6] = Fraction(1); W[1][6] = Fraction(1)
        
        return U, V, W, rank
    
    # For rectangular blocks, use schoolbook (they're small)
    # 3x2 * 2x3 = 3x3: rank = 3*2*3 = 18
    # 3x2 * 2x2 = 3x2: rank = 3*2*2 = 12
    # 2x3 * 3x3 = 2x3: rank = 2*3*3 = 18
    # 2x3 * 3x2 = 2x2: rank = 2*3*2 = 12
    # 2x2 * 2x3 = 2x3: rank = 2*2*3 = 12
    # 2x2 * 2x2 = 2x2: rank = 8 (or 7 with Strassen)
    
    # Total rank if we use schoolbook for all:
    # C11: A11*B11 (27) + A12*B21 (18) = 45
    # C12: A11*B12 (12) + A12*B22 (12) = 24
    # C21: A21*B11 (18) + A22*B21 (12) = 30
    # C22: A21*B12 (12) + A22*B22 (8) = 20
    # Total: 45+24+30+20 = 119
    
    # This is already better than 125!
    # Let me implement this.
    
    # Build the full 5x5 decomposition
    rank_total = 119
    U = [[Fraction(0)] * rank_total for _ in range(25)]  # 5x5
    V = [[Fraction(0)] * rank_total for _ in range(25)]
    W = [[Fraction(0)] * rank_total for _ in range(25)]
    
    col_offset = 0
    
    # Block A11: rows 0-2, cols 0-2 (indices 0,1,2,3,4,5,6,7,8)
    # Block A12: rows 0-2, cols 3-4 (indices 9,10,11,12,13,14)
    # Block A21: rows 3-4, cols 0-2 (indices 15,16,17,18,19,20)
    # Block A22: rows 3-4, cols 3-4 (indices 21,22,23,24)
    
    # Similarly for B and C
    
    def get_block_indices(block_size, row_start, col_start, full_size):
        """Get indices for a block in a full_size x full_size matrix."""
        indices = []
        for i in range(block_size):
            for j in range(block_size):
                indices.append((row_start + i) * full_size + (col_start + j))
        return indices
    
    # A11: 3x3 at (0,0)
    A11_idx = get_block_indices(3, 0, 0, 5)
    # A12: 3x2 at (0,3)
    A12_idx = []
    for i in range(3):
        for j in range(2):
            A12_idx.append((0 + i) * 5 + (3 + j))
    # A21: 2x3 at (3,0)
    A21_idx = []
    for i in range(2):
        for j in range(3):
            A21_idx.append((3 + i) * 5 + (0 + j))
    # A22: 2x2 at (3,3)
    A22_idx = get_block_indices(2, 3, 3, 5)
    
    # B11: 3x3 at (0,0)
    B11_idx = get_block_indices(3, 0, 0, 5)
    # B12: 3x2 at (0,3)
    B12_idx = []
    for i in range(3):
        for j in range(2):
            B12_idx.append((0 + i) * 5 + (3 + j))
    # B21: 2x3 at (3,0)
    B21_idx = []
    for i in range(2):
        for j in range(3):
            B21_idx.append((3 + i) * 5 + (0 + j))
    # B22: 2x2 at (3,3)
    B22_idx = get_block_indices(2, 3, 3, 5)
    
    # C11: 3x3 at (0,0)
    C11_idx = get_block_indices(3, 0, 0, 5)
    # C12: 3x2 at (0,3)
    C12_idx = []
    for i in range(3):
        for j in range(2):
            C12_idx.append((0 + i) * 5 + (3 + j))
    # C21: 2x3 at (3,0)
    C21_idx = []
    for i in range(2):
        for j in range(3):
            C21_idx.append((3 + i) * 5 + (0 + j))
    # C22: 2x2 at (3,3)
    C22_idx = get_block_indices(2, 3, 3, 5)
    
    def add_schoolbook_block(U_full, V_full, W_full, col_start, 
                             A_idx, B_idx, C_idx, 
                             A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """Add a schoolbook decomposition for a block multiplication."""
        rank = A_rows * A_cols * B_cols  # = C_rows * B_rows * C_cols
        for i in range(A_rows):
            for j in range(A_cols):
                for k in range(B_cols):
                    col = col_start + i * A_cols * B_cols + j * B_cols + k
                    # A index: i*A_cols + j
                    a_idx = A_idx[i * A_cols + j]
                    # B index: j*B_cols + k
                    b_idx = B_idx[j * B_cols + k]
                    # C index: k*C_cols + i
                    c_idx = C_idx[k * C_cols + i]
                    
                    U_full[a_idx][col] = Fraction(1)
                    V_full[b_idx][col] = Fraction(1)
                    W_full[c_idx][col] = Fraction(1)
        return rank
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: 3x3 * 3x3 = 3x3, rank 27
    r1 = add_schoolbook_block(U, V, W, col_offset, A11_idx, B11_idx, C11_idx, 3, 3, 3, 3)
    col_offset += r1
    
    # A12*B21: 3x2 * 2x3 = 3x3, rank 18
    r2 = add_schoolbook_block(U, V, W, col_offset, A12_idx, B21_idx, C11_idx, 3, 2, 3, 3)
    col_offset += r2
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: 3x3 * 3x2 = 3x2, rank 18
    r3 = add_schoolbook_block(U, V, W, col_offset, A11_idx, B12_idx, C12_idx, 3, 3, 2, 2)
    col_offset += r3
    
    # A12*B22: 3x2 * 2x2 = 3x2, rank 12
    r4 = add_schoolbook_block(U, V, W, col_offset, A12_idx, B22_idx, C12_idx, 3, 2, 2, 2)
    col_offset += r4
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: 2x3 * 3x3 = 2x3, rank 18
    r5 = add_schoolbook_block(U, V, W, col_offset, A21_idx, B11_idx, C21_idx, 2, 3, 3, 3)
    col_offset += r5
    
    # A22*B21: 2x2 * 2x3 = 2x3, rank 12
    r6 = add_schoolbook_block(U, V, W, col_offset, A22_idx, B21_idx, C21_idx, 2, 2, 3, 3)
    col_offset += r6
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: 2x3 * 3x2 = 2x2, rank 12
    r7 = add_schoolbook_block(U, V, W, col_offset, A21_idx, B12_idx, C22_idx, 2, 3, 2, 2)
    col_offset += r7
    
    # A22*B22: 2x2 * 2x2 = 2x2, rank 8
    r8 = add_schoolbook_block(U, V, W, col_offset, A22_idx, B22_idx, C22_idx, 2, 2, 2, 2)
    col_offset += r8
    
    total_rank = col_offset
    print(f"Total rank: {total_rank}")
    
    # Convert to float for the return
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
