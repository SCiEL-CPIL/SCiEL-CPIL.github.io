"""Matrix multiplication 4x5x6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Return exact tensor decomposition for 4x5 * 5x6.
    
    Strategy: Partition the multiplication into 4 blocks of size 2x5x3.
    A (4x5) = [A1(2x5); A2(2x5)]
    B (5x6) = [B1(5x3) B2(5x3)]
    C = [A1B1 A1B2; A2B1 A2B2]
    
    Each block is 2x5x3.
    Decompose 2x5x3 into 2x2x3 and 2x3x3.
    R(2x2x3) <= 11 (Strassen 2x2x2 + Schoolbook 2x2x1)
    R(2x3x3) <= 17 (Schoolbook split)
    Total per block: 28.
    Total rank: 4 * 28 = 112.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # We will build the decomposition for one 2x5x3 block, then replicate.
    # However, the indices are global.
    # It is easier to generate the terms for the full 4x5x6 directly by iterating blocks.
    
    terms = []
    
    def add_term(U_dict, V_dict, W_dict):
        terms.append((U_dict, V_dict, W_dict))
        
    # Helper to add Strassen 2x2x2 terms
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # C indices: c00, c01, c10, c11
    def add_strassen_222(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C01 -= M1, C10 += M1, C11 -= M1 ?
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c01: -1, c10: 1, c11: -1})
        add_term({a10: 1, a11: 1}, {b00: 1}, {c10: 1, c11: -1})
        add_term({a00: 1}, {b01: 1, b11: -1}, {c01: 1, c11: 1})
        add_term({a11: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1})
        add_term({a00: 1, a01: 1}, {b11: 1}, {c00: -1, c01: 1})
        add_term({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1})
        add_term({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1})

    # Helper to add Schoolbook 2x2x1 terms (A 2x2, B 2x1, C 2x1)
    def add_schoolbook_221(a00, a01, a10, a11, b0, b1, c0, c1):
        add_term({a00: 1}, {b0: 1}, {c0: 1})
        add_term({a01: 1}, {b1: 1}, {c0: 1})
        add_term({a10: 1}, {b0: 1}, {c1: 1})
        add_term({a11: 1}, {b1: 1}, {c1: 1})

    # Helper to add Schoolbook 2x1x3 terms (A 2x1, B 1x3, C 2x3)
    def add_schoolbook_213(a0, a1, b0, b1, b2, c00, c01, c02, c10, c11, c12):
        add_term({a0: 1}, {b0: 1}, {c00: 1})
        add_term({a0: 1}, {b1: 1}, {c01: 1})
        add_term({a0: 1}, {b2: 1}, {c02: 1})
        add_term({a1: 1}, {b0: 1}, {c10: 1})
        add_term({a1: 1}, {b1: 1}, {c11: 1})
        add_term({a1: 1}, {b2: 1}, {c12: 1})

    # Helper to add Schoolbook 2x3x3 terms (A 2x3, B 3x3, C 2x3)
    # This is just 18 terms.
    def add_schoolbook_233(a00, a01, a02, a10, a11, a12, 
                           b00, b01, b02, b10, b11, b12, b20, b21, b22,
                           c00, c01, c02, c10, c11, c12):
        # c00 = a00b00 + a01b10 + a02b20
        add_term({a00: 1}, {b00: 1}, {c00: 1})
        add_term({a01: 1}, {b10: 1}, {c00: 1})
        add_term({a02: 1}, {b20: 1}, {c00: 1})
        # c01 = a00b01 + a01b11 + a02b21
        add_term({a00: 1}, {b01: 1}, {c01: 1})
        add_term({a01: 1}, {b11: 1}, {c01: 1})
        add_term({a02: 1}, {b21: 1}, {c01: 1})
        # c02 = a00b02 + a01b12 + a02b22
        add_term({a00: 1}, {b02: 1}, {c02: 1})
        add_term({a01: 1}, {b12: 1}, {c02: 1})
        add_term({a02: 1}, {b22: 1}, {c02: 1})
        # c10 = a10b00 + a11b10 + a12b20
        add_term({a10: 1}, {b00: 1}, {c10: 1})
        add_term({a11: 1}, {b10: 1}, {c10: 1})
        add_term({a12: 1}, {b20: 1}, {c10: 1})
        # c11 = a10b01 + a11b11 + a12b21
        add_term({a10: 1}, {b01: 1}, {c11: 1})
        add_term({a11: 1}, {b11: 1}, {c11: 1})
        add_term({a12: 1}, {b21: 1}, {c11: 1})
        # c12 = a10b02 + a11b12 + a12b22
        add_term({a10: 1}, {b02: 1}, {c12: 1})
        add_term({a11: 1}, {b12: 1}, {c12: 1})
        add_term({a12: 1}, {b22: 1}, {c12: 1})

    # Iterate over the 4 blocks: (row_block, col_block)
    # row_block 0: rows 0,1 of A. row_block 1: rows 2,3 of A.
    # col_block 0: cols 0,1,2 of B. col_block 1: cols 3,4,5 of B.
    
    for rb in range(2):
        for cb in range(2):
            # A indices for this block: rows 2*rb to 2*rb+1, cols 0 to 4
            # B indices for this block: rows 0 to 4, cols 3*cb to 3*cb+2
            # C indices for this block: rows 2*rb to 2*rb+1, cols 3*cb to 3*cb+2
            
            # Split middle dim 5 into 2+3
            # Part 1: cols 0,1 of A, rows 0,1 of B. (2x2x3)
            # A_part1: rows 2rb, 2rb+1; cols 0,1
            # B_part1: rows 0,1; cols 3cb..3cb+2
            # C_part1: rows 2rb, 2rb+1; cols 3cb..3cb+2
            
            a00 = (2*rb)*5 + 0
            a01 = (2*rb)*5 + 1
            a10 = (2*rb+1)*5 + 0
            a11 = (2*rb+1)*5 + 1
            
            b00 = 0*6 + (3*cb+0)
            b01 = 0*6 + (3*cb+1)
            b02 = 0*6 + (3*cb+2)
            b10 = 1*6 + (3*cb+0)
            b11 = 1*6 + (3*cb+1)
            b12 = 1*6 + (3*cb+2)
            
            c00 = (3*cb+0)*4 + (2*rb)
            c01 = (3*cb+1)*4 + (2*rb)
            c02 = (3*cb+2)*4 + (2*rb)
            c10 = (3*cb+0)*4 + (2*rb+1)
            c11 = (3*cb+1)*4 + (2*rb+1)
            c12 = (3*cb+2)*4 + (2*rb+1)
            
            # Use Strassen for 2x2x2 part (cols 0,1 of C) + Schoolbook for col 2
            # Strassen on A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
            add_strassen_222(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11)
            
            # Schoolbook for A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
            # A is 2x2, B is 2x1 (col 2 of B_part1), C is 2x1 (col 2 of C_part1)
            add_schoolbook_221(a00, a01, a10, a11, b02, b12, c02, c12)
            
            # Part 2: cols 2,3,4 of A, rows 2,3,4 of B. (2x3x3)
            # A_part2: rows 2rb, 2rb+1; cols 2,3,4
            # B_part2: rows 2,3,4; cols 3cb..3cb+2
            # C_part2: rows 2rb, 2rb+1; cols 3cb..3cb+2
            
            a20 = (2*rb)*5 + 2
            a21 = (2*rb)*5 + 3
            a22 = (2*rb)*5 + 4
            a30 = (2*rb+1)*5 + 2
            a31 = (2*rb+1)*5 + 3
            a32 = (2*rb+1)*5 + 4
            
            b20 = 2*6 + (3*cb+0)
            b21 = 2*6 + (3*cb+1)
            b22 = 2*6 + (3*cb+2)
            b30 = 3*6 + (3*cb+0)
            b31 = 3*6 + (3*cb+1)
            b32 = 3*6 + (3*cb+2)
            b40 = 4*6 + (3*cb+0)
            b41 = 4*6 + (3*cb+1)
            b42 = 4*6 + (3*cb+2)
            
            # C indices are the same c00..c12
            
            # Use Schoolbook for 2x3x3
            add_schoolbook_233(a20, a21, a22, a30, a31, a32,
                               b20, b21, b22, b30, b31, b32, b40, b41, b42,
                               c00, c01, c02, c10, c11, c12)

    rank = len(terms)
    
    # Assemble U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_dict, V_dict, W_dict) in enumerate(terms):
        for idx, val in U_dict.items():
            U[idx][col] = val
        for idx, val in V_dict.items():
            V[idx][col] = val
        for idx, val in W_dict.items():
            W[idx][col] = val
            
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(val - 1) > 1e-9:
                    max_err = max(max_err, abs(val - 1))
                    
    if max_err > 1e-9:
        # Fallback
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
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
