import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def _verify_exact(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Convert to Fractions for exact arithmetic
    U_frac = [[Fraction(str(x)) for x in row] for row in U]
    V_frac = [[Fraction(str(x)) for x in row] for row in V]
    W_frac = [[Fraction(str(x)) for x in row] for row in W]
    
    # Check all entries
    for a in range(n * m):
        i, j = a // m, a % m
        for b in range(m * p):
            j2, k = b // p, b % p
            for c in range(p * n):
                k2, i2 = c // n, c % n
                # Expected value
                if j == j2 and k == k2 and i == i2:
                    expected = 1
                else:
                    expected = 0
                
                # Computed value
                computed = 0
                for r in range(rank):
                    computed += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                
                if computed != expected:
                    return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120

    # Strategy: Construct a rank 46 decomposition using Strassen for 2x4x4 part
    # and schoolbook for the remaining parts.
    
    # Dimensions: A (2x5), B (5x5), C (2x5)
    # Split A into A_left (2x4) and A_right (2x1)
    # Split B into B_tl (4x4), B_tr (4x1), B_bl (1x4), B_br (1x1)
    
    # C_left (2x4) = A_left B_tl + A_right B_bl
    # C_right (2x1) = A_left B_tr + A_right B_br
    
    # Rank(A_left B_tl) = Rank(2x4x4) = 28 (using 4 Strassen 2x2x2)
    # Rank(A_left B_tr) = Rank(2x4x1) = 8 (Schoolbook)
    # Rank(A_right B_bl) = Rank(2x1x4) = 8 (Schoolbook)
    # Rank(A_right B_br) = Rank(2x1x1) = 2 (Schoolbook)
    # Total Rank = 28 + 8 + 8 + 2 = 46
    
    # We will build U, V, W lists.
    # U shape: (2*5, 46) = (10, 46)
    # V shape: (5*5, 46) = (25, 46)
    # W shape: (5*2, 46) = (10, 46)
    
    rank_total = 46
    U = [[0.0] * rank_total for _ in range(n * m)]
    V = [[0.0] * rank_total for _ in range(m * p)]
    W = [[0.0] * rank_total for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    # a_indices: indices in A (2x2 block)
    # b_indices: indices in B (2x2 block)
    # c_indices: indices in C (2x2 block)
    # The block is A_block (2x2) * B_block (2x2) = C_block (2x2)
    # A_block indices in global A: (r, c) -> a_idx = r*5 + c
    # B_block indices in global B: (r, c) -> b_idx = r*5 + c
    # C_block indices in global C: (r, c) -> c_idx = r*5 + c (Note: C is 2x5, so r in 0..1, c in 0..4)
    
    def add_strassen_2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        nonlocal col_offset
        # A block is 2x2: rows a_row_start, a_row_start+1; cols a_col_start, a_col_start+1
        # B block is 2x2: rows b_row_start, b_row_start+1; cols b_col_start, b_col_start+1
        # C block is 2x2: rows c_row_start, c_row_start+1; cols c_col_start, c_col_start+1
        
        # Strassen formulas for 2x2 matrices:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to U, V, W contributions.
        # Each M is a scalar multiplication in the tensor sense?
        # No, M is a 2x2 matrix result.
        # Wait, the "rank" of 2x2x2 is 7.
        # The 7 terms correspond to 7 scalar multiplications.
        # The tensors U, V, W for these 7 terms are fixed.
        
        # Let's define the 7 Strassen terms for a single 2x2x2 block.
        # The inputs are 4 scalars from A, 4 from B. Output 4 scalars to C.
        # The standard Strassen decomposition uses 7 multiplications.
        # We need the U, V, W vectors for these 7 multiplications.
        
        # Term 1: (A11+A22)(B11+B22)
        # U part: A11=1, A22=1, others 0.
        # V part: B11=1, B22=1, others 0.
        # W part: C11=1, C22=1, C12=-1, C21=-1? 
        # C11 = M1 + ...
        # C22 = M1 - ...
        # So W11=1, W22=1, W12=0, W21=0 for M1?
        # Let's check the reconstruction:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for M1:
        # W11 += 1, W22 += 1.
        
        # We will define the 7 (U, V, W) patterns for a local 2x2x2 block.
        # Local indices: A(0,0), A(0,1), A(1,0), A(1,1)
        # Local indices: B(0,0), B(0,1), B(1,0), B(1,1)
        # Local indices: C(0,0), C(0,1), C(1,0), C(1,1)
        
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U: [A11, A12, A21, A22]
            # V: [B11, B12, B21, B22]
            # W: [C11, C12, C21, C22]
            
            # M1 = (A11+A22)(B11+B22)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            
            # M2 = (A21+A22)B11
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # C21 += M2, C22 -= M2
            
            # M3 = A11(B12-B22)
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # C12 += M3, C22 += M3
            
            # M4 = A22(B21-B11)
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # C11 += M4, C21 += M4
            
            # M5 = (A11+A12)B22
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # C11 -= M5, C12 += M5
            
            # M6 = (A21-A11)(B11+B12)
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # C22 += M6
            
            # M7 = (A12-A22)(B21+B22)
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # C11 += M7
        ]
        
        for u_coeff, v_coeff, w_coeff in strassen_terms:
            # Map local indices to global
            # A local (r,c) -> global (a_row_start+r, a_col_start+c)
            # A index = (a_row_start+r)*5 + (a_col_start+c)
            # B local (r,c) -> global (b_row_start+r, b_col_start+c)
            # B index = (b_row_start+r)*5 + (b_col_start+c)
            # C local (r,c) -> global (c_row_start+r, c_col_start+c)
            # C index = (c_row_start+r)*5 + (c_col_start+c)
            
            for r in range(2):
                for c in range(2):
                    val = u_coeff[r*2+c]
                    if val != 0:
                        idx_a = (a_row_start + r) * 5 + (a_col_start + c)
                        U[idx_a][col_offset] += val
                    
                    val = v_coeff[r*2+c]
                    if val != 0:
                        idx_b = (b_row_start + r) * 5 + (b_col_start + c)
                        V[idx_b][col_offset] += val
                    
                    val = w_coeff[r*2+c]
                    if val != 0:
                        idx_c = (c_row_start + r) * 5 + (c_col_start + c)
                        W[idx_c][col_offset] += val
            
            col_offset += 1
    
    # 1. A_left (2x4) * B_tl (4x4)
    # Split B_tl into four 2x2 blocks:
    # B_tl[0:2, 0:2] -> B00
    # B_tl[0:2, 2:4] -> B01
    # B_tl[2:4, 0:2] -> B10
    # B_tl[2:4, 2:4] -> B11
    
    # C_left[0:2, 0:2] = A_left[0:2, 0:2] * B00 + A_left[0:2, 2:4] * B10
    # Wait, A_left is 2x4.
    # A_left = [A_00 A_01] where A_00 is 2x2, A_01 is 2x2.
    # B_tl = [[B00, B01], [B10, B11]]
    # C_left[:, 0:2] = A_00 B00 + A_01 B10
    # C_left[:, 2:4] = A_00 B01 + A_01 B11
    
    # We need to compute 4 products of 2x2 matrices:
    # P1 = A_00 B00
    # P2 = A_01 B10
    # P3 = A_00 B01
    # P4 = A_01 B11
    
    # Each product is 2x2x2. Rank 7.
    
    # P1: A_00 (rows 0,1; cols 0,1) * B00 (rows 0,1; cols 0,1) -> C_left (rows 0,1; cols 0,1)
    add_strassen_2x2(0, 0, 0, 0, 0, 0)
    
    # P2: A_01 (rows 0,1; cols 2,3) * B10 (rows 2,3; cols 0,1) -> C_left (rows 0,1; cols 0,1)
    # Note: C_left[:, 0:2] gets contribution from P2.
    # So we add to the same C columns.
    add_strassen_2x2(0, 2, 2, 0, 0, 0)
    
    # P3: A_00 (rows 0,1; cols 0,1) * B01 (rows 0,1; cols 2,3) -> C_left (rows 0,1; cols 2,3)
    add_strassen_2x2(0, 0, 0, 2, 0, 2)
    
    # P4: A_01 (rows 0,1; cols 2,3) * B11 (rows 2,3; cols 2,3) -> C_left (rows 0,1; cols 2,3)
    add_strassen_2x2(0, 2, 2, 2, 0, 2)
    
    # 2. A_left (2x4) * B_tr (4x1)
    # B_tr is rows 0..3, col 4.
    # A_left is rows 0..1, cols 0..3.
    # Result C_right (2x1) rows 0..1, col 4.
    # This is 2x4x1. Rank 8.
    # Schoolbook: 8 terms.
    for i in range(2):
        for j in range(4):
            U[i * 5 + j][col_offset] = 1.0
            V[j * 5 + 4][col_offset] = 1.0
            W[4 * 2 + i][col_offset] = 1.0 # C index: row 4? No.
            # C is 2x5. Index = row * 5 + col.
            # Row i, Col 4. Index = i * 5 + 4.
            W[i * 5 + 4][col_offset] = 1.0
            col_offset += 1
            
    # 3. A_right (2x1) * B_bl (1x4)
    # A_right is rows 0..1, col 4.
    # B_bl is row 4, cols 0..3.
    # Result C_left (2x4) rows 0..1, cols 0..3.
    # This is 2x1x4. Rank 8.
    for i in range(2):
        for k in range(4):
            U[i * 5 + 4][col_offset] = 1.0
            V[4 * 5 + k][col_offset] = 1.0
            W[i * 5 + k][col_offset] = 1.0
            col_offset += 1

    # 4. A_right (2x1) * B_br (1x1)
    # A_right is rows 0..1, col 4.
    # B_br is row 4, col 4.
    # Result C_right (2x1) rows 0..1, col 4.
    # This is 2x1x1. Rank 2.
    for i in range(2):
        U[i * 5 + 4][col_offset] = 1.0
        V[4 * 5 + 4][col_offset] = 1.0
        W[i * 5 + 4][col_offset] = 1.0
        col_offset += 1

    # Verify col_offset == 46
    if col_offset != 46:
        # Fallback to schoolbook if construction fails
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Final exact verification
    if _verify_exact(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
