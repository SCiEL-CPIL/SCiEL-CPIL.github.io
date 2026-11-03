"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import random
import time as _time

def _verify_vectorized(U, V, W, n=2, m=3, p=10):
    """Fast verification using einsum."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    T_computed = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
    T_expected = np.ones((n, m, p))
    return np.allclose(T_computed, T_expected, atol=1e-10)

def _is_valid_entries(U, V, W):
    """Check that all entries are integer or half-integer."""
    for mat in [U, V, W]:
        mat = np.array(mat, dtype=np.float64)
        doubled = 2.0 * mat
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    return True

def _build_transposed_rank37():
    """
    Build a rank-37 decomposition by exploiting the transpose symmetry.
    
    The tensor for (2x3) * (3x10) has the same rank as the tensor for
    (10x3) * (3x2). In the transposed problem:
      - A is 10x3, B is 3x2, C is 10x2
      - We can group rows of A into five 2x3 blocks (using rows 0-1, 2-3, ..., 8-9)
      - Each 2x3 * 3x2 block can be decomposed with Strassen on 2x2 sub-blocks
        plus one extra column: rank 7 per block (5*7 = 35)
      - But we only have 5 full 2-row groups, and the last row pair is rows 8,9.
      - Actually: 10 rows = 5 groups of 2 rows each.
      - Each group: 2x3 * 3x2 -> use Strassen: split into 2x2 blocks.
        A_block = [[A00, A01], [A10, A11]] (2x3 split as 2x2 + 2x1)
        B_block = [[B00, B01], [B10, B11]] (3x2 split as 2x2 + 1x2)
        
    Let me think more carefully. For a 2x3 * 3x2 multiplication:
      A = [[a00, a01, a02],
           [a10, a11, a12]]
      B = [[b00, b01],
           [b10, b11],
           [b20, b21]]
      C = A*B = [[c00, c01], [c10, c11]]
      
    We can write this as:
      Split A = [A_left | a_right] where A_left is 2x2, a_right is 2x1
      Split B = [B_top; b_bottom] where B_top is 2x2, b_bottom is 1x2
      
      C = A_left * B_top + a_right * b_bottom
      
    The 2x2*2x2 part has rank 7 (Strassen).
    The 2x1 * 1x2 part has rank 2 (two scalar multiplications).
    Total: 7 + 2 = 9 per block.
    
    5 blocks * 9 = 45. That's not 37.
    
    Let me reconsider. Maybe I can do better.
    
    Actually, for 2x3 * 3x2, the minimum rank is known to be 8 (not 9).
    Let me verify: the tensor [2,3,2] has rank 8.
    
    If each block has rank 8, then 5*8 = 40. Still not 37.
    
    Hmm, let me reconsider the approach. The directive says "five independent 2x2x2 blocks (rank 35) and a 1x2x2 block (rank 2), yielding a total rank of 37."
    
    So the idea is: in the transposed 10x3 * 3x2 problem, we can structure it as:
    - 5 blocks of 2x2 * 2x2 (each rank 7, total 35)
    - 1 block of 1x2 * 2x2 (rank 2)
    - Total: 37
    
    But wait, 10x3 * 3x2: the 10x3 matrix A has 10 rows and 3 columns.
    The 3x2 matrix B has 3 rows and 2 columns.
    The result C is 10x2.
    
    If we split the 3 columns of A and 3 rows of B:
    A = [A1 | A2 | A3] where A1 is 10x1, A2 is 10x1, A3 is 10x1
    B = [B1; B2; B3] where B1 is 1x2, B2 is 1x2, B3 is 1x2
    
    C = A1*B1 + A2*B2 + A3*B3
    
    Each term Ai*Bi is 10x1 * 1x2 = 10x2, rank 10 each. Total 30. Not helpful.
    
    Alternative: split the 10 rows into groups.
    Rows 0-1: 2x3 * 3x2
    Rows 2-3: 2x3 * 3x2
    Rows 4-5: 2x3 * 3x2
    Rows 6-7: 2x3 * 3x2
    Rows 8-9: 2x3 * 3x2
    
    Each 2x3*3x2 block. The rank of [2,3,2] tensor is 8.
    5 * 8 = 40.
    
    Or maybe the directive's claim of 37 is based on a different structure.
    Let me just try to construct a valid decomposition and see what rank we get.
    
    For now, let me implement a clean Strassen-based approach on the transposed tensor
    and see if we can get below 55.
    """
    n, m, p = 2, 3, 10
    
    # Build the transposed tensor decomposition
    # Transposed problem: (10x3) * (3x2) -> 10x2
    # In transposed coords: n'=10, m'=3, p'=2
    # U' shape: (10*3, rank) = (30, rank)
    # V' shape: (3*2, rank) = (6, rank)  
    # W' shape: (10*2, rank) = (20, rank)
    
    # For each 2-row block of the 10x3 matrix, do Strassen 2x2*2x2 + corrections
    # Block j (j=0..4): rows 2j, 2j+1 of A'
    
    rank = 0
    U_trans = []
    V_trans = []
    W_trans = []
    
    # Strassen 2x2 * 2x2 decomposition
    # For matrices X (2x2), Y (2x2), Z = X*Y
    # Strassen uses 7 multiplications:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    for j in range(5):
        # Block j: rows 2j, 2j+1 of A' (10x3), all of B' (3x2)
        # A'_block is 2x3: [[a00, a01, a02], [a10, a11, a12]]
        # B' is 3x2: [[b00, b01], [b10, b11], [b20, b21]]
        
        # Split: A'_block = [A_left (2x2) | a_right (2x1)]
        # B' = [B_top (2x2); b_bottom (1x2)]
        
        # C_block = A_left * B_top + a_right * b_bottom
        
        # A_left * B_top: 2x2 * 2x2, rank 7 (Strassen)
        # a_right * b_bottom: 2x1 * 1x2, rank 2
        
        # Total per block: 9
        # But we can potentially share structure...
        
        # For now, use 9 per block: 5*9 = 45
        # This is already better than 55!
        
        # Strassen part for A_left * B_top:
        # A_left indices in U': row (2j)*3 + col, for col in {0,1}
        # B_top indices in V': row 0*2 + col, for col in {0,1}
        # C indices in W': row (2j)*2 + col, for col in {0,1}
        
        a00 = (2*j) * 3 + 0
        a01 = (2*j) * 3 + 1
        a10 = (2*j+1) * 3 + 0
        a11 = (2*j+1) * 3 + 1
        
        b00 = 0 * 2 + 0
        b01 = 0 * 2 + 1
        b10 = 1 * 2 + 0
        b11 = 1 * 2 + 1
        
        c00 = (2*j) * 2 + 0
        c01 = (2*j) * 2 + 1
        c10 = (2*j+1) * 2 + 0
        c11 = (2*j+1) * 2 + 1
        
        # Strassen 7 mults
        # M1 = (a00+a11)*(b00+b11)
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a00] = 1; U_col[a11] = 1
        V_col[b00] = 1; V_col[b11] = 1
        W_col[c00] += 1; W_col[c11] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M2 = (a10+a11)*b00
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a10] = 1; U_col[a11] = 1
        V_col[b00] = 1
        W_col[c10] += 1; W_col[c11] -= 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M3 = a00*(b01-b11)
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a00] = 1
        V_col[b01] = 1; V_col[b11] = -1
        W_col[c01] += 1; W_col[c11] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M4 = a11*(b10-b00)
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a11] = 1
        V_col[b10] = 1; V_col[b00] = -1
        W_col[c00] += 1; W_col[c10] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M5 = (a00+a01)*b11
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a00] = 1; U_col[a01] = 1
        V_col[b11] = 1
        W_col[c00] -= 1; W_col[c01] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M6 = (a10-a00)*(b00+b01)
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a10] = 1; U_col[a00] = -1
        V_col[b00] = 1; V_col[b01] = 1
        W_col[c11] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M7 = (a01-a11)*(b10+b11)
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a01] = 1; U_col[a11] = -1
        V_col[b10] = 1; V_col[b11] = 1
        W_col[c00] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # a_right * b_bottom: 2x1 * 1x2
        # a_right = [a02, a12]^T (column 2 of A'_block)
        # b_bottom = [b20, b21] (row 2 of B')
        # C contribution: [[a02*b20, a02*b21], [a12*b20, a12*b21]]
        
        # M8 = a02 * b20
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[(2*j) * 3 + 2] = 1
        V_col[2 * 2 + 0] = 1
        W_col[c00] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M9 = a02 * b21
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[(2*j) * 3 + 2] = 1
        V_col[2 * 2 + 1] = 1
        W_col[c01] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M10 = a12 * b20
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[(2*j+1) * 3 + 2] = 1
        V_col[2 * 2 + 0] = 1
        W_col[c10] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M11 = a12 * b21
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[(2*j+1) * 3 + 2] = 1
        V_col[2 * 2 + 1] = 1
        W_col[c11] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
    
    # Wait, that's 11 per block * 5 = 55. Same as before!
    # I made an error. Let me reconsider.
    
    # The issue: a_right * b_bottom is 2x1 * 1x2 which needs rank 2, not 4.
    # It's an outer product of a 2-vector and a 2-vector.
    # [a02, a12]^T * [b20, b21] = [[a02*b20, a02*b21], [a12*b20, a12*b21]]
    # This is rank 1 as a tensor! One multiplication: t = a02*b20... no.
    # Actually it's: t1 = a02*b20, t2 = a02*b21, t3 = a12*b20, t4 = a12*b21
    # But we can do: t = a02*b20, then c00 += t, and we need a02*b21 and a12*b20 and a12*b21.
    # That's 4 independent products. Unless we can combine.
    # 
    # Actually: [a02, a12] * [b20, b21]^T is a 2x2 matrix of rank 1.
    # As a bilinear form, it requires rank 2:
    #   m1 = a02 * b20 -> contributes to c00
    #   m2 = a12 * b21 -> contributes to c11  
    #   But we also need a02*b21 and a12*b20.
    #
    # Hmm, no. The tensor for outer product of 2-vectors has rank 2.
    # [x1*y1, x1*y2, x2*y1, x2*y2] can be written with 2 rank-1 terms:
    #   x1 * [y1, y2, 0, 0] + x2 * [0, 0, y1, y2]
    # But that's not a valid tensor decomposition in our sense.
    #
    # In our tensor decomposition framework, each "multiplication" is a scalar product
    # of a linear combination of A entries and a linear combination of B entries.
    # The outer product tensor [2,1,2] (2x1 * 1x2) has rank 2.
    # 
    # So: m1 = a02 * b20, m2 = (a02*b21 + a12*b20)... no, that's not linear.
    #
    # Actually the rank of the [2,1,2] tensor is 2:
    #   m1 = a02 * b20
    #   m2 = a12 * b21
    #   But then c01 = a02*b21 and c10 = a12*b20 are not captured.
    #
    # I think the rank of [2,1,2] is actually 2, achieved by:
    #   m1 = a02 * (b20 + b21)
    #   m2 = a12 * b20
    #   c00 = m1 - a02*b21... no, that doesn't work.
    #
    # Let me just accept rank 4 for the cross term for now, giving 55 total.
    # But wait - maybe I can restructure to avoid the cross term entirely.
    
    # Alternative: don't split A and B. Use the full 2x3 * 3x2 directly.
    # The tensor [2,3,2] has known rank 8.
    # 5 * 8 = 40.
    
    # Let me build a rank-8 decomposition for 2x3 * 3x2.
    # A = [[a00,a01,a02],[a10,a11,a12]], B = [[b00,b01],[b10,b11],[b20,b21]]
    # C = [[c00,c01],[c10,c11]]
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    
    # This is a 2x3x2 tensor. Known minimum rank is 8.
    # Let me construct it using Strassen-like ideas.
    
    # For now, let me just return the rank-55 Strassen decomposition as fallback
    # and add a proper rank-40 construction.
    
    # Actually, let me just build the straightforward rank-40 and verify it.
    U_trans = []
    V_trans = []
    W_trans = []
    
    for j in range(5):
        # 2x3 * 3x2 block, rank 8
        # Use a known rank-8 decomposition for [2,3,2]
        # 
        # I'll use: split into Strassen 2x2*2x2 (rank 7) + 2x1*1x2 (rank 2) = 9
        # But maybe there's a rank-8 way.
        #
        # For simplicity, use rank 9 per block: total 45
        # This is already an improvement over 55!
        
        a00 = (2*j) * 3 + 0
        a01 = (2*j) * 3 + 1
        a02 = (2*j) * 3 + 2
        a10 = (2*j+1) * 3 + 0
        a11 = (2*j+1) * 3 + 1
        a12 = (2*j+1) * 3 + 2
        
        b00 = 0
        b01 = 1
        b10 = 2
        b11 = 3
        b20 = 4
        b21 = 5
        
        c00 = (2*j) * 2 + 0
        c01 = (2*j) * 2 + 1
        c10 = (2*j+1) * 2 + 0
        c11 = (2*j+1) * 2 + 1
        
        # Strassen for top-left 2x2 * 2x2:
        # A_tl = [[a00, a01], [a10, a11]], B_tl = [[b00, b01], [b10, b11]]
        
        # M1 = (a00+a11)*(b00+b11)
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a00] = 1; U_col[a11] = 1
        V_col[b00] = 1; V_col[b11] = 1
        W_col[c00] += 1; W_col[c11] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M2 = (a10+a11)*b00
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a10] = 1; U_col[a11] = 1
        V_col[b00] = 1
        W_col[c10] += 1; W_col[c11] -= 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M3 = a00*(b01-b11)
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a00] = 1
        V_col[b01] = 1; V_col[b11] = -1
        W_col[c01] += 1; W_col[c11] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M4 = a11*(b10-b00)
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a11] = 1
        V_col[b10] = 1; V_col[b00] = -1
        W_col[c00] += 1; W_col[c10] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M5 = (a00+a01)*b11
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a00] = 1; U_col[a01] = 1
        V_col[b11] = 1
        W_col[c00] -= 1; W_col[c01] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M6 = (a10-a00)*(b00+b01)
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a10] = 1; U_col[a00] = -1
        V_col[b00] = 1; V_col[b01] = 1
        W_col[c11] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M7 = (a01-a11)*(b10+b11)
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a01] = 1; U_col[a11] = -1
        V_col[b10] = 1; V_col[b11] = 1
        W_col[c00] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # Cross terms: a02*b20, a02*b21, a12*b20, a12*b21
        # Use rank 2 for the 2x1 * 1x2 outer product:
        # m8 = a02 * b20, m9 = a12 * b21
        # c00 += a02*b20, c11 += a12*b21
        # But we're missing a02*b21 and a12*b20!
        #
        # So rank 2 is NOT sufficient. We need rank 4 for the full outer product.
        # Unless we restructure.
        
        # Let me just use 4 separate terms:
        # M8 = a02 * b20
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a02] = 1
        V_col[b20] = 1
        W_col[c00] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M9 = a02 * b21
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a02] = 1
        V_col[b21] = 1
        W_col[c01] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M10 = a12 * b20
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a12] = 1
        V_col[b20] = 1
        W_col[c10] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
        
        # M11 = a12 * b21
        U_col = [0]*30; V_col = [0]*6; W_col = [0]*20
        U_col[a12] = 1
        V_col[b21] = 1
        W_col[c11] += 1
        U_trans.append(U_col); V_trans.append(V_col); W_trans.append(W_col)
    
    # That's 11 per block * 5 = 55. Same as before.
    # The transposition doesn't help with this naive approach.
    
    # Let me fall back to the known rank-55 Strassen decomposition.
    return _build_strassen55()

def _build_strassen55():
    """Build the known rank-55 decomposition."""
    n, m, p = 2, 3, 10
    rank = 55
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    for i in range(n):
        a_idx = i * m + 2
        for k in range(p):
            b_idx = 2 * p + k
            c_idx = k * n + i
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    X00, X01, X10, X11 = 0, 1, 3, 4
    
    for j in range(5):
        Y00 = 2 * j
        Y01 = 2 * j + 1
        Y10 = 10 + 2 * j
        Y11 = 10 + 2 * j + 1
        
        Z00 = 4 * j
        Z10 = 4 * j + 1
        Z01 = 4 * j + 2
        Z11 = 4 * j + 3
        
        U[X00, col] += 1; U[X11, col] += 1
        V[Y00, col] += 1; V[Y11, col] += 1
        W[Z00, col] += 1; W[Z11, col] += 1
        col += 1
        
        U[X10, col] += 1; U[X11, col] += 1
        V[Y00, col] += 1
        W[Z10, col] += 1; W[Z11, col] -= 1
        col += 1
        
        U[X00, col] += 1
        V[Y01, col] += 1; V[Y11, col] -= 1
        W[Z01, col] += 1; W[Z11, col] += 1
        col += 1
        
        U[X11, col] += 1
        V[Y10, col] += 1; V[Y00, col] -= 1
        W[Z00, col] += 1; W[Z10, col] += 1
        col += 1
        
        U[X00, col] += 1; U[X01, col] += 1
        V[Y11, col] += 1
        W[Z00, col] -= 1; W[Z01, col] += 1
        col += 1
        
        U[X10, col] += 1; U[X00, col] -= 1
        V[Y00, col] += 1; V[Y01, col] += 1
        W[Z11, col] += 1
        col += 1
        
        U[X01, col] += 1; U[X11, col] -= 1
        V[Y10, col] += 1; V[Y11, col] += 1
        W[Z00, col] += 1
        col += 1
    
    assert col == rank
    return U, V, W

def _try_eliminate_columns(U, V, W, n=2, m=3, p=10, time_limit=20.0):
    """Try to eliminate redundant columns."""
    start_time = _time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    changed = True
    while changed:
        changed = False
        for col in range(rank):
            if _time.time() - start_time > time_limit:
                break
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, col] = 0
            V_test[:, col] = 0
            W_test[:, col] = 0
            if _verify_vectorized(U_test, V_test, W_test, n, m, p):
                mask = [c for c in range(rank) if c != col]
                U = U[:, mask]
                V = V[:, mask]
                W = W[:, mask]
                rank -= 1
                changed = True
                break
    return U, V, W, rank

def _local_search(U0, V0, W0, n=2, m=3, p=10, time_limit=40.0, max_iter=5000):
    """Local search to find lower rank."""
    start_time = _time.time()
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        if _time.time() - start_time > time_limit:
            break
        n_changes = random.randint(1, 3)
        for _ in range(n_changes):
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = random.randint(0, U.shape[0]-1)
                col = random.randint(0, U.shape[1]-1)
                U[idx, col] += random.choice([-1, 1])
            elif mat == 'V':
                idx = random.randint(0, V.shape[0]-1)
                col = random.randint(0, V.shape[1]-1)
                V[idx, col] += random.choice([-1, 1])
            else:
                idx = random.randint(0, W.shape[0]-1)
                col = random.randint(0, W.shape[1]-1)
                W[idx, col] += random.choice([-1, 1])
        
        if _verify_vectorized(U, V, W, n, m, p) and _is_valid_entries(U, V, W):
            U2, V2, W2, r2 = _try_eliminate_columns(U, V, W, n, m, p, time_limit=5.0)
            if r2 < best_rank:
                best_rank = r2
                best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()
                U, V, W = U2.copy(), V2.copy(), W2.copy()
    
    if best_rank < rank:
        return best_U, best_V, best_W, best_rank
    return None

def _structured_search(n=2, m=3, p=10, time_limit=70.0):
    """Structured search for lower-rank decompositions."""
    start_time = _time.time()
    U, V, W = _build_strassen55()
    best_rank = 55
    
    U, V, W, rank = _try_eliminate_columns(U, V, W, n, m, p, time_limit=20.0)
    best_rank = min(best_rank, rank)
    
    if _time.time() - start_time < time_limit - 10:
        result = _local_search(U, V, W, n, m, p, time_limit=time_limit - 20.0, max_iter=3000)
        if result is not None:
            U, V, W, rank = result
            best_rank = rank
    
    return U, V, W, best_rank

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 x 3x10 matrix multiplication."""
    start_time = _time.time()
    deadline = start_time + 120.0
    
    U, V, W = _build_strassen55()
    best_rank = 55
    
    if _time.time() < deadline - 30:
        U, V, W, rank = _structured_search(n=2, m=3, p=10, time_limit=90.0)
        best_rank = rank
    
    if not _verify_vectorized(U, V, W, 2, 3, 10):
        U, V, W = _build_strassen55()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
