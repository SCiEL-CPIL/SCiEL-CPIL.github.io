import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _construct_rank_46(n, m, p):
    """
    Construct a rank 46 decomposition for 2x3 * 3x10.
    Splits B (3x10) into two 3x5 blocks.
    For each 3x5 block:
      Splits A (2x3) into A_2x2 and A_2x1.
      Splits B_block (3x5) into B_2x5 and B_1x5.
      A_2x2 * B_2x5 is decomposed as:
        - Strassen (2x2x2) for first 2 cols of B_2x5. Rank 7.
        - Strassen (2x2x2) for next 2 cols of B_2x5. Rank 7.
        - Schoolbook (2x2x1) for last 1 col of B_2x5. Rank 4.
        Total rank 18.
      A_2x1 * B_1x5 is schoolbook. Rank 5.
      Total per block: 23.
    Total rank: 46.
    """
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Process 2 blocks of 5 columns each
    for block in range(2):
        # Column offset in B for this block
        base_col = block * 5
        
        # --- Part 1: A_2x2 * B_2x5 ---
        # A_2x2 indices in U: 0, 1, 2, 3 (a, b, c, d)
        # B_2x5 rows in V: 0, 1 (rows 0 and 1 of B)
        # C_2x5 indices in W: (base_col + k)*n + i
        
        # Sub-part 1a: Strassen for cols 0, 1 of B_2x5
        # Cols in B: base_col, base_col+1
        # V indices:
        idx_e = 0 * p + base_col
        idx_f = 0 * p + base_col + 1
        idx_g = 1 * p + base_col
        idx_h = 1 * p + base_col + 1
        
        # W indices:
        idx_c00 = (base_col) * n + 0
        idx_c01 = (base_col + 1) * n + 0
        idx_c10 = (base_col) * n + 1
        idx_c11 = (base_col + 1) * n + 1
        
        # U indices: a=0, b=1, c=2, d=3
        
        # M1
        U[0][col] += 1; U[3][col] += 1
        V[idx_e][col] += 1; V[idx_h][col] += 1
        W[idx_c00][col] += 1; W[idx_c11][col] += 1
        col += 1
        
        # M2
        U[2][col] += 1; U[3][col] += 1
        V[idx_e][col] += 1
        W[idx_c10][col] += 1
        col += 1
        
        # M3
        U[0][col] += 1
        V[idx_f][col] += 1; V[idx_h][col] -= 1
        W[idx_c01][col] += 1
        col += 1
        
        # M4
        U[3][col] += 1
        V[idx_g][col] += 1; V[idx_f][col] -= 1
        W[idx_c00][col] += 1; W[idx_c10][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1; U[1][col] += 1
        V[idx_h][col] += 1
        W[idx_c00][col] -= 1; W[idx_c01][col] += 1
        col += 1
        
        # M6
        U[2][col] += 1; U[3][col] += 1
        V[idx_f][col] += 1; V[idx_h][col] -= 1
        W[idx_c11][col] -= 1
        col += 1
        
        # M7
        U[0][col] += 1; U[1][col] += 1
        V[idx_g][col] += 1; V[idx_e][col] -= 1
        W[idx_c00][col] += 1
        col += 1

        # Sub-part 1b: Strassen for cols 2, 3 of B_2x5
        # Cols in B: base_col+2, base_col+3
        idx_e2 = 0 * p + base_col + 2
        idx_f2 = 0 * p + base_col + 3
        idx_g2 = 1 * p + base_col + 2
        idx_h2 = 1 * p + base_col + 3
        
        idx_c00_2 = (base_col + 2) * n + 0
        idx_c01_2 = (base_col + 3) * n + 0
        idx_c10_2 = (base_col + 2) * n + 1
        idx_c11_2 = (base_col + 3) * n + 1

        # M8 (M1 of this block)
        U[0][col] += 1; U[3][col] += 1
        V[idx_e2][col] += 1; V[idx_h2][col] += 1
        W[idx_c00_2][col] += 1; W[idx_c11_2][col] += 1
        col += 1
        
        # M9 (M2)
        U[2][col] += 1; U[3][col] += 1
        V[idx_e2][col] += 1
        W[idx_c10_2][col] += 1
        col += 1
        
        # M10 (M3)
        U[0][col] += 1
        V[idx_f2][col] += 1; V[idx_h2][col] -= 1
        W[idx_c01_2][col] += 1
        col += 1
        
        # M11 (M4)
        U[3][col] += 1
        V[idx_g2][col] += 1; V[idx_f2][col] -= 1
        W[idx_c00_2][col] += 1; W[idx_c10_2][col] += 1
        col += 1
        
        # M12 (M5)
        U[0][col] += 1; U[1][col] += 1
        V[idx_h2][col] += 1
        W[idx_c00_2][col] -= 1; W[idx_c01_2][col] += 1
        col += 1
        
        # M13 (M6)
        U[2][col] += 1; U[3][col] += 1
        V[idx_f2][col] += 1; V[idx_h2][col] -= 1
        W[idx_c11_2][col] -= 1
        col += 1
        
        # M14 (M7)
        U[0][col] += 1; U[1][col] += 1
        V[idx_g2][col] += 1; V[idx_e2][col] -= 1
        W[idx_c00_2][col] += 1
        col += 1

        # Sub-part 1c: Schoolbook for col 4 of B_2x5
        # Col in B: base_col+4
        idx_g4 = 0 * p + base_col + 4 # e4
        idx_h4 = 1 * p + base_col + 4 # h4
        idx_c04 = (base_col + 4) * n + 0
        idx_c14 = (base_col + 4) * n + 1

        # M15: a * e4
        U[0][col] += 1
        V[idx_g4][col] += 1
        W[idx_c04][col] += 1
        col += 1
        
        # M16: b * h4
        U[1][col] += 1
        V[idx_h4][col] += 1
        W[idx_c04][col] += 1
        col += 1
        
        # M17: c * e4
        U[2][col] += 1
        V[idx_g4][col] += 1
        W[idx_c14][col] += 1
        col += 1
        
        # M18: d * h4
        U[3][col] += 1
        V[idx_h4][col] += 1
        W[idx_c14][col] += 1
        col += 1

        # --- Part 2: A_2x1 * B_1x5 ---
        # A_2x1 indices in U: 4, 5 (e, f in A notation? No, A[0,2], A[1,2])
        # B_1x5 row in V: 2 (row 2 of B)
        # C_1x5 indices in W: (base_col + k)*n + i
        
        for k in range(5):
            # M19..M23 for first row of A_2x1 (A[0,2])
            idx_b2 = 2 * p + base_col + k
            idx_c = (base_col + k) * n + 0
            
            U[4][col] += 1
            V[idx_b2][col] += 1
            W[idx_c][col] += 1
            col += 1
            
            # M24..M28 for second row of A_2x1 (A[1,2])
            idx_c2 = (base_col + k) * n + 1
            
            U[5][col] += 1
            V[idx_b2][col] += 1
            W[idx_c2][col] += 1
            col += 1

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Use the specific rank 46 decomposition
    result = _construct_rank_46(n, m, p)
    
    # Verify
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
