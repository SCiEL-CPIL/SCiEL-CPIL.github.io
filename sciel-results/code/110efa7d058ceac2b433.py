"""Matrix multiplication 3,4,8 with Strassen-based decomposition for rank 88."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose 3x4x8 as (2x4x8) + (1x4x8)
    # 2x4x8: split columns 4 into two 2s, rows 2 into two 1s, use Strassen on each 2x2x2
    # Actually, let's think more carefully.
    
    # Better approach: 3x4x8 = (2x4x8) + (1x4x8)
    # 1x4x8: rank = 4*8 = 32 (no compression possible for 1-row)
    # 2x4x8: split into 2x2x8, then use Strassen on 2x2 blocks of the 4 cols
    #   2x2x8 = sum of 4 blocks of 2x2x2, each with Strassen rank 7 -> 28
    # Total: 32 + 28 = 60? Wait, let me re-derive.
    
    # Actually for 2x4x8: 
    #   Split the 4 columns into two pairs: (cols 0,1) and (cols 2,3)
    #   For each pair, we have a 2x2x8 subproblem
    #   2x2x8: split the 8 into four 2s, each 2x2x2 has Strassen rank 7
    #   So 2x2x8 has rank 4*7 = 28
    #   Two such subproblems: 2*28 = 56
    #   Plus 1x4x8 = 32
    #   Total = 88
    
    # Let me construct this properly.
    
    # Strassen 2x2x2 decomposition:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Build the full 3x4x8 decomposition.
    
    # Part 1: Rows 0,1 (2x4x8) - using Strassen on 2x2 blocks
    # Split 4 cols into two 2-col blocks: cols (0,1) and cols (2,3)
    # Split 8 rows (of B) into four 2-row blocks: rows (0,1), (2,3), (4,5), (6,7)
    
    # Part 2: Row 2 (1x4x8) - schoolbook, rank 32
    
    total_rank = 0
    U = {}  # (row_idx, col_idx) -> list of (rank_idx, value)
    V = {}
    W = {}
    
    # --- Part 1: 2x4x8 via Strassen ---
    # For each 2-col block (b in 0,1) and each 2-row block of B (c in 0,1,2,3):
    #   We have a 2x2x2 subproblem with A being 2x2 (rows 0,1 of A; cols 2b,2b+1)
    #   B being 2x2 (rows 2c,2c+1 of B; cols 2b,2b+1)
    #   C being 2x2 (rows 0,1 of C; cols 2c,2c+1)
    
    for b in range(2):  # col block index (0 or 1)
        for c in range(4):  # B-row block index
            # 7 Strassen terms for this 2x2x2
            for t in range(7):
                rank_idx = total_rank
                total_rank += 1
                
                # Determine U, V, W contributions
                # U has shape (n*m, rank) = (3*4, rank) = (12, rank)
                # V has shape (m*p, rank) = (4*8, rank) = (32, rank)
                # W has shape (p*n, rank) = (8*3, rank) = (24, rank)
                
                # A indices: (i, j) where i in {0,1}, j in {2b, 2b+1}
                # B indices: (j, k) where j in {2b, 2b+1}, k in {2c, 2c+1}
                # C indices: (i, k) where i in {0,1}, k in {2c, 2c+1}
                
                # Strassen term t contributes to:
                # U[i*m + j] += coeff_a * 1  (for the A part)
                # V[j*p + k] += coeff_b * 1  (for the B part)
                # W[k*n + i] += coeff_c * 1  (for the C part)
                
                # Let's define the Strassen terms explicitly:
                # Each term is (A_coeffs, B_coeffs, C_coeffs) where each is a dict mapping (i,j) -> val
                # Actually, let me define them as lists of (index, value) for each matrix.
                
                # A is 2x2, B is 2x2, C is 2x2
                # Local indices: (0,0),(0,1),(1,0),(1,1)
                
                strassen_terms = [
                    # (A_coeffs, B_coeffs, C_coeffs)
                    # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
                    ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                    # M2 = (A10+A11)B00 -> C10+=1
                    ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
                    # M3 = A00(B10-B11) -> C01+=1
                    ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1}),
                    # M4 = A11(B01-B00) -> C00-=1, C10+=1
                    ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):-1, (1,0):1}),
                    # M5 = (A00+A01)B11 -> C00-=1, C01+=1
                    ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
                    # M6 = (A10-A00)(B00+B01) -> C11+=1
                    ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
                    # M7 = (A01-A11)(B10+B11) -> C00+=1, C11+=1
                    ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                ]
                
                A_coeffs, B_coeffs, C_coeffs = strassen_terms[t]
                
                # Map local (i,j) to global A index: i_global = i, j_global = 2*b + j
                # U index = i_global * m + j_global = i*4 + (2*b + j)
                for (li, lj), val in A_coeffs.items():
                    u_idx = li * m + (2*b + lj)
                    if u_idx not in U:
                        U[u_idx] = {}
                    U[u_idx][rank_idx] = U[u_idx].get(rank_idx, 0) + val
                
                # Map local (j,k) to global B index: j_global = 2*b + j, k_global = 2*c + k
                # V index = j_global * p + k_global = (2*b + j)*8 + (2*c + k)
                for (lj, lk), val in B_coeffs.items():
                    v_idx = (2*b + lj) * p + (2*c + lk)
                    if v_idx not in V:
                        V[v_idx] = {}
                    V[v_idx][rank_idx] = V[v_idx].get(rank_idx, 0) + val
                
                # Map local (i,k) to global C index: i_global = i, k_global = 2*c + k
                # W index = k_global * n + i_global = (2*c + k)*3 + i
                for (li, lk), val in C_coeffs.items():
                    w_idx = (2*c + lk) * n + li
                    if w_idx not in W:
                        W[w_idx] = {}
                    W[w_idx][rank_idx] = W[w_idx].get(rank_idx, 0) + val
    
    # --- Part 2: 1x4x8 schoolbook (row 2 of A) ---
    for j in range(4):
        for k in range(8):
            rank_idx = total_rank
            total_rank += 1
            u_idx = 2 * m + j  # row 2, col j
            v_idx = j * p + k  # col j, row k
            w_idx = k * n + 2  # row k, col 2 (i=2)
            
            U.setdefault(u_idx, {})[rank_idx] = 1
            V.setdefault(v_idx, {})[rank_idx] = 1
            W.setdefault(w_idx, {})[rank_idx] = 1
    
    # Convert to dense matrices
    rank = total_rank
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for row, cols in U.items():
        for c, val in cols.items():
            U_mat[row][c] = val
    for row, cols in V.items():
        for c, val in cols.items():
            V_mat[row][c] = val
    for row, cols in W.items():
        for c, val in cols.items():
            W_mat[row][c] = val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
