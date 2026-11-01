"""Rewritten program: structured Strassen-based decomposition for 2x4x6 matrix multiplication."""

DIMENSIONS = [2, 4, 6]

import time

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS

    # Schoolbook baseline (rank 48)
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()

    # Strassen-based decomposition for 2x2x2 blocks
    # Strassen's 7 multiplications for 2x2 matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # For 2x2x6: split B into three 2x2 blocks (columns 0-1, 2-3, 4-5)
    # Each 2x2x2 block uses 7 rank terms.
    # But we can share the A-side computations!
    
    # Actually, let's think more carefully.
    # A is 2x4, B is 4x6.
    # Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    # Split B into B1 (2x2, rows 0-1, cols 0-1), B2 (2x2, rows 0-1, cols 2-3), 
    #            B3 (2x2, rows 0-1, cols 4-5),
    #            B4 (2x2, rows 2-3, cols 0-1), B5 (2x2, rows 2-3, cols 2-3),
    #            B6 (2x2, rows 2-3, cols 4-5)
    
    # C = A1 * [B1 B2 B3] + A2 * [B4 B5 B6]
    #   = A1*B1 + A1*B2 + A1*B3 + A2*B4 + A2*B5 + A2*B6
    
    # Each 2x2 * 2x2 product uses 7 Strassen multiplications.
    # Total: 6 * 7 = 42 rank terms.
    
    # Let's implement this.
    
    # Strassen terms for one 2x2 * 2x2 product:
    # Each term is (coeff_for_A_entry, coeff_for_B_entry, coeff_for_C_entry)
    # We'll represent each rank-1 term as a tuple of (U_contrib, V_contrib, W_contrib)
    # where U_contrib is a vector over the 4 entries of A_block,
    # V_contrib is a vector over the 4 entries of B_block,
    # W_contrib is a vector over the 4 entries of C_block.
    
    # A_block indices: (0,0), (0,1), (1,0), (1,1) -> local indices 0,1,2,3
    # B_block indices: same
    # C_block indices: same
    
    def strassen_2x2():
        # Returns list of 7 tuples (u_vec, v_vec, w_vec)
        # u_vec, v_vec, w_vec are tuples of 4 Fractions
        
        terms = []
        
        # M1 = (A11+A22)(B11+B22)
        u1 = (1, 0, 0, 1)
        v1 = (1, 0, 0, 1)
        w1 = (1, 0, 0, 1)  # contributes to C11 and C22
        terms.append((u1, v1, w1))
        
        # M2 = (A21+A22)B11
        u2 = (0, 0, 1, 1)
        v2 = (1, 0, 0, 0)
        w2 = (0, 0, 1, 0)  # contributes to C21 and C22
        terms.append((u2, v2, w2))
        
        # M3 = A11(B21-B22)
        u3 = (1, 0, 0, 0)
        v3 = (0, 0, 1, -1)
        w3 = (0, 1, 0, 0)  # contributes to C12
        terms.append((u3, v3, w3))
        
        # M4 = A22(B12-B11)
        u4 = (0, 0, 0, 1)
        v4 = (0, 1, -1, 0)
        w4 = (1, 0, 0, 0)  # contributes to C11 and C21
        terms.append((u4, v4, w4))
        
        # M5 = (A11+A12)B22
        u5 = (1, 1, 0, 0)
        v5 = (0, 0, 0, 1)
        w5 = (-1, 0, 0, 0)  # contributes to C11
        terms.append((u5, v5, w5))
        
        # M6 = (A21-A11)(B11+B12)
        u6 = (-1, 0, 1, 0)
        v6 = (1, 1, 0, 0)
        w6 = (0, 0, 0, 1)  # contributes to C22
        terms.append((u6, v6, w6))
        
        # M7 = (A12-A22)(B21+B22)
        u7 = (0, 1, 0, -1)
        v7 = (0, 0, 1, 1)
        w7 = (1, 0, 0, 0)  # contributes to C11
        terms.append((u7, v7, w7))
        
        return terms
    
    strassen_terms = strassen_2x2()
    
    # Now build the full decomposition.
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (24, rank)  
    # W has shape (p*n, rank) = (12, rank)
    
    # Global indices:
    # U row: i*m + j where i in [0,2), j in [0,4)
    # V row: j*p + k where j in [0,4), k in [0,6)
    # W row: k*n + i where k in [0,6), i in [0,2)
    
    # We have 6 sub-problems, each using 7 Strassen terms.
    # Total rank = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for a_block in range(2):  # A1 or A2
        for b_block in range(3):  # B1-B3 (for A1) or B4-B6 (for A2)
            # Global column range for B block
            b_col_start = b_block * 2
            # For A1 (a_block=0): B rows 0-1
            # For A2 (a_block=1): B rows 2-3
            b_row_start = a_block * 2
            
            for t_idx, (u_vec, v_vec, w_vec) in enumerate(strassen_terms):
                # u_vec applies to A_block entries (i, j_local)
                # j_local maps to global j: a_block*2 + j_local
                for i_local in range(2):
                    for j_local in range(2):
                        u_val = u_vec[i_local * 2 + j_local]
                        if u_val != 0:
                            i_global = i_local
                            j_global = a_block * 2 + j_local
                            U[i_global * m + j_global][col] = u_val
                
                # v_vec applies to B_block entries (j_local, k_local)
                # j_local maps to global j: b_row_start + j_local
                # k_local maps to global k: b_col_start + k_local
                for j_local in range(2):
                    for k_local in range(2):
                        v_val = v_vec[j_local * 2 + k_local]
                        if v_val != 0:
                            j_global = b_row_start + j_local
                            k_global = b_col_start + k_local
                            V[j_global * p + k_global][col] = v_val
                
                # w_vec applies to C_block entries (i_local, k_local)
                # i_local maps to global i: i_local (since C is 2x6)
                # k_local maps to global k: b_col_start + k_local
                for i_local in range(2):
                    for k_local in range(2):
                        w_val = w_vec[i_local * 2 + k_local]
                        if w_val != 0:
                            i_global = i_local
                            k_global = b_col_start + k_local
                            W[k_global * n + i_global][col] = w_val
                
                col += 1
    
    # Verify the decomposition
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    return best_result

# EVOLVE-BLOCK-END
