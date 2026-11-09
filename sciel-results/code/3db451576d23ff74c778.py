import numpy as np
import time

DIMENSIONS = (4, 4, 7)

def construct():
    start_time = time.time()
    DEADLINE = 130
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28
    
    # Rank 49 for 4x4x4 part, Rank 48 for 4x4x3 part
    rank_444 = 49
    rank_443 = 48
    total_rank = rank_444 + rank_443
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    # --- Part 1: 4x4x4 using Strassen + Winograd ---
    
    # Strassen definitions
    # s: index 0-6
    # X_def: list of (block_row, block_col, coeff) for A
    # Y_def: list of (block_row, block_col, coeff) for B
    # Dest: list of (block_row, block_col, coeff) for C
    
    # A blocks: 00, 01, 10, 11
    # A[00] = A[0:2, 0:2], A[01] = A[0:2, 2:4], A[10] = A[2:4, 0:2], A[11] = A[2:4, 2:4]
    
    strassen_defs = [
        # s, X_def, Y_def, Dest
        (0, [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        (1, [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        (2, [(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        (3, [(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        (4, [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        (5, [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        (6, [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    # Winograd definitions
    # w: index 0-6
    # u_def: list of (row, col, coeff) for X (2x2)
    # v_def: list of (row, col, coeff) for Y (2x2)
    # out_def: list of (row, col, coeff) for Z (2x2)
    
    winograd_defs = [
        (0, [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        (1, [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        (2, [(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
        (3, [(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        (4, [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        (5, [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        (6, [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    for s_idx, (s, X_def, Y_def, Dest) in enumerate(strassen_defs):
        for w_idx, (w, u_def, v_def, out_def) in enumerate(strassen_defs):
            r = s_idx * 7 + w_idx
            
            # Build U column
            # U[a, r] = sum over X blocks in X_def of coeff_x * (sum over u_def elements mapping to A)
            # This is a composition of linear maps.
            
            # Helper: get coefficient of A[i,j] in X_def composed with u_def
            # X is 2x2. X[i', j'] = sum_{bx, by} coeff_x * A[2*bx + i', 2*by + j']
            # u is linear form of X. u_val = sum_{i', j'} coeff_u * X[i', j']
            # So A[2*bx + i', 2*by + j'] has coeff coeff_x * coeff_u in u_val.
            
            U_col = np.zeros(nm)
            for (bx, by, cx) in X_def:
                for (i_prime, j_prime, cu) in u_def:
                    # A index
                    i = 2 * bx + i_prime
                    j = 2 * by + j_prime
                    a_idx = i * m + j
                    U_col[a_idx] += cx * cu
            
            U[:, r] = U_col
            
            # Build V column
            # V[b, r] = sum over Y blocks in Y_def of coeff_y * (sum over v_def elements mapping to B)
            # Y[i', j'] = sum_{bx, by} coeff_y * B[2*bx + i', 2*by + j']
            # Note: B is 4x4. B indices are j, k.
            # In B block (bx, by), row is 2*bx + i_prime, col is 2*by + j_prime.
            # Wait, B is m x p = 4 x 4.
            # B[row, col].
            # Block (bx, by) corresponds to rows 2*bx..2*bx+1 and cols 2*by..2*by+1.
            # So B[2*bx + i', 2*by + j'].
            
            V_col = np.zeros(mp)
            for (bx, by, cy) in Y_def:
                for (i_prime, j_prime, cv) in v_def:
                    # B index
                    row_b = 2 * bx + i_prime
                    col_b = 2 * by + j_prime
                    b_idx = row_b * p + col_b
                    V_col[b_idx] += cy * cv
            
            V[:, r] = V_col
            
            # Build W column
            # W[c, r] = sum over Dest blocks of coeff_dest * (sum over out_def elements mapping to C)
            # C block (dx, dy) corresponds to rows 2*dx..2*dx+1 and cols 2*dy..2*dy+1.
            # Z[i', j'] from Winograd maps to C[2*dx + i', 2*dy + j'].
            # C index c = col * n + row = (2*dy + j') * 4 + (2*dx + i').
            
            W_col = np.zeros(pn)
            for (dx, dy, cd) in Dest:
                for (i_prime, j_prime, co) in out_def:
                    # C index
                    row_c = 2 * dx + i_prime
                    col_c = 2 * dy + j_prime
                    c_idx = col_c * n + row_c
                    W_col[c_idx] += cd * co
            
            W[:, r] = W_col
    
    # --- Part 2: 4x4x3 using Schoolbook ---
    # A is 4x4, B is 4x3, C is 4x3.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # i in 0..3, j in 0..3, k in 0..2
    
    offset = rank_444
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # p is 7, but we only do first 3 columns here? 
                               # Wait, B is 4x7.
                               # We split B into B1 (4x4) and B2 (4x3).
                               # B1 corresponds to cols 0..3 of B.
                               # B2 corresponds to cols 4..6 of B.
                               # In the 4x4x4 part, we used B1.
                               # In the 4x4x3 part, we use B2.
                               # So j is still 0..3 (row of B).
                               # k is 0..2 (local col of B2).
                               # Global col of B is 4 + k.
                               
                               # A index: i * 4 + j
                               # B index: j * 7 + (4 + k)
                               # C index: (4 + k) * 4 + i
                               
                               a_idx = i * m + j
                               b_idx = j * p + (4 + k)
                               c_idx = (4 + k) * n + i
                               
                               U[a_idx, offset + col] = 1.0
                               V[b_idx, offset + col] = 1.0
                               W[c_idx, offset + col] = 1.0
                               col += 1
                               
    # Validate quickly
    # (Omitted for speed in final version, but good for debug)
    
    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
