import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Schoolbook for 1x4x8 (i=0)
    # i=0, j=0..3, k=0..7
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: Strassen for 2x4x8 (i=1,2)
    # This part handles rows 1 and 2 of A.
    # We decompose into 8 blocks of 2x2x2.
    # Block indices:
    # j_block: 0 (cols 0,1), 1 (cols 2,3)
    # k_block: 0 (cols 0,1), 1 (cols 2,3), 2 (cols 4,5), 3 (cols 6,7)
    # Wait, B is 4x8.
    # B rows 0,1 correspond to j=0,1 of A? No.
    # A is 2x4. B is 4x8.
    # A cols 0,1 multiply B rows 0,1.
    # A cols 2,3 multiply B rows 2,3.
    
    # So for j_block=0 (A cols 0,1), we use B rows 0,1.
    # For j_block=1 (A cols 2,3), we use B rows 2,3.
    
    # B cols are split into 4 blocks of 2:
    # k_block=0: cols 0,1
    # k_block=1: cols 2,3
    # k_block=2: cols 4,5
    # k_block=3: cols 6,7
    
    # So we have 2 (j_blocks) * 4 (k_blocks) = 8 Strassen 2x2x2 blocks.
    
    for jb in range(2):
        for kb in range(4):
            # Global offsets
            i_off = 1 # Rows 1,2 of A
            j_off = jb * 2 # Cols 0,1 or 2,3 of A
            # B rows: if jb=0, rows 0,1. If jb=1, rows 2,3.
            b_row_off = jb * 2
            k_off = kb * 2 # Cols 0,1; 2,3; 4,5; 6,7 of B
            
            # Strassen terms for 2x2x2
            # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
            # Global A index: (i_off + i) * m + (j_off + j)
            # Global B index: (b_row_off + j) * p + (k_off + k)  <-- WAIT
            # B index is j * p + k.
            # Here j is the row of B.
            # In the product A B, A_{i, j} B_{j, k}.
            # So the 'j' index in V corresponds to the row of B.
            # In our 2x2x2 block:
            # A is 2x2. Rows i=0,1. Cols j=0,1.
            # B is 2x2. Rows j=0,1. Cols k=0,1.
            # So V index uses j (row of B).
            
            # Strassen terms:
            # 1: (a+d)(e+h)
            # 2: (c+d)e
            # 3: a(f-h)
            # 4: d(g-e)
            # 5: (a+b)h
            # 6: (c-a)(e+f)
            # 7: (b-d)(g+h)
            
            # Helper to add a term
            def add_strassen_term(u_indices, v_indices, w_indices, coeffs_u, coeffs_v, coeffs_w):
                nonlocal col
                for idx in range(len(coeffs_u)):
                    U[u_indices[idx]][col] += coeffs_u[idx]
                for idx in range(len(coeffs_v)):
                    V[v_indices[idx]][col] += coeffs_v[idx]
                for idx in range(len(coeffs_w)):
                    W[w_indices[idx]][col] += coeffs_w[idx]
                col += 1

            # Indices for A (U)
            # a: i=0, j=0 -> (i_off+0)*m + (j_off+0)
            # b: i=0, j=1 -> (i_off+0)*m + (j_off+1)
            # c: i=1, j=0 -> (i_off+1)*m + (j_off+0)
            # d: i=1, j=1 -> (i_off+1)*m + (j_off+1)
            u_a = (i_off + 0) * m + (j_off + 0)
            u_b = (i_off + 0) * m + (j_off + 1)
            u_c = (i_off + 1) * m + (j_off + 0)
            u_d = (i_off + 1) * m + (j_off + 1)
            
            # Indices for B (V)
            # e: row=0, col=0 -> (b_row_off+0)*p + (k_off+0)
            # f: row=0, col=1 -> (b_row_off+0)*p + (k_off+1)
            # g: row=1, col=0 -> (b_row_off+1)*p + (k_off+0)
            # h: row=1, col=1 -> (b_row_off+1)*p + (k_off+1)
            v_e = (b_row_off + 0) * p + (k_off + 0)
            v_f = (b_row_off + 0) * p + (k_off + 1)
            v_g = (b_row_off + 1) * p + (k_off + 0)
            v_h = (b_row_off + 1) * p + (k_off + 1)
            
            # Indices for C (W)
            # C is 2x2.
            # W index: k * n + i.
            # C_00: k=0, i=0 -> (k_off+0)*n + (i_off+0)
            # C_01: k=1, i=0 -> (k_off+1)*n + (i_off+0)
            # C_10: k=0, i=1 -> (k_off+0)*n + (i_off+1)
            # C_11: k=1, i=1 -> (k_off+1)*n + (i_off+1)
            w_00 = (k_off + 0) * n + (i_off + 0)
            w_01 = (k_off + 1) * n + (i_off + 0)
            w_10 = (k_off + 0) * n + (i_off + 1)
            w_11 = (k_off + 1) * n + (i_off + 1)
            
            # Term 1: (a+d)(e+h) -> C00(+1), C11(+1)
            add_strassen_term(
                [u_a, u_d], [v_e, v_h], [w_00, w_11],
                [1, 1], [1, 1], [1, 1]
            )
            
            # Term 2: (c+d)e -> C10(+1), C11(-1)
            add_strassen_term(
                [u_c, u_d], [v_e], [w_10, w_11],
                [1, 1], [1], [1, -1]
            )
            
            # Term 3: a(f-h) -> C01(+1), C11(+1)
            add_strassen_term(
                [u_a], [v_f, v_h], [w_01, w_11],
                [1], [1, -1], [1, 1]
            )
            
            # Term 4: d(g-e) -> C00(-1), C10(+1)
            add_strassen_term(
                [u_d], [v_g, v_e], [w_00, w_10],
                [1], [1, -1], [-1, 1]
            )
            
            # Term 5: (a+b)h -> C00(-1), C01(+1)
            add_strassen_term(
                [u_a, u_b], [v_h], [w_00, w_01],
                [1, 1], [1], [-1, 1]
            )
            
            # Term 6: (c-a)(e+f) -> C11(+1)
            add_strassen_term(
                [u_c, u_a], [v_e, v_f], [w_11],
                [1, -1], [1, 1], [1]
            )
            
            # Term 7: (b-d)(g+h) -> C00(+1)
            add_strassen_term(
                [u_b, u_d], [v_g, v_h], [w_00],
                [1, -1], [1, 1], [1]
            )
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
