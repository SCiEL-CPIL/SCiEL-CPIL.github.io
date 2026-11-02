"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    
    # Try rank-44 decomposition using Strassen for 2x2x2 blocks
    # A is 3x4, B is 4x4, result is 3x4
    # Split A into [A1|A2] (each 3x2), B into [[B11|B12],[B21|B22]] (each 2x2)
    # Z1 = A1*B11 + A2*B21 (each 3x2x2, rank 11)
    # Z2 = A1*B12 + A2*B22 (each 3x2x2, rank 11)
    # Total rank = 44
    
    # First, verify the Strassen 2x2x2 decomposition
    # 2x2x2 tensor: T[i,j,k] = 1 for i,j,k in {0,1}
    # u indexed by (i,j): u[0]=(0,0), u[1]=(0,1), u[2]=(1,0), u[3]=(1,1)
    # v indexed by (j,k): v[0]=(0,0), v[1]=(0,1), v[2]=(1,0), v[3]=(1,1)
    # w indexed by (k,i): w[0]=(0,0), w[1]=(0,1), w[2]=(1,0), w[3]=(1,1)
    
    # Strassen rank-7 decomposition of 2x2x2 MM tensor
    # Each term c has u_c, v_c, w_c (4-element vectors)
    strassen_u = [
        [1, 0, 0, -1],  # (a-d)
        [1, 0, 0, 1],   # (a+d)
        [0, 0, 0, 1],   # d
        [1, 0, 0, 0],   # a
        [1, 1, 0, 0],   # (a+b)
        [0, 0, 1, -1],  # (c-d)
        [0, 1, -1, 0],  # (b-c)
    ]
    strassen_v = [
        [0, 0, 1, 1],   # (g+h)
        [1, 0, 0, 0],   # e
        [-1, 1, 0, 0],  # (f-e)
        [1, 0, 0, -1],  # (e-h)
        [0, 0, 0, 1],   # h
        [1, 1, 0, 0],   # (e+f)
        [0, 0, 1, 1],   # (g+h)
    ]
    strassen_w = [
        [1, 0, 0, 1],   # C[0,0]+C[1,1]
        [1, 1, 0, 0],   # C[0,0]+C[0,1]
        [0, 0, 1, 1],   # C[1,0]+C[1,1]
        [1, 0, 0, 1],   # C[0,0]+C[1,1]
        [1, 0, 1, 0],   # C[0,0]+C[1,0]
        [0, 0, 1, 1],   # C[1,0]+C[1,1]
        [1, 0, 0, 1],   # C[0,0]+C[1,1]
    ]
    
    # Verify Strassen decomposition
    def verify_2x2x2(u_list, v_list, w_list):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for c in range(7):
                        ui = u_list[c][i*2+j]
                        vj = v_list[c][j*2+k]
                        wk = w_list[c][k*2+i]
                        total += ui * vj * wk
                    if total != 1:
                        return False
        return True
    
    strassen_valid = verify_2x2x2(strassen_u, strassen_v, strassen_w)
    
    if strassen_valid and (time.time() - start_time) < 100:
        # Build rank-44 decomposition
        rank = 44
        U = [[0] * rank for _ in range(n * m)]  # 12 x 44
        V = [[0] * rank for _ in range(m * p)]  # 16 x 44
        W = [[0] * rank for _ in range(p * n)]  # 12 x 44
        
        # 4 blocks of 3x2x2, each rank 11 (7 Strassen + 4 trivial)
        # Block (ai, bi) where ai in {0,1} (which half of A's columns), bi in {0,1} (which half of B's columns)
        # A1 = columns 0,1 of A; A2 = columns 2,3 of A
        # B11 = rows 0,1 cols 0,1; B12 = rows 0,1 cols 2,3
        # B21 = rows 2,3 cols 0,1; B22 = rows 2,3 cols 2,3
        
        # For block (ai, bi):
        # A_part = A[:, ai*2:ai*2+2] (3x2)
        # B_part = B[bi*2:bi*2+2, :] ... wait, B is 4x4, so B_part = B[bi*2:bi*2+2, :] is 2x4? No.
        
        # Actually: Z = A*B where A is 3x4, B is 4x4
        # A = [A_left | A_right] where A_left = A[:, 0:2], A_right = A[:, 2:4]
        # B = [[B_topleft | B_topright], [B_botleft | B_botright]]
        # B_topleft = B[0:2, 0:2], B_topright = B[0:2, 2:4]
        # B_botleft = B[2:4, 0:2], B_botright = B[2:4, 2:4]
        
        # Z = [Z_left | Z_right] where Z_left = A_left * B_topleft + A_right * B_botleft
        # Z_right = A_left * B_topright + A_right * B_botright
        
        # So 4 products:
        # P1 = A_left * B_topleft (3x2 * 2x2 = 3x2)
        # P2 = A_right * B_botleft (3x2 * 2x2 = 3x2)
        # P3 = A_left * B_topright (3x2 * 2x2 = 3x2)
        # P4 = A_right * B_botright (3x2 * 2x2 = 3x2)
        
        # Z_left = P1 + P2, Z_right = P3 + P4
        
        # Each P is a 3x2x2 product. For P = A_part * B_part:
        # A_part is 3x2, B_part is 2x2, result is 3x2
        # Top 2 rows of A_part: 2x2x2 (Strassen, rank 7)
        # Bottom 1 row of A_part: 1x2x2 (trivial, rank 4)
        
        # For the 3x2x2 product, the indices are:
        # i in {0,1,2} (row of A_part and result)
        # j in {0,1} (column of A_part / row of B_part)
        # k in {0,1} (column of B_part and result)
        
        # U index: i*m_total + j_col where j_col is the global column of A
        # For block (ai, bi): A_part column j corresponds to global column ai*2+j
        # B_part row j corresponds to global row bi*2+j... wait no.
        
        # Let me re-index. For product P = A_part * B_part:
        # A_part[i, j] where i in {0,1,2}, j in {0,1}
        # B_part[j, k] where j in {0,1}, k in {0,1}
        # Result[i, k]
        
        # In the global U matrix: U[i*m + j_global, c] where j_global is the column index in A
        # For block ai: j_global = ai*2 + j
        # In the global V matrix: V[j_global2 * p + k_global, c] where j_global2 is row in B
        # For block bi: j_global2 = bi*2 + j
        # In the global W matrix: W[k_global * n + i_global, c]
        # For block bi: k_global = bi*2 + k
        # i_global = i (same for all blocks)
        
        for ai in range(2):
            for bi in range(2):
                # This block computes A[:, ai*2:ai*2+2] * B[bi*2:bi*2+2, :]
                # Wait, that's not right. B is 4x4. B[bi*2:bi*2+2, :] is 2x4.
                # I need B_part to be 2x2.
                
                # Let me redefine. The 4 products are:
                # P[ai][bi] = A[:, ai*2:(ai+1)*2] * B[bi*2:(bi+1)*2, :]
                # But B[bi*2:(bi+1)*2, :] is 2x4, not 2x2.
                
                # I think I need to split B's columns too. Let me reconsider.
                
                # A is 3x4, B is 4x4.
                # A = [A0 | A1] where A0 = A[:, 0:2], A1 = A[:, 2:4]
                # B = [[B00 | B01], [B10 | B11]] where B00 = B[0:2, 0:2], etc.
                
                # Z = A*B
                # Z[:, 0:2] = A0 * B00 + A1 * B10
                # Z[:, 2:4] = A0 * B01 + A1 * B11
                
                # So the 4 products are:
                # A0 * B00: A0 is 3x2, B00 is 2x2
                # A1 * B10: A1 is 3x2, B10 is 2x2
                # A0 * B01: A0 is 3x2, B01 is 2x2
                # A1 * B11: A1 is 3x2, B11 is 2x2
                
                # For product A_part * B_part (both as above):
                # A_part[i, j] where i in {0,1,2}, j in {0,1}
                # B_part[j, k] where j in {0,1}, k in {0,1}
                # Result[i, k]
                
                # Global indices:
                # A_part column j → global A column: ai*2 + j
                # B_part row j → global B row: bi*2 + j
                # B_part column k → global B column: bi*2 + k
                # Result row i → global row i
                # Result column k → global column: bi*2 + k
                
                # U index: i * m + (ai*2 + j) = i*4 + ai*2 + j
                # V index: (bi*2 + j) * p + (bi*2 + k) = (bi*2+j)*4 + bi*2 + k
                # W index: (bi*2 + k) * n + i = (bi*2+k)*3 + i
                
                base_col = (ai * 2 + bi) * 11  # 11 columns per block
                # Strassen part (top 2 rows of A_part)
                for c in range(7):
                    for i in range(2):  # top 2 rows
                        for j in range(2):
                            u_val = strassen_u[c][i * 2 + j]
                            if u_val != 0:
                                U[i * 4 + ai * 2 + j][base_col + c] = u_val
                    for j in range(2):
                        for k in range(2):
                            v_val = strassen_v[c][j * 2 + k]
                            if v_val != 0:
                                V[(bi * 2 + j) * 4 + bi * 2 + k][base_col + c] = v_val
                    for k in range(2):
                        for i in range(2):
                            w_val = strassen_w[c][k * 2 + i]
                            if w_val != 0:
                                W[(bi * 2 + k) * 3 + i][base_col + c] = w_val
                
                # Trivial part (bottom 1 row of A_part)
                for j in range(2):
                    for k in range(2):
                        col = base_col + 7 + j * 2 + k
                        U[2 * 4 + ai * 2 + j][col] = 1  # i=2
                        V[(bi * 2 + j) * 4 + bi * 2 + k][col] = 1
                        W[(bi * 2 + k) * 3 + 2][col] = 1
        
        # Verify full decomposition
        def verify_full(U_mat, V_mat, W_mat, n, m, p):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for c in range(len(U_mat[0])):
                            total += U_mat[i * m + j][c] * V_mat[j * p + k][c] * W_mat[k * n + i][c]
                        if total != 1:
                            return False
            return True
        
        if verify_full(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: trivial rank-48 decomposition
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
# EVOLVE-BLOCK-END
