import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Schoolbook rank as fallback
    rank_school = n * m * p  # 64
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Strassen-based decomposition: rank 56
    # Split A (2x4) into two 2x2 blocks: A0 = A[:, 0:2], A1 = A[:, 2:4]
    # Split B (4x8) into two 2x8 halves: B0 = B[0:2, :], B1 = B[2:4, :]
    # Each 2x2 * 2x8 = four 2x2 * 2x2 multiplications
    # Total: 8 sub-multiplications, each rank 7 -> rank 56
    
    # Strassen 7 patterns for 2x2*2x2:
    # For X=[[x0,x1],[x2,x3]], Y=[[y0,y1],[y2,y3]], Z=X*Y
    # M0 = (x0+x3)*(y0+y3)
    # M1 = (x2+x3)*y0
    # M2 = x0*(y1-y3)
    # M3 = x3*(y2-y0)
    # M4 = (x0+x1)*y3
    # M5 = (x2-x0)*(y0+y1)
    # M6 = (x1-x3)*(y2+y3)
    # Z00 = M0+M3-M4+M6
    # Z01 = M2+M4
    # Z10 = M1+M3
    # Z11 = M0-M1+M2+M5
    
    # U vectors over (x0,x1,x2,x3) i.e. A[i, j_local] for i in {0,1}, j_local in {0,1}
    # V vectors over (y0,y1,y2,y3) i.e. B[j_local, k_local]
    # W vectors over (z00,z01,z10,z11) i.e. C[i, k_local]
    
    U_strassen = [
        [1, 0, 0, 1],  # M0: x0+x3
        [0, 0, 1, 1],  # M1: x2+x3
        [1, 0, 0, 0],  # M2: x0
        [0, 0, 0, 1],  # M3: x3
        [1, 1, 0, 0],  # M4: x0+x1
        [1, 0, -1, 0], # M5: x2-x0
        [0, 1, 0, -1], # M6: x1-x3
    ]
    
    V_strassen = [
        [1, 0, 0, 1],  # M0: y0+y3
        [1, 0, 0, 0],  # M1: y0
        [0, 1, 0, -1], # M2: y1-y3
        [-1, 0, 1, 0], # M3: y2-y0
        [0, 0, 0, 1],  # M4: y3
        [1, 1, 0, 0],  # M5: y0+y1
        [0, 0, 1, 1],  # M6: y2+y3
    ]
    
    W_strassen = [
        [1, 0, 0, 0],  # M0 -> z00
        [0, 0, 0, 0],  # M1 -> (handled in z10, z11)
        [0, 0, 0, 0],  # M2 -> (handled in z01, z11)
        [0, 0, 0, 0],  # M3 -> (handled in z00, z10)
        [0, 0, 0, 0],  # M4 -> (handled in z00, z01)
        [0, 0, 0, 0],  # M5 -> (handled in z11)
        [0, 0, 0, 0],  # M6 -> (handled in z00)
    ]
    
    # Actually, let me define W properly for each term's contribution to each z element:
    # z00 = M0 + M3 - M4 + M6
    # z01 = M2 + M4
    # z10 = M1 + M3
    # z11 = M0 - M1 + M2 + M5
    
    W_strassen = [
        [1, 0, 0, 1],  # M0: contributes to z00 (+1) and z11 (+1)
        [0, 0, 1, -1], # M1: contributes to z10 (+1) and z11 (-1)
        [0, 1, 0, 1],  # M2: contributes to z01 (+1) and z11 (+1)
        [1, 0, 1, 0],  # M3: contributes to z00 (+1) and z10 (+1)
        [-1, 1, 0, 0], # M4: contributes to z00 (-1) and z01 (+1)
        [0, 0, 0, 1],  # M5: contributes to z11 (+1)
        [1, 0, 0, 0],  # M6: contributes to z00 (+1)
    ]
    
    # Verify Strassen for 2x2x2x2
    def verify_2x2():
        for c in range(7):
            pass
        # Check: sum_c U[c][i*2+j] * V[c][j*2+k] * W[c][k*2+i] == 1 for all i,j,k in {0,1}
        U_arr = np.array(U_strassen)  # (7, 4)
        V_arr = np.array(V_strassen)  # (7, 4)
        W_arr = np.array(W_strassen)  # (7, 4)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for c in range(7):
                        s += U_arr[c, i*2+j] * V_arr[c, j*2+k] * W_arr[c, k*2+i]
                    if s != 1:
                        return False
        return True
    
    if not verify_2x2():
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Build the full 2x4 x 4x8 decomposition
    rank = 56
    U_full = np.zeros((n * m, rank), dtype=int)  # (8, 56)
    V_full = np.zeros((m * p, rank), dtype=int)  # (32, 56)
    W_full = np.zeros((p * n, rank), dtype=int)  # (16, 56)
    
    U_arr = np.array(U_strassen)  # (7, 4)
    V_arr = np.array(V_strassen)  # (7, 4)
    W_arr = np.array(W_strassen)  # (7, 4)
    
    # For each of the 8 sub-blocks:
    # block_a in {0,1}: which 2x2 block of A (columns 0-1 or 2-3)
    # block_b in {0,1,2,3}: which 2x2 column block of the corresponding B half
    
    col_offset = 0
    for block_a in range(2):
        # A block: rows 0-1, cols block_a*2 to block_a*2+1
        # B half: rows block_a*2 to block_a*2+1
        for block_b in range(4):
            # B column block: cols block_b*2 to block_b*2+1
            for c in range(7):
                # U: A[i, block_a*2 + j_local] for i in {0,1}, j_local in {0,1}
                for i in range(2):
                    for j_local in range(2):
                        j = block_a * 2 + j_local
                        U_full[i * m + j, col_offset] = U_arr[c, i * 2 + j_local]
                
                # V: B[block_a*2 + j_local, block_b*2 + k_local] for j_local in {0,1}, k_local in {0,1}
                for j_local in range(2):
                    for k_local in range(2):
                        j = block_a * 2 + j_local
                        k = block_b * 2 + k_local
                        V_full[j * p + k, col_offset] = V_arr[c, j_local * 2 + k_local]
                
                # W: C[i, block_b*2 + k_local] for i in {0,1}, k_local in {0,1}
                for i in range(2):
                    for k_local in range(2):
                        k = block_b * 2 + k_local
                        W_full[k * n + i, col_offset] = W_arr[c, k_local * 2 + i]
                
                col_offset += 1
    
    # Verify the full decomposition
    def check_full(U, V, W, R):
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(R):
                            s += U[i*m+j][c] * V[jp*p+k][c] * W[k*n+i][c]
                        if j == jp:
                            if s != 1:
                                return False
                        else:
                            if s != 0:
                                return False
        return True
    
    if check_full(U_full.tolist(), V_full.tolist(), W_full.tolist(), rank):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
