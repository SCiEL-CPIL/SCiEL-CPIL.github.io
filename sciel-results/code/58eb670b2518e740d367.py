import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Rank-4 Winograd scheme for 2x2 matrix multiplication
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # M0 = (a0+a1+a2+a3)(b0+b2+b1+b3)
    # M1 = (a0+a1-a2-a3)(b0-b2+b1-b3)
    # M2 = (a0-a1+a2-a3)(b0+b2-b1-b3)
    # M3 = (a0-a1-a2+a3)(b0-b2-b1+b3)
    # c00 = (M0+M1+M2+M3)/4
    # c01 = (M0-M1+M2-M3)/4
    # c10 = (M0+M1-M2-M3)/4
    # c11 = (M0-M1-M2+M3)/4
    
    # For 2x2 * 2x8: split B into 4 pairs of columns
    # Each pair uses the rank-4 Winograd scheme
    # Total: 4 (Winograd) * 4 (column pairs) = 16 rank-1 terms per 2x2*2x8
    
    # For 2x4 * 4x8:
    # A = [A0 | A1] where A0, A1 are 2x2
    # B = [B0; B1] where B0, B1 are 2x8
    # C = A0*B0 + A1*B1
    # Each 2x2*2x8 uses 16 rank-1 terms
    # Total: 2 * 16 = 32
    
    total_rank = 32
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Winograd coefficients for 2x2
    # U coefficients (for A matrix, 2x2):
    # M0: a0+a1+a2+a3 -> [1, 1, 1, 1]
    # M1: a0+a1-a2-a3 -> [1, 1, -1, -1]
    # M2: a0-a1+a2-a3 -> [1, -1, 1, -1]
    # M3: a0-a1-a2+a3 -> [1, -1, -1, 1]
    U_wino = np.array([
        [1, 1, 1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1]
    ], dtype=float)
    
    # V coefficients (for B matrix, 2x2):
    # M0: b0+b2+b1+b3 -> [1, 1, 1, 1] (b0,b1 from row0, b2,b3 from row1)
    #     Actually: b0+b1+b2+b3 -> indices [0,1,2,3] for [b0,b1,b2,b3]
    #     Wait, B = [[b0, b1], [b2, b3]], so b0 is (0,0), b1 is (0,1), b2 is (1,0), b3 is (1,1)
    #     M0: (b0+b2+b1+b3) = b0+b1+b2+b3 -> [1, 1, 1, 1]
    # M1: (b0-b2+b1-b3) = b0+b1-b2-b3 -> [1, 1, -1, -1]
    # M2: (b0+b2-b1-b3) = b0-b1+b2-b3 -> [1, -1, 1, -1]
    # M3: (b0-b2-b1+b3) = b0-b1-b2+b3 -> [1, -1, -1, 1]
    V_wino = np.array([
        [1, 1, 1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1]
    ], dtype=float)
    
    # W coefficients (for C matrix, 2x2):
    # c00 = (M0+M1+M2+M3)/4 -> [1/4, 1/4, 1/4, 1/4]
    # c01 = (M0-M1+M2-M3)/4 -> [1/4, -1/4, 1/4, -1/4]
    # c10 = (M0+M1-M2-M3)/4 -> [1/4, 1/4, -1/4, -1/4]
    # c11 = (M0-M1-M2+M3)/4 -> [1/4, -1/4, -1/4, 1/4]
    W_wino = np.array([
        [1/4, 1/4, 1/4, 1/4],
        [1/4, -1/4, 1/4, -1/4],
        [1/4, 1/4, -1/4, -1/4],
        [1/4, -1/4, -1/4, 1/4]
    ], dtype=float)
    
    # For 2x2 * 2x8:
    # A is 2x2 (indices a0,a1,a2,a3)
    # B is 2x8, split into 4 pairs: (b0,b1), (b2,b3), (b4,b5), (b6,b7)
    # Each pair uses the rank-4 Winograd scheme
    
    # For the first 2x2*2x8 (A0 * B0):
    # A0 is rows 0-1 of A, columns 0-1
    # B0 is rows 0-1 of B, all 8 columns
    
    # For the second 2x2*2x8 (A1 * B1):
    # A1 is rows 0-1 of A, columns 2-3
    # B1 is rows 2-3 of B, all 8 columns
    
    # Process each 2x2*2x8 block
    for block in range(2):
        # block 0: A0 (cols 0-1) * B0 (rows 0-1)
        # block 1: A1 (cols 2-3) * B1 (rows 2-3)
        
        a_col_offset = block * 2  # 0 or 2
        b_row_offset = block * 2  # 0 or 2
        
        for col_pair in range(4):
            # col_pair 0: B columns 0-1
            # col_pair 1: B columns 2-3
            # col_pair 2: B columns 4-5
            # col_pair 3: B columns 6-7
            b_col_offset = col_pair * 2  # 0, 2, 4, 6
            
            base_r = block * 16 + col_pair * 4
            
            for w in range(4):
                r = base_r + w
                
                # U coefficients (for A)
                # A indices: a0=(0,a_col_offset), a1=(0,a_col_offset+1), a2=(1,a_col_offset), a3=(1,a_col_offset+1)
                U[0 * 4 + a_col_offset + 0, r] += U_wino[0, w]
                U[0 * 4 + a_col_offset + 1, r] += U_wino[1, w]
                U[1 * 4 + a_col_offset + 0, r] += U_wino[2, w]
                U[1 * 4 + a_col_offset + 1, r] += U_wino[3, w]
                
                # V coefficients (for B)
                # B indices: b0=(b_row_offset, b_col_offset), b1=(b_row_offset, b_col_offset+1),
                #            b2=(b_row_offset+1, b_col_offset), b3=(b_row_offset+1, b_col_offset+1)
                V[(b_row_offset + 0) * 8 + b_col_offset + 0, r] += V_wino[0, w]
                V[(b_row_offset + 0) * 8 + b_col_offset + 1, r] += V_wino[1, w]
                V[(b_row_offset + 1) * 8 + b_col_offset + 0, r] += V_wino[2, w]
                V[(b_row_offset + 1) * 8 + b_col_offset + 1, r] += V_wino[3, w]
                
                # W coefficients (for C)
                # C indices: c00=(0, 2*col_pair), c01=(0, 2*col_pair+1),
                #            c10=(1, 2*col_pair), c11=(1, 2*col_pair+1)
                # W index: k*n+i where k is column of C, i is row of C
                # c00: i=0, k=2*col_pair -> W[2*col_pair*2 + 0, r]
                # c01: i=0, k=2*col_pair+1 -> W[(2*col_pair+1)*2 + 0, r]
                # c10: i=1, k=2*col_pair -> W[2*col_pair*2 + 1, r]
                # c11: i=1, k=2*col_pair+1 -> W[(2*col_pair+1)*2 + 1, r]
                W[(2 * col_pair + 0) * 2 + 0, r] += W_wino[0, w]
                W[(2 * col_pair + 1) * 2 + 0, r] += W_wino[1, w]
                W[(2 * col_pair + 0) * 2 + 1, r] += W_wino[2, w]
                W[(2 * col_pair + 1) * 2 + 1, r] += W_wino[3, w]
    
    # Verify the decomposition
    def verify_decomposition(U_mat, V_mat, W_mat):
        R = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False, (i, j, k, s)
        return True, None
    
    valid, err = verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to rank 56 Strassen
        U56 = np.zeros((8, 56), dtype=int)
        V56 = np.zeros((32, 56), dtype=int)
        W56 = np.zeros((16, 56), dtype=int)
        
        U_local = np.zeros((4, 7), dtype=int)
        V_local = np.zeros((4, 7), dtype=int)
        W_local = np.zeros((4, 7), dtype=int)
        
        U_local[0,0] = 1; U_local[3,0] = 1
        V_local[0,0] = 1; V_local[3,0] = 1
        W_local[0,0] = 1; W_local[3,0] = 1
        
        U_local[2,1] = 1; U_local[3,1] = 1
        V_local[0,1] = 1
        W_local[2,1] = 1; W_local[3,1] = -1
        
        U_local[0,2] = 1
        V_local[1,2] = 1; V_local[3,2] = -1
        W_local[1,2] = 1; W_local[3,2] = 1
        
        U_local[3,3] = 1
        V_local[2,3] = 1; V_local[0,3] = -1
        W_local[0,3] = 1; W_local[2,3] = 1
        
        U_local[0,4] = 1; U_local[1,4] = 1
        V_local[3,4] = 1
        W_local[0,4] = -1; W_local[1,4] = 1
        
        U_local[2,5] = 1; U_local[0,5] = -1
        V_local[0,5] = 1; V_local[1,5] = 1
        W_local[3,5] = 1
        
        U_local[1,6] = 1; U_local[3,6] = -1
        V_local[2,6] = 1; V_local[3,6] = 1
        W_local[0,6] = 1
        
        for j in range(4):
            for s in range(2):
                base_r = (j * 2 + s) * 7
                a_map = [0, 1, 4, 5] if s == 0 else [2, 3, 6, 7]
                if s == 0:
                    b_map = [0*8+2*j, 0*8+2*j+1, 1*8+2*j, 1*8+2*j+1]
                else:
                    b_map = [2*8+2*j, 2*8+2*j+1, 3*8+2*j, 3*8+2*j+1]
                c_map = [2*j*2+0, (2*j+1)*2+0, 2*j*2+1, (2*j+1)*2+1]
                
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        val = U_local[la, l]
                        if val != 0:
                            U56[a_map[la], r] += val
                    for lb in range(4):
                        val = V_local[lb, l]
                        if val != 0:
                            V56[b_map[lb], r] += val
                    for lc in range(4):
                        val = W_local[lc, l]
                        if val != 0:
                            W56[c_map[lc], r] += val
        
        return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
    
    # Convert to list
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
