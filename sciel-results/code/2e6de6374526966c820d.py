import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Target: Rank 189 (27 * 7)
    # Structure: 3x3 blocks of 2x2 matrices.
    # T_6 = T_3 (Schoolbook, rank 27) x T_2 (Strassen, rank 7)
    
    n, m, p = DIMENSIONS
    
    # 1. Define Strassen (2x2) factors
    # U2, V2, W2 of size 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # P1 = (a+d)(e+h)
    # A: a(0,0), d(1,1) -> idx 0, 3
    U2[0][0] = 1
    U2[3][0] = 1
    # B: e(0,0), h(1,1) -> idx 0, 3
    V2[0][0] = 1
    V2[3][0] = 1
    # C: C00(0,0), C11(1,1) -> idx 0, 3
    W2[0][0] = 1
    W2[3][0] = 1
    
    # P2 = (c+d)e
    # A: c(1,0), d(1,1) -> idx 2, 3
    U2[2][1] = 1
    U2[3][1] = 1
    # B: e(0,0) -> idx 0
    V2[0][1] = 1
    # C: C10(1,0) +1, C11(1,1) -1
    # C10: i=1, k=0 -> idx 1
    # C11: i=1, k=1 -> idx 3
    W2[1][1] = 1
    W2[3][1] = -1
    
    # P3 = a(f-h)
    # A: a(0,0) -> idx 0
    U2[0][2] = 1
    # B: f(0,1), h(1,1) -> idx 1, 3
    V2[1][2] = 1
    V2[3][2] = -1
    # C: C01(0,1) +1, C11(1,1) +1
    # C01: i=0, k=1 -> idx 2
    # C11: i=1, k=1 -> idx 3
    W2[2][2] = 1
    W2[3][2] = 1
    
    # P4 = d(g-e)
    # A: d(1,1) -> idx 3
    U2[3][3] = 1
    # B: g(1,0), e(0,0) -> idx 2, 0
    V2[2][3] = 1
    V2[0][3] = -1
    # C: C00(0,0) +1, C10(1,0) +1
    # C00: i=0, k=0 -> idx 0
    # C10: i=1, k=0 -> idx 1
    W2[0][3] = 1
    W2[1][3] = 1
    
    # P5 = (a+b)h
    # A: a(0,0), b(0,1) -> idx 0, 1
    U2[0][4] = 1
    U2[1][4] = 1
    # B: h(1,1) -> idx 3
    V2[3][4] = 1
    # C: C00(0,0) -1, C01(0,1) +1
    # C00: i=0, k=0 -> idx 0
    # C01: i=0, k=1 -> idx 2
    W2[0][4] = -1
    W2[2][4] = 1
    
    # P6 = (c-a)(e+f)
    # A: c(1,0), a(0,0) -> idx 2, 0
    U2[2][5] = 1
    U2[0][5] = -1
    # B: e(0,0), f(0,1) -> idx 0, 1
    V2[0][5] = 1
    V2[1][5] = 1
    # C: C11(1,1) +1
    # C11: i=1, k=1 -> idx 3
    W2[3][5] = 1
    
    # P7 = (b-d)(g+h)
    # A: b(0,1), d(1,1) -> idx 1, 3
    U2[1][6] = 1
    U2[3][6] = -1
    # B: g(1,0), h(1,1) -> idx 2, 3
    V2[2][6] = 1
    V2[3][6] = 1
    # C: C00(0,0) +1
    # C00: i=0, k=0 -> idx 0
    W2[0][6] = 1
    
    # 2. Define Schoolbook (3x3) factors
    # U3, V3, W3 of size 9x27
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3 + j][col] = 1
                V3[j*3 + k][col] = 1
                W3[k*3 + i][col] = 1
                col += 1
                
    # 3. Combine
    rank_2 = 7
    rank_3 = 27
    total_rank = rank_2 * rank_3 # 189
    
    U = [[0] * total_rank for _ in range(36)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # Mapping:
    # i = i3*2 + i2
    # j = j3*2 + j2
    # k = k3*2 + k2
    # u_idx = i*6 + j
    # v_idx = j*6 + k
    # w_idx = k*6 + i
    
    for r3 in range(rank_3):
        for r2 in range(rank_2):
            r = r3 * rank_2 + r2
            
            for i3 in range(3):
                for j3 in range(3):
                    for k3 in range(3):
                        # We need to iterate over the non-zero elements of U3, V3, W3
                        # But it's easier to just compute the full product for all i2, j2, k2
                        # Since U2, V2, W2 are sparse, we can just loop over all 4 indices for 2x2
                        # There are 4*4*4 = 64 combinations of (i2, j2, k2) per (r3, r2)
                        # Total ops: 189 * 64 * 3 matrices ~ 36k ops. Very fast.
                        
                        # Actually, we can just fill U, V, W directly
                        # U[u_idx, r] = U3[i3*3+j3, r3] * U2[i2*2+j2, r2]
                        
                        # Iterate over all i2, j2
                        for i2 in range(2):
                            for j2 in range(2):
                                u_idx = (i3*2 + i2)*6 + (j3*2 + j2)
                                val_u = U3[i3*3 + j3][r3] * U2[i2*2 + j2][r2]
                                U[u_idx][r] = val_u
                                
                        # Iterate over all j2, k2
                        for j2 in range(2):
                            for k2 in range(2):
                                v_idx = (j3*2 + j2)*6 + (k3*2 + k2)
                                val_v = V3[j3*3 + k3][r3] * V2[j2*2 + k2][r2]
                                V[v_idx][r] = val_v
                                
                        # Iterate over all k2, i2
                        for k2 in range(2):
                            for i2 in range(2):
                                w_idx = (k3*2 + k2)*6 + (i3*2 + i2)
                                val_w = W3[k3*3 + i3][r3] * W2[k2*2 + i2][r2]
                                W[w_idx][r] = val_w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
