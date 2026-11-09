import time
import sys

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Build M_2 (Strassen) factors
    # U2, V2, W2 are 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: m1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=1: m2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # r=2: m3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # r=3: m4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # r=4: m5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # r=5: m6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # r=6: m7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # Build M_3 (Schoolbook) factors
    # U3, V3, W3 are 9x27
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # U3 row i*3+j
                U3[i*3+j][col] = 1
                # V3 row j*3+k
                V3[j*3+k][col] = 1
                # W3 row k*3+i
                W3[k*3+i][col] = 1
                col += 1

    R3 = 27
    R2 = 7
    Rank = R3 * R2
    
    U = [[0]*Rank for _ in range(n*m)]
    V = [[0]*Rank for _ in range(m*p)]
    W = [[0]*Rank for _ in range(p*n)]
    
    # Fill U, V, W
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for c3 in range(R3):
                    if U3[i*3+j][c3] != 0:
                        u3_val = U3[i*3+j][c3]
                        v3_val = V3[j*3+k][c3]
                        w3_val = W3[k*3+i][c3]
                        
                        if u3_val == 0 and v3_val == 0 and w3_val == 0:
                            continue
                            
                        for c2 in range(R2):
                            r_total = c3 * R2 + c2
                            
                            # Sub-indices
                            # U2 index: i_sub * 2 + j_sub
                            # V2 index: j_sub * 2 + k_sub
                            # W2 index: k_sub * 2 + i_sub
                            
                            # We need to iterate over sub-indices to fill the 6x6 matrices
                            # This loop is heavy: 3*3*3*27*7 = 5103 iterations for the outer part
                            # Inner loop over 4 sub-indices for U, V, W?
                            # No, U2 is 4x7.
                            # For a fixed c2, U2[i_sub*2+j_sub][c2] is a value.
                            # We need to add to U[i*6+j, r_total].
                            # i = i_block*2 + i_sub
                            # j = j_block*2 + j_sub
                            
                            # It is more efficient to precompute the contributions.
                            # But 5103 * 4 * 4 * 4 is too much?
                            # 5103 * 64 ~ 326k operations. This is fine.
                            
                            # Actually, we can just loop i_sub, j_sub, k_sub
                            for i_sub in range(2):
                                for j_sub in range(2):
                                    u2_val = U2[i_sub*2 + j_sub][c2]
                                    if u2_val == 0:
                                        continue
                                    i_total = i * 2 + i_sub
                                    j_total = j * 2 + j_sub
                                    u_total = u3_val * u2_val
                                    U[i_total * 6 + j_total][r_total] = u_total
                                    
                            for j_sub in range(2):
                                for k_sub in range(2):
                                    v2_val = V2[j_sub*2 + k_sub][c2]
                                    if v2_val == 0:
                                        continue
                                    j_total = j * 2 + j_sub
                                    k_total = k * 2 + k_sub
                                    v_total = v3_val * v2_val
                                    V[j_total * 6 + k_total][r_total] = v_total
                                    
                            for k_sub in range(2):
                                for i_sub in range(2):
                                    w2_val = W2[k_sub*2 + i_sub][c2]
                                    if w2_val == 0:
                                        continue
                                    k_total = k * 2 + k_sub
                                    i_total = i * 2 + i_sub
                                    w_total = w3_val * w2_val
                                    W[k_total * 6 + i_total][r_total] = w_total
                                    
    if time.time() > deadline:
        # Return best found (which is this one)
        pass
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
