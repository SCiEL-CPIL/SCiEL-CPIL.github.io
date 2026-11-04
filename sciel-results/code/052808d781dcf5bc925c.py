import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def get_strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct_4x4x4_strassen():
    """Construct rank-49 decomposition for 4x4x4 matrix multiplication via Kronecker product of 2x2 Strassen."""
    U2, V2, W2 = get_strassen_2x2()
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    I = (i1*2+i2)*4 + (j1*2+j2)
                    u_idx1 = i1*2+j1
                    u_idx2 = i2*2+j2
                    for t1 in range(7):
                        for t2 in range(7):
                            R = t1*7+t2
                            U4[I][R] = U2[u_idx1][t1] * U2[u_idx2][t2]
                    
                    for k1 in range(2):
                        for k2 in range(2):
                            J = (j1*2+j2)*4 + (k1*2+k2)
                            v_idx1 = j1*2+k1
                            v_idx2 = j2*2+k2
                            for t1 in range(7):
                                for t2 in range(7):
                                    R = t1*7+t2
                                    V4[J][R] = V2[v_idx1][t1] * V2[v_idx2][t2]
                    
                    for k1 in range(2):
                        for k2 in range(2):
                            K = (k1*2+k2)*4 + (i1*2+i2)
                            w_idx1 = k1*2+i1
                            w_idx2 = k2*2+i2
                            for t1 in range(7):
                                for t2 in range(7):
                                    R = t1*7+t2
                                    W4[K][R] = W2[w_idx1][t1] * W2[w_idx2][t2]
    
    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140
    
    # Part 1: 4x4x4 Strassen, rank 49
    U4, V4, W4 = construct_4x4x4_strassen()
    
    # Part 2: 4x4x3 schoolbook, rank 48
    rank2 = 48
    U2 = [[0]*rank2 for _ in range(16)]
    V2 = [[0]*rank2 for _ in range(12)]
    W2 = [[0]*rank2 for _ in range(12)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U2[i*4+j][col] = 1
                V2[j*3+k][col] = 1
                W2[k*4+i][col] = 1
                col += 1
    
    total_rank = 49 + 48
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    # Fill U
    for i in range(16):
        for t in range(49):
            U[i][t] = U4[i][t]
        for t in range(48):
            U[i][49+t] = U2[i][t]
    
    # Fill V: V4 row j*4+k -> V row j*7+k (k=0..3)
    for j in range(4):
        for k in range(4):
            v4_row = j*4+k
            v_row = j*7+k
            for t in range(49):
                V[v_row][t] = V4[v4_row][t]
    
    # V2 row j*3+k' -> V row j*7+(4+k') (k'=0..2)
    for j in range(4):
        for k in range(3):
            v2_row = j*3+k
            v_row = j*7+(4+k)
            for t in range(48):
                V[v_row][49+t] = V2[v2_row][t]
    
    # Fill W: W4 row k*4+i -> W row k*4+i (k=0..3)
    for k in range(4):
        for i in range(4):
            w4_row = k*4+i
            w_row = k*4+i
            for t in range(49):
                W[w_row][t] = W4[w4_row][t]
    
    # W2 row k'*4+i -> W row (4+k')*4+i (k'=0..2)
    for k in range(3):
        for i in range(4):
            w2_row = k*4+i
            w_row = (4+k)*4+i
            for t in range(48):
                W[w_row][49+t] = W2[w2_row][t]
    
    # Quick verification
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        ok = True
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    s = 0
                    for t in range(total_rank):
                        s += U[i*4+j][t] * V[j*7+k][t] * W[k*4+i][t]
                    if abs(s - 1) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if not ok:
            rank = 4*4*7
            U_fb = [[0]*rank for _ in range(16)]
            V_fb = [[0]*rank for _ in range(28)]
            W_fb = [[0]*rank for _ in range(28)]
            c = 0
            for i in range(4):
                for j in range(4):
                    for k in range(7):
                        U_fb[i*4+j][c] = 1
                        V_fb[j*7+k][c] = 1
                        W_fb[k*4+i][c] = 1
                        c += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
