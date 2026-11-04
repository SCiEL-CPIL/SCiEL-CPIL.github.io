import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    def add_strassen_block(col_offset, u_idx, v_idx, w_idx):
        # u_idx, v_idx, w_idx are lists of 4 global indices for the 2x2 block
        # Mapping:
        # x00 -> u_idx[0], x01 -> u_idx[1], x10 -> u_idx[2], x11 -> u_idx[3]
        # y00 -> v_idx[0], y01 -> v_idx[1], y10 -> v_idx[2], y11 -> v_idx[3]
        # p00 -> w_idx[0], p01 -> w_idx[1], p10 -> w_idx[2], p11 -> w_idx[3]
        
        # M1 = (x00 + x11)(y00 + y11)
        U[u_idx[0]][col_offset + 0] += 1
        U[u_idx[3]][col_offset + 0] += 1
        V[v_idx[0]][col_offset + 0] += 1
        V[v_idx[3]][col_offset + 0] += 1
        W[w_idx[0]][col_offset + 0] += 1
        W[w_idx[3]][col_offset + 0] += 1
        
        # M2 = (x10 + x11) y00
        U[u_idx[2]][col_offset + 1] += 1
        U[u_idx[3]][col_offset + 1] += 1
        V[v_idx[0]][col_offset + 1] += 1
        W[w_idx[2]][col_offset + 1] += 1
        W[w_idx[3]][col_offset + 1] -= 1
        
        # M3 = x00 (y10 - y11)
        U[u_idx[0]][col_offset + 2] += 1
        V[v_idx[2]][col_offset + 2] += 1
        V[v_idx[3]][col_offset + 2] -= 1
        W[w_idx[1]][col_offset + 2] += 1
        W[w_idx[3]][col_offset + 2] += 1
        
        # M4 = x11 (y01 - y00)
        U[u_idx[3]][col_offset + 3] += 1
        V[v_idx[1]][col_offset + 3] += 1
        V[v_idx[0]][col_offset + 3] -= 1
        W[w_idx[0]][col_offset + 3] += 1
        W[w_idx[2]][col_offset + 3] += 1
        
        # M5 = (x00 + x01) y11
        U[u_idx[0]][col_offset + 4] += 1
        U[u_idx[1]][col_offset + 4] += 1
        V[v_idx[3]][col_offset + 4] += 1
        W[w_idx[0]][col_offset + 4] -= 1
        W[w_idx[1]][col_offset + 4] += 1
        
        # M6 = (x10 - x00)(y00 + y01)
        U[u_idx[2]][col_offset + 5] += 1
        U[u_idx[0]][col_offset + 5] -= 1
        V[v_idx[0]][col_offset + 5] += 1
        V[v_idx[1]][col_offset + 5] += 1
        W[w_idx[3]][col_offset + 5] += 1
        
        # M7 = (x01 - x11)(y10 + y11)
        U[u_idx[1]][col_offset + 6] += 1
        U[u_idx[3]][col_offset + 6] -= 1
        V[v_idx[2]][col_offset + 6] += 1
        V[v_idx[3]][col_offset + 6] += 1
        W[w_idx[0]][col_offset + 6] += 1

    # Block 1: A1 B11 -> C1 (left part of C)
    # A1: rows 0,1 of A (which are global U 0,1 for row 0 and 4,5 for row 1? No.
    # A is 2x4. U index = i * m + k.
    # i=0, k=0 -> U[0]
    # i=0, k=1 -> U[1]
    # i=1, k=0 -> U[4]
    # i=1, k=1 -> U[5]
    # So A1 (cols 0,1) uses U[0], U[1], U[4], U[5].
    # Order: x00=U[0], x01=U[1], x10=U[4], x11=U[5].
    u1 = [0, 1, 4, 5]
    
    # B11: rows 0,1 of B. V index = k * p + j.
    # k=0, j=0 -> V[0]
    # k=0, j=1 -> V[1]
    # k=1, j=0 -> V[4]
    # k=1, j=1 -> V[5]
    # Order: y00=V[0], y01=V[1], y10=V[4], y11=V[5].
    v1 = [0, 1, 4, 5]
    
    # C1: left part of C. W index = k * n + i.
    # C1 corresponds to C[0,0], C[0,1], C[1,0], C[1,1].
    # k=0, i=0 -> W[0]
    # k=1, i=0 -> W[1]
    # k=0, i=1 -> W[2]
    # k=1, i=1 -> W[3]
    # Order: p00=W[0], p01=W[1], p10=W[2], p11=W[3].
    w1 = [0, 1, 2, 3]
    
    add_strassen_block(0, u1, v1, w1)

    # Block 2: A2 B21 -> C1 (left part of C)
    # A2: cols 2,3 of A.
    # i=0, k=2 -> U[2]
    # i=0, k=3 -> U[3]
    # i=1, k=2 -> U[6]
    # i=1, k=3 -> U[7]
    u2 = [2, 3, 6, 7]
    
    # B21: rows 2,3 of B, cols 0,1.
    # k=2, j=0 -> V[8]
    # k=2, j=1 -> V[9]
    # k=3, j=0 -> V[12]
    # k=3, j=1 -> V[13]
    v2 = [8, 9, 12, 13]
    
    # C1: same as before.
    w2 = [0, 1, 2, 3]
    
    add_strassen_block(7, u2, v2, w2)

    # Block 3: A1 B12 -> C2 (right part of C)
    # A1: same as before.
    u3 = [0, 1, 4, 5]
    
    # B12: rows 0,1 of B, cols 2,3.
    # k=0, j=2 -> V[2]
    # k=0, j=3 -> V[3]
    # k=1, j=2 -> V[6]
    # k=1, j=3 -> V[7]
    v3 = [2, 3, 6, 7]
    
    # C2: right part of C.
    # k=2, i=0 -> W[4]
    # k=3, i=0 -> W[5]
    # k=2, i=1 -> W[6]
    # k=3, i=1 -> W[7]
    w3 = [4, 5, 6, 7]
    
    add_strassen_block(14, u3, v3, w3)

    # Block 4: A2 B22 -> C2 (right part of C)
    # A2: same as before.
    u4 = [2, 3, 6, 7]
    
    # B22: rows 2,3 of B, cols 2,3.
    # k=2, j=2 -> V[10]
    # k=2, j=3 -> V[11]
    # k=3, j=2 -> V[14]
    # k=3, j=3 -> V[15]
    v4 = [10, 11, 14, 15]
    
    # C2: same as before.
    w4 = [4, 5, 6, 7]
    
    add_strassen_block(21, u4, v4, w4)

    return {"U": U, "V": V, "W": W}
