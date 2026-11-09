import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(u_idx, v_idx, w_idx, col_start):
        # u_idx: [u11, u12, u21, u22]
        # v_idx: [v11, v12, v21, v22]
        # w_idx: [w11, w12, w21, w22]
        # col_start: index in U,V,W to start writing to
        
        u11, u12, u21, u22 = u_idx
        v11, v12, v21, v22 = v_idx
        w11, w12, w21, w22 = w_idx
        
        # P1 = (a11 + a22)(b11 + b22)
        c = col_start
        U[u11][c] += 1; U[u22][c] += 1
        V[v11][c] += 1; V[v22][c] += 1
        W[w11][c] += 1; W[w22][c] += 1
        c += 1
        
        # P2 = (a21 + a22)b11
        U[u21][c] += 1; U[u22][c] += 1
        V[v11][c] += 1
        W[w21][c] += 1; W[w22][c] += 1
        c += 1
        
        # P3 = a11(b12 - b22)
        U[u11][c] += 1
        V[v12][c] += 1; V[v22][c] -= 1
        W[w12][c] += 1; W[w22][c] -= 1
        c += 1
        
        # P4 = a22(b21 - b11)
        U[u22][c] += 1
        V[v21][c] += 1; V[v11][c] -= 1
        W[w11][c] -= 1; W[w21][c] += 1
        c += 1
        
        # P5 = (a11 + a12)b22
        U[u11][c] += 1; U[u12][c] += 1
        V[v22][c] += 1
        W[w12][c] += 1; W[w22][c] += 1
        c += 1
        
        # P6 = (a21 - a11)(b11 + b12)
        U[u21][c] += 1; U[u11][c] -= 1
        V[v11][c] += 1; V[v12][c] += 1
        W[w11][c] += 1; W[w21][c] += 1
        c += 1
        
        # P7 = (a12 - a22)(b21 + b22)
        U[u12][c] += 1; U[u22][c] -= 1
        V[v21][c] += 1; V[v22][c] += 1
        W[w11][c] += 1; W[w12][c] += 1
        c += 1
        
        return c

    # Block 1: A_L (cols 1,2) * B_T1 (rows 1,2, cols 1,2)
    # A_L indices: i in 1,2; j in 1,2
    # U[i*m+j] -> U[1*3+1]=4, U[1*3+2]=5, U[2*3+1]=7, U[2*3+2]=8
    u_idx_1 = [4, 5, 7, 8]
    # B_T1 indices: j in 1,2; k in 1,2
    # V[j*p+k] -> V[1*6+1]=7, V[1*6+2]=8, V[2*6+1]=13, V[2*6+2]=14
    v_idx_1 = [7, 8, 13, 14]
    # C_1 indices: i in 1,2; k in 1,2
    # W[k*n+i] -> W[1*2+1]=3, W[2*2+1]=5, W[1*2+2]=4, W[2*2+2]=6
    w_idx_1 = [3, 5, 4, 6]
    
    col = 0
    col = add_strassen_block(u_idx_1, v_idx_1, w_idx_1, col)
    
    # Block 2: A_L (cols 1,2) * B_T2 (rows 1,2, cols 3,4)
    # V indices for k=3,4
    v_idx_2 = [9, 10, 15, 16] # V[1*6+3], V[1*6+4], V[2*6+3], V[2*6+4]
    # W indices for k=3,4
    w_idx_2 = [7, 9, 8, 10]   # W[3*2+1], W[4*2+1], W[3*2+2], W[4*2+2]
    
    col = add_strassen_block(u_idx_1, v_idx_2, w_idx_2, col)
    
    # Block 3: A_L (cols 1,2) * B_T3 (rows 1,2, cols 5,6)
    # V indices for k=5,6
    v_idx_3 = [11, 12, 17, 18] # V[1*6+5], V[1*6+6], V[2*6+5], V[2*6+6]
    # W indices for k=5,6
    w_idx_3 = [11, 13, 12, 14] # W[5*2+1], W[6*2+1], W[5*2+2], W[6*2+2]
    
    col = add_strassen_block(u_idx_1, v_idx_3, w_idx_3, col)
    
    # Block 4: A_R (col 3) * B_B (row 3)
    # A_R indices: i in 1,2; j=3
    # U[1*3+3]=6, U[2*3+3]=9
    u_13 = 6
    u_23 = 9
    
    # B_B indices: j=3; k in 1..6
    # V[3*6+k] = 18+k
    
    # C_R indices: i in 1,2; k in 1..6
    # W[k*2+i]
    
    for k in range(1, 7):
        # a13 * b3k -> c1k
        U[u_13][col] = 1
        V[18 + k][col] = 1
        W[2 * k + 1][col] = 1
        col += 1
        
        # a23 * b3k -> c2k
        U[u_23][col] = 1
        V[18 + k][col] = 1
        W[2 * k + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
