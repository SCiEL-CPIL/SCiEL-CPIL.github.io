import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[:, 0:2] (2x2) * B[0:2, :] (2x9)
    # Split B[0:2, :] into 4 blocks of 2 columns and 1 block of 1 column.
    
    # 4 Strassen blocks for B columns 0-7
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # A indices
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        # B indices
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        # C output indices
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # Strassen products
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1; U[u_d][col] += 1
        V[v_e][col] += 1; V[v_h][col] += 1
        # M2 = (c+d)(e)
        U[u_c][col+1] += 1; U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        # M3 = (a)(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1; V[v_h][col+2] -= 1
        # M4 = (d)(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1; V[v_e][col+3] -= 1
        # M5 = (a+b)(h)
        U[u_a][col+4] += 1; U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1
        V[v_e][col+5] += 1; V[v_f][col+5] += 1
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1
        V[v_g][col+6] += 1; V[v_h][col+6] += 1
        
        # Reconstruct
        W[w_00][col] += 1; W[w_00][col+3] += 1
        W[w_00][col+4] -= 1; W[w_00][col+6] += 1
        W[w_01][col+2] += 1; W[w_01][col+4] += 1
        W[w_10][col+1] += 1; W[w_10][col+3] += 1
        W[w_11][col] += 1; W[w_11][col+1] -= 1
        W[w_11][col+2] += 1; W[w_11][col+5] += 1
        
        col += 7
        
    # 1 Direct block for B column 8
    k8 = 8
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    U[u_a][col] += 1; V[v_e][col] += 1; W[w_08][col] += 1
    U[u_b][col+1] += 1; V[v_g][col+1] += 1; W[w_08][col+1] += 1
    U[u_c][col+2] += 1; V[v_e][col+2] += 1; W[w_18][col+2] += 1
    U[u_d][col+3] += 1; V[v_g][col+3] += 1; W[w_18][col+3] += 1
    
    col += 4
    
    # Block 2: A[:, 2] (2x1) * B[2, :] (1x9)
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2
    
    for k in range(9):
        v_b = 2 * p + k
        w_0k = k * n + 0
        w_1k = k * n + 1
        
        U[u_ar0][col] += 1; V[v_b][col] += 1; W[w_0k][col] += 1
        U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[w_1k][col+1] += 1
        
        col += 2
        
    return {"U": U, "V": V, "W": W}
