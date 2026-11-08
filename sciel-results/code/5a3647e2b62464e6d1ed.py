import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # u_base: index in U for i=0, j=0
    # v_base: index in V for j=0, k=0
    # w_base: index in W for k=0, i=0
    # i_step, j_step, k_step: steps for global indices
    # U index: u_base + i*i_step_u + j*j_step_u
    # V index: v_base + j*j_step_v + k*k_step_v
    # W index: w_base + k*k_step_w + i*i_step_w
    
    def add_strassen_222(u_base, v_base, w_base, i_step_u, j_step_u, j_step_v, k_step_v, k_step_w, i_step_w):
        nonlocal col
        # Strassen coefficients
        # M1 = (x11+x22)(y11+y22)
        # M2 = (x21+x22)y11
        # M3 = x11(y12-y22)
        # M4 = x22(y21-y11)
        # M5 = (x11+x12)y22
        # M6 = (x21-x11)(y11+y12)
        # M7 = (x12-x22)(y21+y22)
        
        # M1
        # u: x11, x22 -> (1,1)
        # v: y11, y22 -> (1,1)
        # w: z11, z22 -> (1,1) with signs +, -
        U[u_base + 0*i_step_u + 0*j_step_u][col] += 1
        U[u_base + 1*i_step_u + 1*j_step_u][col] += 1
        V[v_base + 0*j_step_v + 0*k_step_v][col] += 1
        V[v_base + 1*j_step_v + 1*k_step_v][col] += 1
        W[w_base + 0*k_step_w + 0*i_step_w][col] += 1  # z11
        W[w_base + 1*k_step_w + 1*i_step_w][col] -= 1  # z22
        col += 1
        
        # M2
        # u: x21, x22 -> (0,1)
        # v: y11 -> (1,0)
        # w: z21 -> (1,0)
        U[u_base + 1*i_step_u + 0*j_step_u][col] += 1
        U[u_base + 1*i_step_u + 1*j_step_u][col] += 1
        V[v_base + 0*j_step_v + 0*k_step_v][col] += 1
        W[w_base + 0*k_step_w + 1*i_step_w][col] += 1  # z21
        col += 1
        
        # M3
        # u: x11 -> (1,0)
        # v: y12, y22 -> (1,-1)
        # w: z12 -> (0,1)
        U[u_base + 0*i_step_u + 0*j_step_u][col] += 1
        V[v_base + 0*j_step_v + 1*k_step_v][col] += 1
        V[v_base + 1*j_step_v + 1*k_step_v][col] -= 1
        W[w_base + 1*k_step_w + 0*i_step_w][col] += 1  # z12
        col += 1
        
        # M4
        # u: x22 -> (1,1)
        # v: y21, y11 -> (-1,1)
        # w: z11, z21 -> (1,1)
        U[u_base + 1*i_step_u + 1*j_step_u][col] += 1
        V[v_base + 1*j_step_v + 0*k_step_v][col] -= 1
        V[v_base + 0*j_step_v + 0*k_step_v][col] += 1
        W[w_base + 0*k_step_w + 0*i_step_w][col] += 1  # z11
        W[w_base + 0*k_step_w + 1*i_step_w][col] += 1  # z21
        col += 1
        
        # M5
        # u: x11, x12 -> (1,1)
        # v: y22 -> (1,1)
        # w: z11, z12 -> (-1,1)
        U[u_base + 0*i_step_u + 0*j_step_u][col] += 1
        U[u_base + 0*i_step_u + 1*j_step_u][col] += 1
        V[v_base + 1*j_step_v + 1*k_step_v][col] += 1
        W[w_base + 0*k_step_w + 0*i_step_w][col] -= 1  # z11
        W[w_base + 1*k_step_w + 0*i_step_w][col] += 1  # z12
        col += 1
        
        # M6
        # u: x21, x11 -> (1,-1)
        # v: y11, y12 -> (1,1)
        # w: z22 -> (0,1)
        U[u_base + 1*i_step_u + 0*j_step_u][col] += 1
        U[u_base + 0*i_step_u + 0*j_step_u][col] -= 1
        V[v_base + 0*j_step_v + 0*k_step_v][col] += 1
        V[v_base + 0*j_step_v + 1*k_step_v][col] += 1
        W[w_base + 1*k_step_w + 1*i_step_w][col] += 1  # z22
        col += 1
        
        # M7
        # u: x12, x22 -> (1,-1)
        # v: y21, y22 -> (1,1)
        # w: z11, z22 -> (1,1)
        U[u_base + 0*i_step_u + 1*j_step_u][col] += 1
        U[u_base + 1*i_step_u + 1*j_step_u][col] -= 1
        V[v_base + 1*j_step_v + 0*k_step_v][col] += 1
        V[v_base + 1*j_step_v + 1*k_step_v][col] += 1
        W[w_base + 0*k_step_w + 0*i_step_w][col] += 1  # z11
        W[w_base + 1*k_step_w + 1*i_step_w][col] += 1  # z22
        col += 1

    def add_schoolbook_221(u_base, v_base, w_base, i_step_u, j_step_u, j_step_v, k_step_v, k_step_w, i_step_w):
        nonlocal col
        # z1 = x11*y1 + x12*y2
        # z2 = x21*y1 + x22*y2
        
        # Term 1: x11 * y1 -> z1
        U[u_base + 0*i_step_u + 0*j_step_u][col] += 1
        V[v_base + 0*j_step_v + 0*k_step_v][col] += 1
        W[w_base + 0*k_step_w + 0*i_step_w][col] += 1
        col += 1
        
        # Term 2: x12 * y2 -> z1
        U[u_base + 0*i_step_u + 1*j_step_u][col] += 1
        V[v_base + 1*j_step_v + 0*k_step_v][col] += 1
        W[w_base + 0*k_step_w + 0*i_step_w][col] += 1
        col += 1
        
        # Term 3: x21 * y1 -> z2
        U[u_base + 1*i_step_u + 0*j_step_u][col] += 1
        V[v_base + 0*j_step_v + 0*k_step_v][col] += 1
        W[w_base + 0*k_step_w + 1*i_step_w][col] += 1
        col += 1
        
        # Term 4: x22 * y2 -> z2
        U[u_base + 1*i_step_u + 1*j_step_u][col] += 1
        V[v_base + 1*j_step_v + 0*k_step_v][col] += 1
        W[w_base + 0*k_step_w + 1*i_step_w][col] += 1
        col += 1

    # Part 1: j in {0, 1}
    # U base: i*4 + j. i_step=4, j_step=1.
    # V base: j*7 + k. j_step=7, k_step=1.
    # W base: k*2 + i. k_step=2, i_step=1.
    
    # Strassen Y1 (k=0,1)
    add_strassen_222(0, 0, 0, 4, 1, 7, 1, 2, 1)
    # Strassen Y2 (k=2,3)
    add_strassen_222(0, 2, 4, 4, 1, 7, 1, 2, 1)
    # Strassen Y3 (k=4,5)
    add_strassen_222(0, 4, 8, 4, 1, 7, 1, 2, 1)
    # Schoolbook y4 (k=6)
    add_schoolbook_221(0, 6, 12, 4, 1, 7, 1, 2, 1)
    
    # Part 2: j in {2, 3}
    # U base: i*4 + (2+j'). i_step=4, j_step=1. Base offset 2.
    # V base: (2+j')*7 + k. j_step=7, k_step=1. Base offset 14.
    # W base: k*2 + i. k_step=2, i_step=1.
    
    # Strassen Y1 (k=0,1)
    add_strassen_222(2, 14, 0, 4, 1, 7, 1, 2, 1)
    # Strassen Y2 (k=2,3)
    add_strassen_222(2, 16, 4, 4, 1, 7, 1, 2, 1)
    # Strassen Y3 (k=4,5)
    add_strassen_222(2, 18, 8, 4, 1, 7, 1, 2, 1)
    # Schoolbook y4 (k=6)
    add_schoolbook_221(2, 20, 12, 4, 1, 7, 1, 2, 1)

    return {"U": U, "V": V, "W": W}
