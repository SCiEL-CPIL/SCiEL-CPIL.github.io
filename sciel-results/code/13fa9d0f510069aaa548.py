"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen parameters for 2x2 matrix multiplication
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U factors (A)
    # a=A00, b=A01, c=A10, d=A11
    # u0 = a+d
    # u1 = c+d
    # u2 = a
    # u3 = d
    # u4 = a+b
    # u5 = c-a
    # u6 = b-d
    
    U_factors = [
        [1, 0, 0, 1], # u0
        [0, 0, 1, 1], # u1
        [1, 0, 0, 0], # u2
        [0, 0, 0, 1], # u3
        [1, 1, 0, 0], # u4
        [-1, 0, 1, 0],# u5
        [0, 1, 0, -1] # u6
    ]
    
    # V factors (B) for a single 2x2 block
    # e=B00, f=B01, g=B10, h=B11
    # v0 = e+h
    # v1 = e
    # v2 = f-h
    # v3 = g-e
    # v4 = h
    # v5 = e+f
    # v6 = g+h
    
    V_factors = [
        [1, 0, 0, 1], # v0
        [1, 0, 0, 0], # v1
        [0, 1, 0, -1],# v2
        [0, 0, 1, -1],# v3
        [0, 0, 0, 1], # v4
        [1, 1, 0, 0], # v5
        [0, 0, 1, 1]  # v6
    ]
    
    # W factors (C)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # w_s[i] is coefficient of M_s in C_i
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    W_factors = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    num_blocks = p // 2 # 5 blocks
    
    for i in range(num_blocks):
        for s in range(7):
            c = i * 7 + s
            
            # Fill U
            for a_idx in range(4):
                U[a_idx][c] = U_factors[s][a_idx]
                
            # Fill V
            # B indices for block i:
            # e = B[0, 2i] -> V row 10*0 + 2i = 2i
            # f = B[0, 2i+1] -> V row 2i+1
            # g = B[1, 2i] -> V row 10*1 + 2i = 10+2i
            # h = B[1, 2i+1] -> V row 10+2i+1
            v_rows = [2*i, 2*i+1, 10+2*i, 10+2*i+1]
            for b_idx in range(4):
                V[v_rows[b_idx]][c] = V_factors[s][b_idx]
                
            # Fill W
            # C indices for block i:
            # C11 = C[0, 2i] -> W row 2*(2i) + 0 = 4i
            # C12 = C[0, 2i+1] -> W row 2*(2i+1) + 0 = 4i+2
            # C21 = C[1, 2i] -> W row 2*(2i) + 1 = 4i+1
            # C22 = C[1, 2i+1] -> W row 2*(2i+1) + 1 = 4i+3
            w_rows = [4*i, 4*i+2, 4*i+1, 4*i+3]
            for c_idx in range(4):
                W[w_rows[c_idx]][c] = W_factors[s][c_idx]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
