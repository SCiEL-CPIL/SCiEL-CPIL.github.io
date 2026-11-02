import numpy as np

def construct():
    n, m, p = 2, 2, 10
    rank = 35
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 * 2x2
    # Inputs: A = [[a, b], [c, d]], B = [[x, y], [z, w]]
    # M1 = (a+d)(x+w)
    # M2 = (c+d)x
    # M3 = a(y-w)
    # M4 = d(z-x)
    # M5 = (a+b)w
    # M6 = (c-a)(x+y)
    # M7 = (b-d)(z+w)
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3
    
    # Define the terms (u_idx, u_val, v_idx, v_val, w_idx, w_val)
    # u_idx is index in A (0:a, 1:b, 2:c, 3:d)
    # v_idx is index in B (0:x, 1:y, 2:z, 3:w) -> mapped to global
    # w_idx is index in C (0:C11, 1:C12, 2:C21, 3:C22) -> mapped to global
    
    # For each block l (0..4), B cols are 2l, 2l+1.
    # B indices: 0,1 for col 2l; 2,3 for col 2l+1? 
    # Global V index: j*p + k.
    # j=0, k=2l -> 2l
    # j=0, k=2l+1 -> 2l+1
    # j=1, k=2l -> p + 2l
    # j=1, k=2l+1 -> p + 2l+1
    
    # Global W index: k*n + i.
    # k=2l, i=0 -> 2l
    # k=2l, i=1 -> 2l+1
    # k=2l+1, i=0 -> 2l+2
    # k=2l+1, i=1 -> 2l+3
    
    # Global U index: i*m + j.
    # i=0, j=0 -> 0 (a)
    # i=0, j=1 -> 1 (b)
    # i=1, j=0 -> 2 (c)
    # i=1, j=1 -> 3 (d)

    for l in range(5):
        base_v0 = 2 * l
        base_v1 = 2 * l + 1
        base_v2 = p + 2 * l
        base_v3 = p + 2 * l + 1
        
        base_w0 = 2 * l
        base_w1 = 2 * l + 1
        base_w2 = 2 * l + 2
        base_w3 = 2 * l + 3
        
        base_col = 7 * l
        
        # M1
        U[0][base_col] += 1; U[3][base_col] += 1
        V[base_v0][base_col] += 1; V[base_v3][base_col] += 1
        W[base_w0][base_col] += 1; W[base_w3][base_col] += 1
        
        # M2
        U[2][base_col+1] += 1; U[3][base_col+1] += 1
        V[base_v0][base_col+1] += 1
        W[base_w1][base_col+1] += 1; W[base_w3][base_col+1] += -1
        
        # M3
        U[0][base_col+2] += 1
        V[base_v1][base_col+2] += 1; V[base_v3][base_col+2] += -1
        W[base_w2][base_col+2] += 1; W[base_w3][base_col+2] += 1
        
        # M4
        U[3][base_col+3] += 1
        V[base_v2][base_col+3] += 1; V[base_v0][base_col+3] += -1
        W[base_w0][base_col+3] += 1; W[base_w1][base_col+3] += 1
        
        # M5
        U[0][base_col+4] += 1; U[1][base_col+4] += 1
        V[base_v3][base_col+4] += 1
        W[base_w0][base_col+4] += -1; W[base_w2][base_col+4] += 1
        
        # M6
        U[2][base_col+5] += 1; U[0][base_col+5] += -1
        V[base_v0][base_col+5] += 1; V[base_v1][base_col+5] += 1
        W[base_w3][base_col+5] += 1
        
        # M7
        U[1][base_col+6] += 1; U[3][base_col+6] += -1
        V[base_v2][base_col+6] += 1; V[base_v3][base_col+6] += 1
        W[base_w0][base_col+6] += 1

    return {"U": U, "V": V, "W": W}
