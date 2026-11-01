"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x2 * 2x8
    # Split 2x8 into four 2x2 blocks
    # Each block uses rank 7 (Strassen)
    # Total rank 28
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U indices: 0:a00, 1:a01, 2:a10, 3:a11
    # V indices: j*p + k. 
    # Block m (0..3) uses cols 2m, 2m+1
    # V row for b_{0, 2m} is 0*8 + 2m = 2m
    # V row for b_{1, 2m} is 1*8 + 2m = 8 + 2m
    # V row for b_{0, 2m+1} is 0*8 + 2m+1 = 2m+1
    # V row for b_{1, 2m+1} is 1*8 + 2m+1 = 8 + 2m+1
    
    # W indices: k*n + i.
    # C_{0, 2m} -> w = 2m*2 + 0 = 4m
    # C_{1, 2m} -> w = 2m*2 + 1 = 4m+1
    # C_{0, 2m+1} -> w = (2m+1)*2 + 0 = 4m+2
    # C_{1, 2m+1} -> w = (2m+1)*2 + 1 = 4m+3
    
    for m in range(4):
        r0 = 7 * m
        # V indices
        v_b00 = 2 * m
        v_b10 = 8 + 2 * m
        v_b01 = 2 * m + 1
        v_b11 = 8 + 2 * m + 1
        
        # W indices
        w_c00 = 4 * m
        w_c10 = 4 * m + 1
        w_c01 = 4 * m + 2
        w_c11 = 4 * m + 3
        
        # M1: (a00+a11)(b00+b11)
        U[0][r0] += 1
        U[3][r0] += 1
        V[v_b00][r0] += 1
        V[v_b11][r0] += 1
        W[w_c00][r0] += 1
        W[w_c11][r0] += 1
        
        # M2: (a10+a11)b00
        U[2][r0+1] += 1
        U[3][r0+1] += 1
        V[v_b00][r0+1] += 1
        W[w_c10][r0+1] += 1
        
        # M3: a00(b01-b11)
        U[0][r0+2] += 1
        V[v_b01][r0+2] += 1
        V[v_b11][r0+2] += -1
        W[w_c01][r0+2] += 1
        
        # M4: a11(b10-b00)
        U[3][r0+3] += 1
        V[v_b10][r0+3] += 1
        V[v_b00][r0+3] += -1
        W[w_c00][r0+3] += -1
        W[w_c10][r0+3] += 1
        
        # M5: (a00+a01)b11
        U[0][r0+4] += 1
        U[1][r0+4] += 1
        V[v_b11][r0+4] += 1
        W[w_c01][r0+4] += 1
        W[w_c11][r0+4] += -1
        
        # M6: (a10-a00)(b00+b01)
        U[2][r0+5] += 1
        U[0][r0+5] += -1
        V[v_b00][r0+5] += 1
        V[v_b01][r0+5] += 1
        W[w_c00][r0+5] += -1
        W[w_c11][r0+5] += 1
        
        # M7: (a01-a11)(b10+b11)
        U[1][r0+6] += 1
        U[3][r0+6] += -1
        V[v_b10][r0+6] += 1
        V[v_b11][r0+6] += 1
        W[w_c00][r0+6] += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
