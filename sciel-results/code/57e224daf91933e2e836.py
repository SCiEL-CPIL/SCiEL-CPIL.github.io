import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    # 18 for 2x2x5 part (A[0:2, 0:2] * B[0:2, :])
    # 10 for 2x1x5 part (A[0:2, 2] * B[2, :])
    
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val):
        U[i * m + j][r] += val
    def set_v(j, k, r, val):
        V[j * p + k][r] += val
    def set_w(k, i, r, val):
        W[k * n + i][r] += val
        
    # Part 1: 2x2x5 multiplication
    # A1 is A[0:2, 0:2], B1 is B[0:2, 0:5]
    # We split B1 into B1[0:2] (2x2), B1[2:4] (2x2), B1[4:5] (2x1)
    
    # Strassen 1: for B cols 0,1
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    r = 0
    # M1
    set_u(0, 0, r, 1); set_u(1, 1, r, 1)
    set_v(0, 0, r, 1); set_v(1, 1, r, 1)
    # C00 += M1, C11 += M1
    set_w(0, 0, r, 1); set_w(1, 1, r, 1)
    r += 1
    
    # M2
    set_u(1, 0, r, 1); set_u(1, 1, r, 1)
    set_v(0, 0, r, 1)
    # C10 += M2
    set_w(0, 1, r, 1)
    r += 1
    
    # M3
    set_u(0, 0, r, 1)
    set_v(0, 1, r, 1); set_v(1, 1, r, -1)
    # C01 += M3
    set_w(1, 0, r, 1)
    r += 1
    
    # M4
    set_u(1, 1, r, 1)
    set_v(1, 0, r, 1); set_v(0, 0, r, -1)
    # C10 += M4, C11 -= M4
    set_w(0, 1, r, 1); set_w(1, 1, r, -1)
    r += 1
    
    # M5
    set_u(0, 0, r, 1); set_u(0, 1, r, 1)
    set_v(1, 1, r, 1)
    # C00 -= M5, C01 += M5
    set_w(0, 0, r, -1); set_w(1, 0, r, 1)
    r += 1
    
    # M6
    set_u(1, 0, r, 1); set_u(0, 0, r, -1)
    set_v(0, 0, r, 1); set_v(0, 1, r, 1)
    # C11 += M6
    set_w(1, 1, r, 1)
    r += 1
    
    # M7
    set_u(0, 1, r, 1); set_u(1, 1, r, -1)
    set_v(1, 0, r, 1); set_v(1, 1, r, 1)
    # C00 += M7, C11 -= M7
    set_w(0, 0, r, 1); set_w(1, 1, r, -1)
    r += 1
    
    # Strassen 2: for B cols 2,3
    # Same logic, shift B indices by 2
    for k_offset in [2, 3]:
        pass # We'll just repeat the pattern with k+2
        
    # Actually, let's just write a helper for Strassen block
    def strassen_block(u_rows, v_cols, w_rows, w_cols, r_start):
        # u_rows: list of (i, j) for A
        # v_cols: list of (j, k) for B
        # w_rows: list of (k, i) for C
        # This is getting complex to generalize. Let's just hardcode the next 7.
        pass

    # Strassen 2: B cols 2,3 (indices 2,3)
    # M1
    set_u(0, 0, r, 1); set_u(1, 1, r, 1)
    set_v(0, 2, r, 1); set_v(1, 3, r, 1) # Wait, B is 2x5. B[0,2], B[1,3]?
    # B[0:2, 2:4]. So B[0,2], B[0,3], B[1,2], B[1,3].
    # Strassen on B_sub = [[b02, b03], [b12, b13]]
    # M1 = (a00+a11)(b02+b13)
    set_v(0, 2, r, 1); set_v(1, 3, r, 1)
    set_w(2, 0, r, 1); set_w(3, 1, r, 1) # C[0,2] += M1, C[1,3] += M1
    r += 1
    
    # M2 = (a10+a11)b02
    set_u(1, 0, r, 1); set_u(1, 1, r, 1)
    set_v(0, 2, r, 1)
    set_w(2, 1, r, 1) # C[1,2] += M2
    r += 1
    
    # M3 = a00(b03-b13)
    set_u(0, 0, r, 1)
    set_v(0, 3, r, 1); set_v(1, 3, r, -1)
    set_w(3, 0, r, 1) # C[0,3] += M3
    r += 1
    
    # M4 = a11(b12-b02)
    set_u(1, 1, r, 1)
    set_v(1, 2, r, 1); set_v(0, 2, r, -1)
    set_w(2, 1, r, 1); set_w(3, 1, r, -1) # C[1,2] += M4, C[1,3] -= M4
    r += 1
    
    # M5 = (a00+a01)b13
    set_u(0, 0, r, 1); set_u(0, 1, r, 1)
    set_v(1, 3, r, 1)
    set_w(2, 0, r, -1); set_w(3, 0, r, 1) # C[0,2] -= M5, C[0,3] += M5
    r += 1
    
    # M6 = (a10-a00)(b02+b03)
    set_u(1, 0, r, 1); set_u(0, 0, r, -1)
    set_v(0, 2, r, 1); set_v(0, 3, r, 1)
    set_w(3, 1, r, 1) # C[1,3] += M6
    r += 1
    
    # M7 = (a01-a11)(b12+b13)
    set_u(0, 1, r, 1); set_u(1, 1, r, -1)
    set_v(1, 2, r, 1); set_v(1, 3, r, 1)
    set_w(2, 0, r, 1); set_w(3, 1, r, -1) # C[0,2] += M7, C[1,3] -= M7
    r += 1
    
    # Schoolbook for B col 4 (index 4)
    # C[0,4] = a00*b04 + a01*b14
    # C[1,4] = a10*b04 + a11*b14
    
    # a00*b04
    set_u(0, 0, r, 1)
    set_v(0, 4, r, 1)
    set_w(4, 0, r, 1)
    r += 1
    
    # a01*b14
    set_u(0, 1, r, 1)
    set_v(1, 4, r, 1)
    set_w(4, 0, r, 1)
    r += 1
    
    # a10*b04
    set_u(1, 0, r, 1)
    set_v(0, 4, r, 1)
    set_w(4, 1, r, 1)
    r += 1
    
    # a11*b14
    set_u(1, 1, r, 1)
    set_v(1, 4, r, 1)
    set_w(4, 1, r, 1)
    r += 1
    
    # Part 2: 2x1x5 multiplication
    # A col 2 (index 2), B row 2 (index 2)
    # C[0,k] = a02*b2k
    # C[1,k] = a12*b2k
    
    for k in range(5):
        # a02*b2k
        set_u(0, 2, r, 1)
        set_v(2, k, r, 1)
        set_w(k, 0, r, 1)
        r += 1
        
        # a12*b2k
        set_u(1, 2, r, 1)
        set_v(2, k, r, 1)
        set_w(k, 1, r, 1)
        r += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
