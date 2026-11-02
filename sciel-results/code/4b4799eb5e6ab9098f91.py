import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strategy: Use the known rank-7 Strassen decomposition for 2x2 matrix multiplication.
    # Since B is 2x12, we can treat the multiplication as 6 independent 2x2 block multiplications
    # by partitioning the 12 columns of B into 6 pairs of 2 columns.
    # Total rank = 6 * 7 = 42.
    #
    # The tensor decomposition requires:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # Strassen's 7 products for multiplying A (2x2) by B (2x2):
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    #
    # c00 = M1 - M5 + M3
    # c01 = M5 + M4
    # c10 = M2 + M7
    # c11 = M1 - M2 + M6
    #
    # Wait, let me double check the standard Strassen formulas.
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)  -> (a00+a11)(b00+b11)
    # M2 = (A21 + A22)B11          -> (a10+a11)b00
    # M3 = A11(B21 - B22)          -> a00(b10 - b11)  [Note: indices vary by convention]
    # Let's stick to the 0-indexed version carefully.
    #
    # Let's use the version from the current program which seems to be a standard implementation.
    # The current program uses:
    # s=0: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # s=1: M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
    # s=2: M3 = A00(B01-B11)       -> C01 += M3, C11 += M3
    # s=3: M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
    # s=4: M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
    # s=5: M6 = (A10-A00)(B00+B01) -> C11 += M6
    # s=6: M7 = (A01-A11)(B10+B11) -> C00 += M7
    #
    # Let's verify c00: M1 + M4 - M5 + M7
    # c00 = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    #     = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    #     = A00B00 + A01B10
    # This is correct for c00 = A00B00 + A01B10.
    #
    # Let's verify c01: M3 + M5
    # c01 = A00(B01-B11) + (A00+A01)B11
    #     = A00B01 - A00B11 + A00B11 + A01B11
    #     = A00B01 + A01B11
    # This is correct for c01 = A00B01 + A01B11.
    #
    # Let's verify c10: M2 + M4
    # c10 = (A10+A11)B00 + A11(B10-B00)
    #     = A10B00 + A11B00 + A11B10 - A11B00
    #     = A10B00 + A11B10
    # This is correct for c10 = A10B00 + A11B10.
    #
    # Let's verify c11: M1 - M2 + M6 + M3
    # c11 = (A00+A11)(B00+B11) - (A10+A11)B00 + (A10-A00)(B00+B01) + A00(B01-B11)
    #     = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A10B00 + A10B01 - A00B00 - A00B01 + A00B01 - A00B11
    #     = A11B11 + A10B01
    # This is correct for c11 = A10B01 + A11B11.
    
    # The decomposition is valid.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    num_blocks = p // 2  # 6 blocks
    
    for t in range(num_blocks):
        if time.time() > deadline:
            break
        base = t * 7
        
        # B indices for block t:
        # B[0, 2t] -> index 0*p + 2t = 2t
        # B[0, 2t+1] -> index 0*p + 2t+1 = 2t+1
        # B[1, 2t] -> index 1*p + 2t = p + 2t
        # B[1, 2t+1] -> index 1*p + 2t+1 = p + 2t+1
        
        v_b00 = 2*t
        v_b01 = 2*t + 1
        v_b10 = p + 2*t
        v_b11 = p + 2*t + 1
        
        # C indices for block t:
        # C[0, 2t] -> index 2t*n + 0 = 2t
        # C[0, 2t+1] -> index (2t+1)*n + 0 = 4t+1
        # C[1, 2t] -> index 2t*n + 1 = 2t+1
        # C[1, 2t+1] -> index (2t+1)*n + 1 = 4t+2
        
        w_c00 = 2*t
        w_c01 = 4*t + 1
        w_c10 = 2*t + 1
        w_c11 = 4*t + 2
        
        # A indices:
        # A[0,0] -> 0
        # A[0,1] -> 1
        # A[1,0] -> 2
        # A[1,1] -> 3
        
        u_a00 = 0
        u_a01 = 1
        u_a10 = 2
        u_a11 = 3
        
        # s=0: M1
        c = base + 0
        U[u_a00][c] = 1
        U[u_a11][c] = 1
        V[v_b00][c] = 1
        V[v_b11][c] = 1
        W[w_c00][c] = 1
        W[w_c11][c] = 1
        
        # s=1: M2
        c = base + 1
        U[u_a10][c] = 1
        U[u_a11][c] = 1
        V[v_b00][c] = 1
        W[w_c10][c] = 1
        W[w_c11][c] = -1
        
        # s=2: M3
        c = base + 2
        U[u_a00][c] = 1
        V[v_b01][c] = 1
        V[v_b11][c] = -1
        W[w_c01][c] = 1
        W[w_c11][c] = 1
        
        # s=3: M4
        c = base + 3
        U[u_a11][c] = 1
        V[v_b10][c] = 1
        V[v_b00][c] = -1
        W[w_c00][c] = 1
        W[w_c10][c] = 1
        
        # s=4: M5
        c = base + 4
        U[u_a00][c] = 1
        U[u_a01][c] = 1
        V[v_b11][c] = 1
        W[w_c00][c] = -1
        W[w_c01][c] = 1
        
        # s=5: M6
        c = base + 5
        U[u_a10][c] = 1
        U[u_a00][c] = -1
        V[v_b00][c] = 1
        V[v_b01][c] = 1
        W[w_c11][c] = 1
        
        # s=6: M7
        c = base + 6
        U[u_a01][c] = 1
        U[u_a11][c] = -1
        V[v_b10][c] = 1
        V[v_b11][c] = 1
        W[w_c00][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
