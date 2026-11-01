"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

import time

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based rank-35 decomposition for 2x2 x 2x10 multiplication.
    # The 2x10 matrix B is treated as 5 independent 2x2 blocks.
    # Each 2x2 x 2x2 block uses Strassen's 7-term algorithm.
    # Total rank = 5 * 7 = 35.
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    num_blocks = p // 2   # 5 blocks of width 2
    
    # Strassen factors for 2x2 matrix multiplication.
    # Each factor r has:
    #   U coefficients: (u00, u01, u10, u11) for A[i,j]
    #   V coefficients: (v00, v01, v10, v11) for B[j,k] within a block
    #   W coefficients: (w00, w01, w10, w11) for C[i,k] within a block
    #
    # The 7 Strassen products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B01)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C = [[M1-M2+M3+M6, M4-M5+M7],
    #      [M5+M6,       M1-M2+M3+M7]]
    #
    # Let me re-verify the Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00+B01)
    # M3 = (A00+A01)(B10+B11)
    # M4 = A00(B11-B01)
    # M5 = A11(B00-B10)
    # M6 = (A01-A11)(B00+B10)
    # M7 = (A10-A00)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 - M1 - M4 + M6
    # C10 = M2 - M3 - M5 - M7
    # C11 = M1 - M2 + M4 + M5
    
    # Let me use the standard Strassen decomposition:
    # p1 = (A00+A11)*(B00+B11)
    # p2 = (A10+A11)*B00
    # p3 = A00*(B11-B01)
    # p4 = A11*(B10-B00)
    # p5 = (A00+A01)*B11
    # p6 = (A10-A00)*(B00+B01)
    # p7 = (A01-A11)*(B10+B11)
    #
    # C00 = p1 - p2 + p3 + p6
    # C01 = p4 - p5 + p7
    # C10 = p5 + p6
    # C11 = p1 - p2 + p3 + p7
    
    # Define the 7 factors.
    # For each factor, we specify (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs = [u00, u01, u10, u11] where U row = i*2+j
    # V_coeffs = [v00, v01, v10, v11] where V row = j*2+k (within block)
    # W_coeffs = [w00, w01, w10, w11] where W row = k*2+i (within block)
    
    factors = [
        # p1 = (A00+A11)*(B00+B11)
        # U: A00=1, A11=1 -> [1,0,0,1]
        # V: B00=1, B11=1 -> [1,0,0,1]
        # W: C00=1, C11=1 -> [1,0,0,1]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        
        # p2 = (A10+A11)*B00
        # U: A10=1, A11=1 -> [0,0,1,1]
        # V: B00=1 -> [1,0,0,0]
        # W: C10=1, C11=-1 -> [0,-1,1,0]
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, -1, 1, 0]),
        
        # p3 = A00*(B11-B01)
        # U: A00=1 -> [1,0,0,0]
        # V: B11=1, B01=-1 -> [0,-1,0,1]
        # W: C00=1, C11=1 -> [1,0,0,1]
        ([1, 0, 0, 0], [0, -1, 0, 1], [1, 0, 0, 1]),
        
        # p4 = A11*(B10-B00)
        # U: A11=1 -> [0,0,0,1]
        # V: B10=1, B00=-1 -> [-1,0,1,0]
        # W: C01=1 -> [0,1,0,0]
        ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 1, 0, 0]),
        
        # p5 = (A00+A01)*B11
        # U: A00=1, A01=1 -> [1,1,0,0]
        # V: B11=1 -> [0,0,0,1]
        # W: C01=-1, C10=1 -> [0,-1,1,0]
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, -1, 1, 0]),
        
        # p6 = (A10-A00)*(B00+B01)
        # U: A10=1, A00=-1 -> [-1,0,1,0]
        # V: B00=1, B01=1 -> [1,1,0,0]
        # W: C00=1, C10=1 -> [1,0,1,0]
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0]),
        
        # p7 = (A01-A11)*(B10+B11)
        # U: A01=1, A11=-1 -> [0,1,0,-1]
        # V: B10=1, B11=1 -> [0,0,1,1]
        # W: C01=1, C11=1 -> [0,1,0,1]
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]),
    ]
    
    rank = num_blocks * 7  # 35
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    for b in range(num_blocks):
        for r_idx, (u_coeffs, v_coeffs, w_coeffs) in enumerate(factors):
            col = b * 7 + r_idx
            
            # Set U values: row = i*2+j
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = u_coeffs[i * 2 + j]
            
            # Set V values: row = j*10 + (2*b + k)
            for j in range(m):
                for k in range(2):
                    V[j * p + (2 * b + k)][col] = v_coeffs[j * 2 + k]
            
            # Set W values: row = (2*b + k)*2 + i
            for k in range(2):
                for i in range(n):
                    W[(2 * b + k) * n + i][col] = w_coeffs[k * 2 + i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

