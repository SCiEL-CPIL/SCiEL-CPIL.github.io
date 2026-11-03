"""Tensor decomposition for 2x4 * 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen on 2x2 blocks
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (x11+x22)(y11+y22)
    # M2 = (x21+x22)y11
    # M3 = x11(y12-y22)
    # M4 = x22(y21-y11)
    # M5 = (x11+x12)y22
    # M6 = (x21-x11)(y11+y12)
    # M7 = (x12-x22)(y21+y22)
    
    # U coefficients for X (2x2)
    # Indices in U: i*m + j. For X=A1, i in {0,1}, j in {0,1}.
    # For X=A2, i in {0,1}, j in {2,3}.
    # V coefficients for Y (2x2)
    # Indices in V: j*p + k.
    # For Y=B1b, j in {0,1}, k in {2b, 2b+1}.
    # For Y=B2b, j in {2,3}, k in {2b, 2b+1}.
    
    # W coefficients for Z (2x2)
    # Indices in W: k*n + i.
    # For Z block b, k in {2b, 2b+1}, i in {0,1}.
    
    # Strassen recovery coefficients c[i'][k'][s]
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # s=0: M1, s=1: M2, s=2: M3, s=3: M4, s=4: M5, s=5: M6, s=6: M7
    # Z[0][0] (i'=0, k'=0): [1, 0, 0, 1, -1, 0, 1]
    # Z[0][1] (i'=0, k'=1): [0, 0, 1, 0, 1, 0, 0]
    # Z[1][0] (i'=1, k'=0): [0, 1, 0, 1, 0, 0, 0]
    # Z[1][1] (i'=1, k'=1): [1, -1, 1, 0, 0, 1, 0]
    
    recovery = [
        [1, 0, 0, 1, -1, 0, 1], # i'=0, k'=0
        [0, 0, 1, 0, 1, 0, 0],  # i'=0, k'=1
        [0, 1, 0, 1, 0, 0, 0],  # i'=1, k'=0
        [1, -1, 1, 0, 0, 1, 0]  # i'=1, k'=1
    ]
    
    # U linear forms for X
    # s=0: x11+x22 -> [1, 0, 0, 1] (x11, x12, x21, x22)
    # s=1: x21+x22 -> [0, 0, 1, 1]
    # s=2: x11     -> [1, 0, 0, 0]
    # s=3: x22     -> [0, 0, 0, 1]
    # s=4: x11+x12 -> [1, 1, 0, 0]
    # s=5: x21-x11 -> [-1, 0, 1, 0]
    # s=6: x12-x22 -> [0, 1, 0, -1]
    
    U_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # V linear forms for Y
    # s=0: y11+y22 -> [1, 0, 0, 1] (y11, y12, y21, y22)
    # s=1: y11     -> [1, 0, 0, 0]
    # s=2: y12-y22 -> [0, 1, 0, -1]
    # s=3: y21-y11 -> [-1, 0, 1, 0]
    # s=4: y22     -> [0, 0, 0, 1]
    # s=5: y11+y12 -> [1, 1, 0, 0]
    # s=6: y21+y22 -> [0, 0, 1, 1]
    
    V_forms = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Loop over 3 blocks
    for b in range(3):
        # Block 1: A1 (cols 0,1) * B1b (rows 0,1)
        # A1 indices in U: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
        # B1b indices in V: (0,2b), (0,2b+1), (1,2b), (1,2b+1) -> 2b, 2b+1, 6+2b, 6+2b+1
        # Output indices in W: (2b,0), (2b,1), (2b+1,0), (2b+1,1) -> 2b*2+0, 2b*2+1, (2b+1)*2+0, (2b+1)*2+1
        
        for s in range(7):
            r = b * 14 + s
            # U
            u_coeffs = U_forms[s]
            # Map (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
            U[0][r] += u_coeffs[0]
            U[1][r] += u_coeffs[1]
            U[4][r] += u_coeffs[2]
            U[5][r] += u_coeffs[3]
            
            # V
            v_coeffs = V_forms[s]
            # Map (0,2b)->2b, (0,2b+1)->2b+1, (1,2b)->6+2b, (1,2b+1)->6+2b+1
            V[2*b][r] += v_coeffs[0]
            V[2*b+1][r] += v_coeffs[1]
            V[6+2*b][r] += v_coeffs[2]
            V[6+2*b+1][r] += v_coeffs[3]
            
            # W
            # Recovery coefficients for Z[i'][k']
            # Z[0][0] -> W[(2b)*2+0]
            # Z[0][1] -> W[(2b)*2+1]
            # Z[1][0] -> W[(2b+1)*2+0]
            # Z[1][1] -> W[(2b+1)*2+1]
            W[2*b*2+0][r] += recovery[0][s]
            W[2*b*2+1][r] += recovery[1][s]
            W[(2*b+1)*2+0][r] += recovery[2][s]
            W[(2*b+1)*2+1][r] += recovery[3][s]
            
        # Block 2: A2 (cols 2,3) * B2b (rows 2,3)
        # A2 indices in U: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
        # B2b indices in V: (2,2b), (2,2b+1), (3,2b), (3,2b+1) -> 12+2b, 12+2b+1, 18+2b, 18+2b+1
        
        for s in range(7):
            r = b * 14 + 7 + s
            # U
            u_coeffs = U_forms[s]
            U[2][r] += u_coeffs[0]
            U[3][r] += u_coeffs[1]
            U[6][r] += u_coeffs[2]
            U[7][r] += u_coeffs[3]
            
            # V
            v_coeffs = V_forms[s]
            V[12+2*b][r] += v_coeffs[0]
            V[12+2*b+1][r] += v_coeffs[1]
            V[18+2*b][r] += v_coeffs[2]
            V[18+2*b+1][r] += v_coeffs[3]
            
            # W
            # Same output indices as Block 1
            W[2*b*2+0][r] += recovery[0][s]
            W[2*b*2+1][r] += recovery[1][s]
            W[(2*b+1)*2+0][r] += recovery[2][s]
            W[(2*b+1)*2+1][r] += recovery[3][s]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
