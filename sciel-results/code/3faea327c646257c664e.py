"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for A_L (2x2) * B_T (2x10)
    # B_T is split into 5 blocks of 2x2
    # A_L rows in U: 0, 1, 3, 4 (indices for A[0,0], A[0,1], A[1,0], A[1,1])
    # B_T rows in V: 0..9 (B[0,0..9]), 10..19 (B[1,0..9])
    # C rows in W: 0..19 (C[0,0..9], C[1,0..9])
    
    # Define Strassen coefficients
    # M1: (A11+A22)(B11+B22) -> C11, C22
    # M2: (A21+A22)B11 -> C21, -C22
    # M3: A11(B21-B22) -> C12, C22
    # M4: A22(B11-B12) -> C11, C21
    # M5: (A11+A12)B22 -> -C11, C12
    # M6: (A21-A11)(B11+B12) -> C22
    # M7: (A12-A22)(B21+B22) -> C11, -C22
    
    # U indices: A11=0, A12=1, A21=3, A22=4
    # V indices: B11=0, B12=1, B21=10, B22=11 (local to block)
    # W indices: C11=0, C12=1, C21=2, C22=3 (local to block)
    
    # Coefficients for U (A11, A12, A21, A22) for each M
    U_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # Coefficients for V (B11, B12, B21, B22) for each M
    V_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [1, -1, 0, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Coefficients for W (C11, C12, C21, C22) for each M
    W_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, -1] # M7
    ]
    
    for b in range(5):
        col_base = 7 * b
        # V row base for B11, B12 is 2*b, 2*b+1
        # V row base for B21, B22 is 10+2*b, 10+2*b+1
        # W row base for C11..C22 is 4*b .. 4*b+3
        
        for m_idx in range(7):
            col = col_base + m_idx
            
            # Update U
            U[0][col] += U_coeffs[m_idx][0]
            U[1][col] += U_coeffs[m_idx][1]
            U[3][col] += U_coeffs[m_idx][2]
            U[4][col] += U_coeffs[m_idx][3]
            
            # Update V
            V[2*b][col] += V_coeffs[m_idx][0]
            V[2*b+1][col] += V_coeffs[m_idx][1]
            V[10+2*b][col] += V_coeffs[m_idx][2]
            V[10+2*b+1][col] += V_coeffs[m_idx][3]
            
            # Update W
            W[4*b][col] += W_coeffs[m_idx][0]
            W[4*b+1][col] += W_coeffs[m_idx][1]
            W[4*b+2][col] += W_coeffs[m_idx][2]
            W[4*b+3][col] += W_coeffs[m_idx][3]
            
    # Outer product part for A_R (2x1) * B_B (1x10)
    # A_R is col 2 of A (rows 2, 5 in U)
    # B_B is row 2 of B (rows 20..29 in V)
    # C is all rows (0..19 in W)
    
    col_base = 35
    for i in range(2):
        for k in range(10):
            col = col_base + i * 10 + k
            U[i * 3 + 2][col] = 1
            V[20 + k][col] = 1
            W[k * 2 + i][col] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
