import time
import sys

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 solution: 5 blocks of 2x2x2, each using Strassen's algorithm (rank 7)
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplications for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[x, y], [z, w]]
    # M1 = (a+d)(x+w)
    # M2 = (c+d)x
    # M3 = a(y-w)
    # M4 = d(z-x)
    # M5 = (a+b)w
    # M6 = (c-a)(x+z)
    # M7 = (b-d)(y+w)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M7
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M6
    
    # Coefficients for A in each M
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    A_coeffs = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [0, 0, 1, -1],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    # Coefficients for B in each M (local 2x2)
    # B local indices: 00, 01, 10, 11
    B_coeffs = [
        [1, 1, 0, 0], # M1: x+w
        [1, 0, 0, 0], # M2: x
        [0, 1, 0, -1],# M3: y-w
        [0, 0, 1, -1],# M4: z-x
        [0, 0, 0, 1], # M5: w
        [1, 0, 1, 0], # M6: x+z
        [0, 1, 0, 1]  # M7: y+w
    ]
    
    # Coefficients for C in each M (local 2x2)
    # C local indices: 00, 01, 10, 11
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M7
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M6
    C_coeffs = [
        [1, 0, 0, 1],  # M1 contributes to C00, C11
        [0, 0, 1, -1], # M2 contributes to C10, C11
        [0, 0, 1, 1],  # M3 contributes to C10, C11
        [1, 0, 0, 0],  # M4 contributes to C00
        [-1, 1, 0, 0], # M5 contributes to C00, C01
        [0, 0, 0, 1],  # M6 contributes to C11
        [-1, 1, 0, 0]  # M7 contributes to C00, C01
    ]
    
    # Note: The C_coeffs above represent the contribution of M_s to C_ik.
    # We need to map this to the global W vector.
    
    col = 0
    for l in range(num_blocks):
        for s in range(7):
            # Update U
            for i in range(n):
                for j in range(m):
                    U[i*m + j][col] = A_coeffs[s][i*m + j]
            
            # Update V
            # B global indices: row j (0..1), col k (0..9)
            # Local B indices: row j (0..1), col k' (0..1)
            # Global col = 2*l + k'
            for j in range(m):
                for k_local in range(2):
                    k_global = 2 * l + k_local
                    V[j*p + k_global][col] = B_coeffs[s][j*2 + k_local]
            
            # Update W
            # C global indices: row i (0..1), col k (0..9)
            # W index: k*n + i
            # Local C indices: row i (0..1), col k' (0..1)
            # Global col = 2*l + k'
            for i in range(n):
                for k_local in range(2):
                    k_global = 2 * l + k_local
                    w_index = k_global * n + i
                    W[w_index][col] = C_coeffs[s][i*2 + k_local]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
