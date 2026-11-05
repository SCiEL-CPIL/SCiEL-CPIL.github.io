"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen for 2x2 blocks of 3x3 matrices.
    # Rank = 7 * 27 = 189.
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 blocks
    # alpha[r][I][J], beta[r][I][J], gamma[r][K][L]
    # r is 0..6
    # I, J, K, L are 0..1
    
    alpha = [
        [[1, 0], [0, 1]], # r=0: A00 + A11
        [[0, 0], [1, 1]], # r=1: A10 + A11
        [[1, 0], [0, 0]], # r=2: A00
        [[0, 0], [0, 1]], # r=3: A11
        [[1, 1], [0, 0]], # r=4: A00 + A01
        [[-1, 0], [1, 0]],# r=5: A10 - A00
        [[0, 1], [0, -1]] # r=6: A01 - A11
    ]
    
    beta = [
        [[1, 0], [0, 1]], # r=0: B00 + B11
        [[1, 0], [0, 0]], # r=1: B00
        [[0, 1], [0, -1]],# r=2: B01 - B11
        [[-1, 0], [1, 0]],# r=3: B10 - B00
        [[0, 0], [0, 1]], # r=4: B11
        [[1, 1], [0, 0]], # r=5: B00 + B01
        [[0, 0], [1, 1]]  # r=6: B10 + B11
    ]
    
    gamma = [
        [[1, -1], [-1, 1]], # r=0: C00+=, C01-=-, C10-=-, C11+=
        [[0, 0], [1, 0]],   # r=1: C10+=
        [[0, 1], [0, -1]],  # r=2: C01+=, C11-=-
        [[-1, 0], [1, 0]],  # r=3: C00-=-, C10+=
        [[0, 1], [0, 1]],   # r=4: C01+=, C11+=
        [[-1, 0], [0, 0]],  # r=5: C00-=-
        [[1, 0], [0, -1]]   # r=6: C00+=, C11-=-
    ]
    
    for r in range(7):
        for i3 in range(3):
            for j3 in range(3):
                for k3 in range(3):
                    idx = r * 27 + i3 * 9 + j3 * 3 + k3
                    
                    # Update U
                    # A element (3I+i3, 3J+j3)
                    for I in range(2):
                        for J in range(2):
                            a_coeff = alpha[r][I][J]
                            if a_coeff != 0:
                                row = 3 * I + i3
                                col = 3 * J + j3
                                a_idx = row * 6 + col
                                U[a_idx][idx] += a_coeff
                                
                    # Update V
                    # B element (3I+j3, 3J+k3)
                    for I in range(2):
                        for J in range(2):
                            b_coeff = beta[r][I][J]
                            if b_coeff != 0:
                                row = 3 * I + j3
                                col = 3 * J + k3
                                b_idx = row * 6 + col
                                V[b_idx][idx] += b_coeff
                                
                    # Update W
                    # C element (3K+k3, 3L+i3)
                    for K in range(2):
                        for L in range(2):
                            c_coeff = gamma[r][K][L]
                            if c_coeff != 0:
                                row = 3 * K + k3
                                col = 3 * L + i3
                                c_idx = row * 6 + col
                                W[c_idx][idx] += c_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
