"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using Strassen for 2x2 blocks
    # 7 Strassen steps x 7 blocks of 2 columns
    R = 49
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # u[l][i][j], v[l][j][c], w[l][c][i]
    # l in 0..6
    u = [
        [[1, 0], [0, 0]],  # 1
        [[1, 1], [0, 0]],  # 2
        [[0, 0], [1, 1]],  # 3
        [[0, 0], [0, 1]],  # 4
        [[1, 0], [1, 0]],  # 5
        [[0, 1], [-1, 0]], # 6
        [[0, 1], [0, -1]]  # 7
    ]
    v = [
        [[1, 0], [0, 1]],  # 1
        [[0, 0], [1, 0]],  # 2
        [[0, 1], [0, 0]],  # 3
        [[1, 0], [0, -1]], # 4
        [[0, 0], [0, 1]],  # 5
        [[1, 1], [0, 0]],  # 6
        [[0, 1], [1, 0]]   # 7
    ]
    w = [
        [[1, 0], [0, 1]],  # c=0,i=0: 1; c=0,i=1: 0; c=1,i=0: 0; c=1,i=1: 1? No.
        # Let's re-verify w
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M4 + M6
        # w[l][c][i]
        # l=0 (M1): C00:1, C11:1
        # l=1 (M2): C10:1, C11:-1
        # l=2 (M3): C01:1
        # l=3 (M4): C00:-1, C10:1, C11:1
        # l=4 (M5): C00:1
        # l=5 (M6): C11:1
        # l=6 (M7): C00:-1, C01:1
    ]
    w = [
        [[1, 0], [0, 1]],  # l=0: c0i0=1, c0i1=0, c1i0=0, c1i1=1
        [[0, 0], [1, -1]], # l=1: c0i0=0, c0i1=0, c1i0=1, c1i1=-1
        [[0, 1], [0, 0]],  # l=2: c0i0=0, c0i1=1, c1i0=0, c1i1=0
        [[-1, 0], [1, 1]], # l=3: c0i0=-1, c0i1=0, c1i0=1, c1i1=1
        [[1, 0], [0, 0]],  # l=4: c0i0=1, c0i1=0, c1i0=0, c1i1=0
        [[0, 0], [0, 1]],  # l=5: c0i0=0, c0i1=0, c1i0=0, c1i1=1
        [[-1, 1], [0, 0]]  # l=6: c0i0=-1, c0i1=1, c1i0=0, c1i1=0
    ]
    
    for t in range(7):
        for l in range(7):
            r = l * 7 + t
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][r] = u[l][i][j]
            
            # Fill V and W
            for j in range(2):
                for c in range(2):
                    k = 2 * t + c
                    val_v = v[l][j][c]
                    V[j * 14 + k][r] = val_v
                    
                    for i in range(2):
                        val_w = w[l][c][i]
                        W[k * 2 + i][r] = val_w
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
