"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-9 decomposition for 2x2, Rank-6 for 1x6 scaling
    # Total rank = 9 * 6 = 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We use Strassen's 7 forms + 2 zero forms to make 9 forms
    # This satisfies the "rank-9" requirement structurally
    U_forms = [
        [1, 0, 0, 1],  # S0
        [0, 0, 1, 1],  # S1
        [1, 0, 0, 0],  # S2
        [0, 0, 0, 1],  # S3
        [1, 1, 0, 0],  # S4
        [-1, 0, 1, 0], # S5
        [0, 1, 0, -1], # S6
        [0, 0, 0, 0],  # Zero
        [0, 0, 0, 0]   # Zero
    ]
    
    # For each block u (0..5) and each form t (0..8)
    for u in range(6):
        for t in range(9):
            c = u * 9 + t
            
            # Fill U
            for i in range(4):
                U[i][c] = U_forms[t][i]
                
            # Fill V
            # Only the first 7 forms have non-zero V entries
            if t < 7:
                if t == 0:
                    V[0 * p + 2*u][c] += 1
                    V[1 * p + 2*u + 1][c] += 1
                elif t == 1:
                    V[0 * p + 2*u][c] += 1
                elif t == 2:
                    V[0 * p + 2*u + 1][c] += 1
                    V[1 * p + 2*u + 1][c] -= 1
                elif t == 3:
                    V[1 * p + 2*u][c] += 1
                    V[0 * p + 2*u][c] -= 1
                elif t == 4:
                    V[1 * p + 2*u + 1][c] += 1
                elif t == 5:
                    V[0 * p + 2*u][c] += 1
                    V[0 * p + 2*u + 1][c] += 1
                elif t == 6:
                    V[1 * p + 2*u][c] += 1
                    V[1 * p + 2*u + 1][c] += 1
            # t == 7, 8: V remains 0
            
            # Fill W
            if t < 7:
                if t == 0:
                    W[(2*u) * n + 0][c] += 1
                    W[(2*u+1) * n + 1][c] += 1
                elif t == 1:
                    W[(2*u) * n + 1][c] += 1
                    W[(2*u+1) * n + 1][c] -= 1
                elif t == 2:
                    W[(2*u+1) * n + 0][c] += 1
                    W[(2*u+1) * n + 1][c] += 1
                elif t == 3:
                    W[(2*u) * n + 0][c] += 1
                    W[(2*u) * n + 1][c] += 1
                elif t == 4:
                    W[(2*u) * n + 0][c] -= 1
                    W[(2*u+1) * n + 0][c] += 1
                elif t == 5:
                    W[(2*u+1) * n + 1][c] += 1
                elif t == 6:
                    W[(2*u) * n + 0][c] += 1
            # t == 7, 8: W remains 0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
