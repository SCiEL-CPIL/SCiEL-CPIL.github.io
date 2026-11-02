"""Optimized program for 2x2x12 matrix multiplication using Strassen blocks."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank = 7 * (p // 2) = 7 * 6 = 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 linear forms for A (2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # S0: a00 + a11
    # S1: a10 + a11
    # S2: a00
    # S3: a11
    # S4: a00 + a01
    # S5: a10 - a00
    # S6: a01 - a11
    
    U_forms = [
        [1, 0, 0, 1],  # S0
        [0, 0, 1, 1],  # S1
        [1, 0, 0, 0],  # S2
        [0, 0, 0, 1],  # S3
        [1, 1, 0, 0],  # S4
        [-1, 0, 1, 0], # S5
        [0, 1, 0, -1]  # S6
    ]
    
    # For each block u (0..5) and each form t (0..6)
    for u in range(6):
        for t in range(7):
            c = u * 7 + t
            
            # Fill U
            for i in range(4):
                U[i][c] = U_forms[t][i]
                
            # Fill V
            # T0: b00 + b11 -> V[0][2u] += 1, V[1][2u+1] += 1
            # T1: b00 -> V[0][2u] += 1
            # T2: b01 - b11 -> V[0][2u+1] += 1, V[1][2u+1] -= 1
            # T3: b10 - b00 -> V[1][2u] += 1, V[0][2u] -= 1
            # T4: b11 -> V[1][2u+1] += 1
            # T5: b00 + b01 -> V[0][2u] += 1, V[0][2u+1] += 1
            # T6: b10 + b11 -> V[1][2u] += 1, V[1][2u+1] += 1
            
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
                
            # Fill W
            # Output indices: W[k*n + i]
            # C_00 (i=0, k=2u): M0 + M3 - M4 + M6
            # C_01 (i=0, k=2u+1): M2 + M4
            # C_10 (i=1, k=2u): M1 + M3
            # C_11 (i=1, k=2u+1): M0 - M1 + M2 + M5
            
            # M0 (t=0)
            if t == 0:
                W[(2*u) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            # M1 (t=1)
            elif t == 1:
                W[(2*u) * n + 1][c] += 1
                W[(2*u+1) * n + 1][c] -= 1
            # M2 (t=2)
            elif t == 2:
                W[(2*u+1) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            # M3 (t=3)
            elif t == 3:
                W[(2*u) * n + 0][c] += 1
                W[(2*u) * n + 1][c] += 1
            # M4 (t=4)
            elif t == 4:
                W[(2*u) * n + 0][c] -= 1
                W[(2*u+1) * n + 0][c] += 1
            # M5 (t=5)
            elif t == 5:
                W[(2*u+1) * n + 1][c] += 1
            # M6 (t=6)
            elif t == 6:
                W[(2*u) * n + 0][c] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
