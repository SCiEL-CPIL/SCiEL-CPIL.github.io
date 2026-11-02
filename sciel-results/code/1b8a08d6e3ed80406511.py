"""Tensor decomposition for 2x2 by 2x4 matrix multiplication."""

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14

    # Strassen forms for A (same for both blocks)
    # U indices: 0:a, 1:b, 2:c, 3:d
    U_forms = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [-1, 0, 1, 0], # M6: c-a
        [0, 1, 0, -1]  # M7: b-d
    ]

    # Strassen forms for B Block 1 (k=0,1)
    # V indices: 0:e, 1:f, 2:g, 3:h
    V_forms_block1 = [
        [1, 0, 1, 0],  # M1: e+g
        [1, 0, 1, 0],  # M2: e+g
        [0, 1, -1, 0], # M3: f-g
        [-1, 0, 0, 1], # M4: h-e
        [0, 0, 1, 1],  # M5: g+h
        [1, 1, 0, 0],  # M6: e+f
        [0, 1, 0, 1]   # M7: h+f
    ]

    # Strassen forms for B Block 2 (k=2,3)
    # V indices: 4:e', 5:f', 6:g', 7:h'
    V_forms_block2 = [
        [0, 0, 0, 0, 1, 0, 1, 0],  # M1': e'+g'
        [0, 0, 0, 0, 1, 0, 1, 0],  # M2': e'+g'
        [0, 0, 0, 0, 0, 1, -1, 0], # M3': f'-g'
        [0, 0, 0, 0, -1, 0, 0, 1], # M4': h'-e'
        [0, 0, 0, 0, 0, 0, 1, 1],  # M5': g'+h'
        [0, 0, 0, 0, 1, 1, 0, 0],  # M6': e'+f'
        [0, 0, 0, 0, 0, 1, 0, 1]   # M7': h'+f'
    ]

    # W contributions
    # W indices: 0:C00, 1:C10, 2:C01, 3:C11, 4:C02, 5:C12, 6:C03, 7:C13
    # For Block 1 (output indices 0-3):
    # M1: C00+=1, C11+=-1
    # M2: C00+=-1, C10+=1
    # M3: C01+=1
    # M4: C00+=1, C01+=1
    # M5: C01+=1, C11+=-1
    # M6: C10+=1
    # M7: C11+=1
    
    W_block1 = [
        [1, 0, 0, -1, 0, 0, 0, 0], # M1
        [-1, 1, 0, 0, 0, 0, 0, 0], # M2
        [0, 0, 1, 0, 0, 0, 0, 0],  # M3
        [1, 0, 1, 0, 0, 0, 0, 0],  # M4
        [0, 0, 1, -1, 0, 0, 0, 0], # M5
        [0, 1, 0, 0, 0, 0, 0, 0],  # M6
        [0, 0, 0, 1, 0, 0, 0, 0]   # M7
    ]

    # For Block 2 (output indices 4-7):
    # Shifted by 4
    W_block2 = [
        [0, 0, 0, 0, 1, 0, 0, -1], # M1'
        [0, 0, 0, 0, -1, 1, 0, 0], # M2'
        [0, 0, 0, 0, 0, 0, 1, 0],  # M3'
        [0, 0, 0, 0, 1, 0, 1, 0],  # M4'
        [0, 0, 0, 0, 0, 0, 1, -1], # M5'
        [0, 0, 0, 0, 0, 1, 0, 0],  # M6'
        [0, 0, 0, 0, 0, 0, 0, 1]   # M7'
    ]

    # Fill U, V, W
    for r in range(7):
        # Block 1
        for i in range(4):
            U[i][r] = U_forms[r][i]
        for j in range(8):
            V[j][r] = V_forms_block1[r][j]
        for k in range(8):
            W[k][r] = W_block1[r][k]
            
        # Block 2
        for i in range(4):
            U[i][r + 7] = U_forms[r][i]
        for j in range(8):
            V[j][r + 7] = V_forms_block2[r][j]
        for k in range(8):
            W[k][r + 7] = W_block2[r][k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
