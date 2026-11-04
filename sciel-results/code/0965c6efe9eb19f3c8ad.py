import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Inner Strassen factors for 2x2x2
    # u, v, w are lists of 7 vectors of length 4
    # Indices: 0:11, 1:12, 2:21, 3:22
    u_in = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1],# M6
        [0, 1, 0, -1] # M7
    ]
    v_in = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    w_in = [
        [1, 0, 0, 1], # M1 -> c11, c22
        [0, 0, 1, 0], # M2 -> c21
        [0, 1, 0, 0], # M3 -> c12
        [1, 0, 0, 0], # M4 -> c11
        [0, 1, 0, 0], # M5 -> c12
        [0, 0, 0, 1], # M6 -> c22
        [1, 0, 0, 0]  # M7 -> c11
    ]
    
    # Outer Strassen coefficients
    # alpha[s][I][K] for X_s
    # beta[s][K][L] for Y_s
    # gamma[s][I][L] for C reconstruction
    
    # I, K, L are 1-based in logic, 0-based in code
    # I: 0,1 (rows of A)
    # K: 0,1 (cols of A / rows of B)
    # L: 0,1 (cols of B / cols of C)
    
    alpha = [
        [[1, 0], [0, 1]], # s=0 (M1): A11 + A22
        [[0, 0], [1, 1]], # s=1 (M2): A21 + A22
        [[1, 0], [0, 0]], # s=2 (M3): A11
        [[0, 0], [0, 1]], # s=3 (M4): A22
        [[1, 1], [0, 0]], # s=4 (M5): A11 + A12
        [[0, 0], [1, -1]],# s=5 (M6): A21 - A11 (Note: A11 is I=0,K=0, A21 is I=1,K=0)
        # Wait, M6 = (A21 - A11)(B11 + B12).
        # A21 is I=1, K=0. A11 is I=0, K=0.
        # So alpha[5][1][0] = 1, alpha[5][0][0] = -1.
        # Let's re-verify indices.
        # A[I][K]
        # M6: X6 = A21 - A11.
        # A21: I=1, K=0.
        # A11: I=0, K=0.
        # So alpha[5][1][0] = 1, alpha[5][0][0] = -1.
        # My list above: [[0,0],[1,-1]] means alpha[1][0]=0, alpha[1][1]=0? No.
        # alpha[s][I][K].
        # alpha[5] should be:
        # I=0: [ -1, 0 ]
        # I=1: [ 1, 0 ]
        # Let's correct the lists.
    ]
    
    # Correct Alpha (X_s)
    alpha = [
        [[1, 0], [0, 1]], # M1: A11 + A22
        [[0, 0], [1, 1]], # M2: A21 + A22
        [[1, 0], [0, 0]], # M3: A11
        [[0, 0], [0, 1]], # M4: A22
        [[1, 1], [0, 0]], # M5: A11 + A12
        [[-1, 0], [1, 0]],# M6: A21 - A11
        [[0, 1], [0, -1]] # M7: A12 - A22
    ]
    
    # Correct Beta (Y_s)
    beta = [
        [[1, 0], [0, 1]], # M1: B11 + B22
        [[1, 0], [0, 0]], # M2: B11
        [[0, 1], [0, -1]],# M3: B12 - B22
        [[0, -1], [1, 0]],# M4: B21 - B11
        [[0, 0], [0, 1]], # M5: B22
        [[1, 1], [0, 0]], # M6: B11 + B12
        [[0, 0], [1, 1]]  # M7: B21 + B22
    ]
    
    # Correct Gamma (C reconstruction)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    gamma = [
        [[1, 0], [0, 1]], # M1: C11 + C22
        [[0, 0], [1, 0]], # M2: C21
        [[0, 1], [0, 0]], # M3: C12
        [[1, 0], [0, 0]], # M4: C11
        [[0, 1], [0, 0]], # M5: C12 (with -1 in formula) -> Wait.
        # Formula: C11 = M1 + M4 - M5 + M7.
        # So gamma[4][0][0] should be -1.
        # Let's re-read Strassen.
        # C11 = M1 + M4 - M5 + M7.
        # C12 = M3 + M5.
        # C21 = M2 + M4.
        # C22 = M1 - M2 + M3 + M6.
        
        # gamma[s][I][L]
        # s=0 (M1): C11(+1), C22(+1)
        # s=1 (M2): C21(+1), C22(-1)
        # s=2 (M3): C12(+1), C22(+1)
        # s=3 (M4): C11(+1), C21(+1)
        # s=4 (M5): C11(-1), C12(+1)
        # s=5 (M6): C22(+1)
        # s=6 (M7): C11(+1)
        
    ]
    
    gamma = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, -1]],# M2
        [[0, 1], [0, 1]], # M3
        [[1, 0], [1, 0]], # M4
        [[-1, 1], [0, 0]],# M5
        [[0, 0], [0, 1]], # M6
        [[1, 0], [0, 0]]  # M7
    ]
    
    rank = 98
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for s in range(7):
        for r in range(7):
            for side in [0, 1]: # 0: Left, 1: Right
                
                # U factor
                for I in range(2):
                    for K in range(2):
                        a_val = alpha[s][I][K]
                        if a_val != 0:
                            # Base indices for A[I][K]
                            base_r = 2 * I
                            base_c = 2 * K
                            for lr in range(2):
                                for lc in range(2):
                                    local_idx = lr * 2 + lc
                                    val = a_val * u_in[r][local_idx]
                                    if val != 0:
                                        gr = base_r + lr
                                        gc = base_c + lc
                                        U[gr * 4 + gc][col] += val
                                        
                # V factor
                for K in range(2):
                    for L in range(2):
                        b_val = beta[s][K][L]
                        if b_val != 0:
                            # Base indices for B[K][L]
                            base_r = 2 * K
                            base_c = 4 * L
                            # Side 0: cols 0,1. Side 1: cols 2,3
                            col_offset = side * 2
                            for lr in range(2):
                                for lc in range(2):
                                    local_idx = lr * 2 + lc
                                    val = b_val * v_in[r][local_idx]
                                    if val != 0:
                                        gr = base_r + lr
                                        gc = base_c + col_offset + lc
                                        V[gr * 8 + gc][col] += val
                                        
                # W factor
                for I in range(2):
                    for L in range(2):
                        c_val = gamma[s][I][L]
                        if c_val != 0:
                            # Base indices for C[I][L]
                            base_r = 2 * I
                            base_c = 4 * L
                            col_offset = side * 2
                            for lr in range(2):
                                for lc in range(2):
                                    local_idx = lr * 2 + lc
                                    val = c_val * w_in[r][local_idx]
                                    if val != 0:
                                        gr = base_r + lr
                                        gc = base_c + col_offset + lc
                                        W[gr * 8 + gc][col] += val
                                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
