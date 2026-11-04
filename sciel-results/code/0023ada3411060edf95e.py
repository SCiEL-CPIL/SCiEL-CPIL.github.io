"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 5
    R = 65
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Strassen 2x2 terms
    strassen_2x2 = [
        # M1
        (np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])),
        # M2
        (np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]), np.array([[0,0],[1,0]])),
        # M3
        (np.array([[1,0],[0,0]]), np.array([[0,1],[0,-1]]), np.array([[0,1],[0,0]])),
        # M4
        (np.array([[0,0],[0,1]]), np.array([[-1,0],[0,1]]), np.array([[1,0],[1,0]])),
        # M5
        (np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]]), np.array([[-1,0],[0,-1]])),
        # M6
        (np.array([[-1,0],[1,0]]), np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]])),
        # M7
        (np.array([[0,-1],[1,0]]), np.array([[0,0],[1,1]]), np.array([[1,0],[0,-1]]))
    ]
    
    # Outer Strassen definitions
    # X, Y, Z are 2x2 arrays of (block_idx, coeff)
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    outer_strassen = [
        # M1
        (
            [[(0,1), (1,0)], [(0,0), (1,1)]],
            [[(0,1), (1,0)], [(0,0), (1,1)]],
            [[(0,1), (1,0)], [(2,0), (3,1)]]
        ),
        # M2
        (
            [[(0,0), (1,0)], [(2,1), (3,1)]],
            [[(0,1), (1,0)], [(0,0), (0,0)]],
            [[(0,0), (1,0)], [(2,1), (3,0)]]
        ),
        # M3
        (
            [[(0,1), (1,0)], [(0,0), (0,0)]],
            [[(0,0), (1,1)], [(0,0), (3,-1)]],
            [[(0,0), (1,1)], [(2,0), (3,0)]]
        ),
        # M4
        (
            [[(0,0), (1,0)], [(0,0), (3,1)]],
            [[(0,-1), (1,0)], [(2,1), (3,0)]],
            [[(0,1), (1,0)], [(2,1), (3,0)]]
        ),
        # M5
        (
            [[(0,1), (1,1)], [(0,0), (0,0)]],
            [[(0,0), (1,0)], [(0,0), (3,1)]],
            [[(0,-1), (1,-1)], [(2,0), (3,0)]]
        ),
        # M6
        (
            [[(0,-1), (1,0)], [(2,1), (3,0)]],
            [[(0,1), (1,1)], [(0,0), (0,0)]],
            [[(0,0), (1,0)], [(2,0), (3,1)]]
        ),
        # M7
        (
            [[(0,0), (1,1)], [(0,0), (3,-1)]],
            [[(0,0), (1,0)], [(2,1), (3,1)]],
            [[(0,1), (1,0)], [(2,0), (3,1)]]
        )
    ]
    
    col = 0
    for k in range(7):
        X, Y, Z = outer_strassen[k]
        for l in range(7):
            U_l, V_l, W_l = strassen_2x2[l]
            for i in range(2):
                for j in range(2):
                    # U part
                    block_idx, coeff = X[i][j]
                    if coeff != 0:
                        row_start = (block_idx // 2) * 2
                        col_start = (block_idx % 2) * 2
                        for ii in range(2):
                            for jj in range(2):
                                global_i = row_start + ii
                                global_j = col_start + jj
                                U[global_i * 4 + global_j, col] += coeff * U_l[i][j]
                    
                    # V part
                    block_idx, coeff = Y[i][j]
                    if coeff != 0:
                        row_start = (block_idx // 2) * 2
                        col_start = (block_idx % 2) * 2
                        for ii in range(2):
                            for jj in range(2):
                                global_i = row_start + ii
                                global_j = col_start + jj
                                V[global_i * 5 + global_j, col] += coeff * V_l[i][j]
                    
                    # W part
                    for r in range(2):
                        for s in range(2):
                            block_idx, coeff = Z[r][s]
                            if coeff != 0:
                                row_start = r * 2
                                col_start = s * 2
                                for ii in range(2):
                                    for jj in range(2):
                                        global_i = row_start + ii
                                        global_j = col_start + jj
                                        W[global_j * 4 + global_i, col] += coeff * W_l[r][s]
            col += 1
            
    # Part 2: 4x4x1 (Schoolbook)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
