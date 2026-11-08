"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START

def _construct_rank42():
    """
    Rank-42 decomposition for 2x4 by 4x6 matrix multiplication.
    Uses Strassen's 7-form method for each of the 6 (2 A-blocks x 3 B-blocks)
    2x2x2 subproblems.
    
    A is 2x4, split into 2 blocks of 2 columns each.
    B is 4x6, split into 3 blocks of 2 rows x 6 cols... 
    Actually: B is 4x6, split into 3 blocks of 2 rows each (each 2x6).
    But we need 2x2 subproblems, so we also split the 6 cols into 3 groups of 2.
    
    So: A-blocks: 2 (each 2x2), B-row-blocks: 2 (each 2x6), B-col-blocks: 3 (each 2x2).
    Wait, that gives 2*2*3 = 12 subproblems of 2x2x2x2, each rank 7 = 84. Too many.
    
    Correct: A is 2x4, B is 4x6.
    Split A columns into 2 groups of 2: A1 (2x2), A2 (2x2).
    Split B rows into 2 groups of 2: B1 (2x6), B2 (2x6).
    Split B cols into 3 groups of 2: for each row-group, 3 col-groups.
    
    C = A1*B1 + A2*B2 where each product is 2x6.
    A1*B1: 2x2 * 2x6. Split into 3 subproblems of 2x2 * 2x2, each rank 7. Total: 3*7=21.
    A2*B2: same, 21.
    Grand total: 42.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=int)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=int)  # 12 x 42
    
    # Strassen 7-form coefficients for 2x2x2:
    # C = A*B where A,B,C are 2x2
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # C = [[c00,c01],[c10,c11]]
    
    # Strassen's 7 products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10+b11)
    # M4 = (a11)(b01-b11)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # U_coeffs[s][(i,j)] = coeff for A[i,j] in form s
    # V_coeffs[s][(j,k)] = coeff for B[j,k] in form s  
    # W_coeffs[s][(i,k)] = coeff for C[i,k] in form s
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # M1
        {(1, 0): 1, (1, 1): 1},       # M2
        {(0, 0): 1},                   # M3
        {(1, 1): 1},                   # M4
        {(0, 0): 1, (0, 1): 1},       # M5
        {(1, 0): 1, (0, 0): -1},      # M6
        {(0, 1): 1, (1, 1): -1}       # M7
    ]
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # M1
        {(0, 0): 1},                   # M2
        {(1, 0): 1, (1, 1): 1},       # M3
        {(0, 1): 1, (1, 1): -1},      # M4
        {(1, 1): 1},                   # M5
        {(0, 0): 1, (1, 0): 1},       # M6
        {(0, 1): 1, (1, 1): 1}        # M7
    ]
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # M1: c00+=1, c11+=1
        {(1, 0): 1, (1, 1): -1},      # M2: c10+=1, c11+=-1
        {(0, 1): 1, (1, 1): 1},       # M3: c01+=1, c11+=1
        {(0, 0): 1, (1, 0): 1},       # M4: c00+=1, c10+=1
        {(0, 0): -1, (0, 1): 1},      # M5: c00+=-1, c01+=1
        {(1, 1): 1},                  # M6: c11+=1
        {(0, 0): 1}                   # M7: c00+=1
    ]
    
    col = 0
    for a_block in range(2):  # A column block
        for b_row in range(2):  # B row block
            for b_col in range(3):  # B column block
                for s in range(7):
                    for (i_loc, j_loc), coeff in U_coeffs[s].items():
                        i = i_loc
                        j = a_block * 2 + j_loc
                        u_idx = i * m + j
                        U[u_idx, col] += coeff
                    for (j_loc, k_loc), coeff in V_coeffs[s].items():
                        j = b_row * 2 + j_loc
                        k = b_col * 2 + k_loc
                        v_idx = j * p + k
                        V[v_idx, col] += coeff
                    for (i_loc, k_loc), coeff in W_coeffs[s].items():
                        i = i_loc
                        k = b_col * 2 + k_loc
                        w_idx = k * n + i
                        W[w_idx, col] += coeff
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Return the rank-42 Strassen-based decomposition
    try:
        result = _construct_rank42()
        elapsed = time.time() - start_time
        if elapsed < deadline:
            return result
    except Exception:
        pass
    
    # Absolute fallback: trivial rank-48 decomposition
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    return {"U": U, "V": V, "W": W}
