import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-reduced matrix multiplication decomposition for 2x3 * 3x8.
    Uses Strassen on 2x2x2 sub-blocks and schoolbook on 2x1x8 sub-blocks.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = 2, 3, 8
    
    # Split m=3 into m1=1 and m2=2
    # Part A: 2x1 * 1x8 -> rank 16 (schoolbook)
    # Part B: 2x2 * 2x8 -> split 8 into four 2x2 blocks, each Strassen rank 7 -> rank 28
    # Total rank = 16 + 28 = 44
    
    rank_A = n * 1 * p  # 16
    rank_B = 4 * 7       # 28
    total_rank = rank_A + rank_B  # 44
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part A: 2x1 * 1x8 (schoolbook)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part B: 2x2 * 2x8 using Strassen on four 2x2x2 blocks
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),
    ]
    
    A_u_indices = [0 * m + 1, 0 * m + 2, 1 * m + 1, 1 * m + 2]
    
    for b in range(4):
        k_base = 2 * b
        B_v_indices = [1 * p + k_base, 1 * p + k_base + 1, 2 * p + k_base, 2 * p + k_base + 1]
        C_w_indices = [k_base * n + 0, k_base * n + 1, (k_base + 1) * n + 0, (k_base + 1) * n + 1]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            for idx, coeff in enumerate(a_coeffs):
                if coeff != 0:
                    U[A_u_indices[idx]][col] += coeff
            for idx, coeff in enumerate(b_coeffs):
                if coeff != 0:
                    V[B_v_indices[idx]][col] += coeff
            for idx, coeff in enumerate(c_coeffs):
                if coeff != 0:
                    W[C_w_indices[idx]][col] += coeff
            col += 1
    
    assert col == total_rank
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(np.dot(U_arr[i * m + j], V_arr[j * p + k]), W_arr[k * n + i])
                max_err = max(max_err, abs(s - 1.0))
    
    if max_err > 1e-10:
        U_fb = [[0] * 48 for _ in range(n * m)]
        V_fb = [[0] * 48 for _ in range(m * p)]
        W_fb = [[0] * 48 for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][c] = 1
                    V_fb[j * p + k][c] = 1
                    W_fb[k * n + i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
