import time
import sys

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Coefficients for the 7 Strassen terms M_s
    # cA[s] = [c0, c1, c2, c3] for blocks A11, A12, A21, A22
    # cB[s] = [c0, c1, c2, c3] for blocks B11, B12, B21, B22
    cA = [
        [1, 0, 0, 1],  # M0: A11 + A22
        [0, 0, 1, 1],  # M1: A21 + A22
        [1, 0, 0, 0],  # M2: A11
        [0, 0, 0, 1],  # M3: A22
        [1, 1, 0, 0],  # M4: A11 + A12
        [-1, 0, 1, 0], # M5: A21 - A11
        [0, 1, 0, -1]  # M6: A12 - A22
    ]
    cB = [
        [1, 0, 0, 1],  # M0: B11 + B22
        [1, 0, 0, 0],  # M1: B11
        [0, 1, 0, -1], # M2: B12 - B22
        [-1, 0, 1, 0], # M3: B21 - B11
        [0, 0, 0, 1],  # M4: B22
        [1, 1, 0, 0],  # M5: B11 + B12
        [0, 0, 1, 1]   # M6: B21 + B22
    ]

    # Local Strassen coefficients for U and V for each of the 7 sub-terms t
    # u_local[t] = [u00, u01, u10, u11]
    # v_local[t] = [v00, v01, v10, v11]
    u_local = [
        [1, 0, 0, 1],  # t0: X11 + X22
        [0, 0, 1, 1],  # t1: X21 + X22
        [1, 0, 0, 0],  # t2: X11
        [0, 0, 0, 1],  # t3: X22
        [1, 1, 0, 0],  # t4: X11 + X12
        [-1, 0, 1, 0], # t5: X21 - X11
        [0, 1, 0, -1]  # t6: X12 - X22
    ]
    v_local = [
        [1, 0, 0, 1],  # t0: Y11 + Y22
        [1, 0, 0, 0],  # t1: Y11
        [0, 1, 0, -1], # t2: Y12 - Y22
        [-1, 0, 1, 0], # t3: Y21 - Y11
        [0, 0, 0, 1],  # t4: Y22
        [1, 1, 0, 0],  # t5: Y11 + Y12
        [0, 0, 1, 1]   # t6: Y21 + Y22
    ]

    # Coefficients for W
    # K[s][b] where b is block index 0..3 (C11, C12, C21, C22)
    K = [
        [1, 0, 0, 1],  # M0
        [0, 0, 1, -1], # M1
        [0, 1, 0, 1],  # M2
        [1, 0, 1, 0],  # M3
        [-1, 1, 0, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 0, 0, 0]   # M6
    ]
    # L[t][l] where l is local index 0..3 (C11, C12, C21, C22)
    # Note: L is the same as K in this structure
    L = K

    # Helper to get block and local index from global index
    # Global index g = i * 4 + j
    # Block index b:
    # if i<2 and j<2: b=0 (C11)
    # if i<2 and j>=2: b=1 (C12)
    # if i>=2 and j<2: b=2 (C21)
    # if i>=2 and j>=2: b=3 (C22)
    # Local index l:
    # li = i % 2, lj = j % 2
    # l = li * 2 + lj

    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            
            # Populate U and V
            for i in range(4):
                for j in range(4):
                    idx_U = i * 4 + j
                    idx_V = j * 4 + i # V is m*p, so j is row, k is col? 
                    # Task says V is (m*p, rank). m=4, p=4.
                    # V[j*p+k, r]. Here j is 0..3, k is 0..3.
                    # In our notation, B is m*p.
                    # So V row index is j * 4 + k.
                    # But wait, the loop variable for V row is j (from A) and k (from B)?
                    # No, V corresponds to B.
                    # U corresponds to A. U[i*m+j, r]. i from A, j from A.
                    # V corresponds to B. V[j*p+k, r]. j from B, k from B.
                    # Wait, the standard tensor decomposition T = sum U_i V_i W_i.
                    # U has indices (i, j) corresponding to A[i, j].
                    # V has indices (j, k) corresponding to B[j, k].
                    # W has indices (k, i) corresponding to C[k, i].
                    
                    # So for U, we iterate i, j.
                    # Block of A for (i, j):
                    bi = i // 2
                    bj = j // 2
                    block_A = bi * 2 + bj # 0:00, 1:01, 2:10, 3:11? 
                    # My block mapping:
                    # 0: A11 (i<2, j<2)
                    # 1: A12 (i<2, j>=2)
                    # 2: A21 (i>=2, j<2)
                    # 3: A22 (i>=2, j>=2)
                    # So block_A = (i >= 2) * 2 + (j >= 2) ?
                    # If i<2, j<2 -> 0.
                    # If i<2, j>=2 -> 1.
                    # If i>=2, j<2 -> 2.
                    # If i>=2, j>=2 -> 3.
                    # Yes.
                    block_A = (1 if i >= 2 else 0) * 2 + (1 if j >= 2 else 0)
                    local_A = (i % 2) * 2 + (j % 2)
                    
                    coeff_U = cA[s][block_A] * u_local[t][local_A]
                    U[idx_U][r] += coeff_U
                    
                    # For V, we need to iterate j, k.
                    # But we are currently iterating i, j for U.
                    # We should do V in a separate loop or combined carefully.
                    # Let's do V in a separate loop for clarity.
            
            for j in range(4):
                for k in range(4):
                    idx_V = j * 4 + k
                    block_B = (1 if j >= 2 else 0) * 2 + (1 if k >= 2 else 0)
                    local_B = (j % 2) * 2 + (k % 2)
                    
                    coeff_V = cB[s][block_B] * v_local[t][local_B]
                    V[idx_V][r] += coeff_V

            # Populate W
            for k in range(4):
                for i in range(4):
                    idx_W = k * 4 + i
                    block_C = (1 if k >= 2 else 0) * 2 + (1 if i >= 2 else 0)
                    local_C = (k % 2) * 2 + (i % 2)
                    
                    coeff_W = K[s][block_C] * L[t][local_C]
                    W[idx_W][r] += coeff_W

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
