import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin well under 120s

    n, m, p = DIMENSIONS
    rank = 49  # 7 * 7: outer Strassen (7 terms) x inner Strassen (7 terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2 coefficients (7 terms each for A, B, C linear forms)
    # a_coeffs[s][i][j]: coefficient of A[i][j] in s-th A-form
    # b_coeffs[s][j][k]: coefficient of B[j][k] in s-th B-form
    # c_coeffs[s][i][k]: coefficient of C[i][k] in s-th C-form
    #
    # Standard Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 - M3 + M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M5

    a_coeffs = [
        [[1, 0], [0, 1]],   # s=0: A00 + A11
        [[0, 0], [1, 1]],   # s=1: A10 + A11
        [[1, 0], [0, 0]],   # s=2: A00
        [[0, 0], [0, 1]],   # s=3: A11
        [[1, 1], [0, 0]],   # s=4: A00 + A01
        [[-1, 0], [1, 0]],  # s=5: A10 - A00
        [[0, 1], [0, -1]],  # s=6: A01 - A11
    ]

    b_coeffs = [
        [[1, 0], [0, 1]],   # s=0: B00 + B11
        [[1, 0], [0, 0]],   # s=1: B00
        [[0, 1], [0, -1]],  # s=2: B01 - B11
        [[-1, 0], [1, 0]],  # s=3: B10 - B00
        [[0, 0], [0, 1]],   # s=4: B11
        [[1, 1], [0, 0]],   # s=5: B00 + B01
        [[0, 0], [1, 1]],   # s=6: B10 + B11
    ]

    c_coeffs = [
        [[1, 0], [0, 1]],   # s=0: C00 + C11
        [[0, 0], [1, -1]],  # s=1: C10 - C11
        [[0, 1], [0, 1]],   # s=2: C01 + C11
        [[1, 0], [1, 0]],   # s=3: C00 + C10
        [[-1, 1], [0, 0]],  # s=4: -C00 + C01
        [[0, 0], [0, 1]],   # s=5: C11
        [[1, 0], [0, 0]],   # s=6: C00
    ]

    # Build U, V, W using the tensor product structure
    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t

            # U: row index = i*4 + j, where i = 2*S + i2, j = 2*T + j2
            for S in range(2):
                for T in range(2):
                    outer_a = a_coeffs[s][S][T]
                    if outer_a == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            inner_a = a_coeffs[t][i2][j2]
                            if inner_a == 0:
                                continue
                            i = 2 * S + i2
                            j = 2 * T + j2
                            row_u = i * 4 + j
                            U[row_u][c_idx] += outer_a * inner_a

            # V: row index = j*4 + k, where j = 2*T + j2, k = 2*K + k2
            for T in range(2):
                for K in range(2):
                    outer_b = b_coeffs[s][T][K]
                    if outer_b == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            inner_b = b_coeffs[t][j2][k2]
                            if inner_b == 0:
                                continue
                            j = 2 * T + j2
                            k = 2 * K + k2
                            row_v = j * 4 + k
                            V[row_v][c_idx] += outer_b * inner_b

            # W: row index = k*4 + i, where k = 2*K + k2, i = 2*S + i2
            for K in range(2):
                for S in range(2):
                    outer_c = c_coeffs[s][S][K]
                    if outer_c == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            inner_c = c_coeffs[t][i2][k2]
                            if inner_c == 0:
                                continue
                            k = 2 * K + k2
                            i = 2 * S + i2
                            row_w = k * 4 + i
                            W[row_w][c_idx] += outer_c * inner_c

            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    # Try to reduce rank by checking for linear dependencies
    # Convert to numpy arrays for efficient computation
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Check for zero columns
    nonzero_cols = []
    for c in range(rank):
        if np.any(U_arr[:, c] != 0) and np.any(V_arr[:, c] != 0) and np.any(W_arr[:, c] != 0):
            nonzero_cols.append(c)

    if len(nonzero_cols) < rank:
        U_arr = U_arr[:, nonzero_cols]
        V_arr = V_arr[:, nonzero_cols]
        W_arr = W_arr[:, nonzero_cols]
        rank = len(nonzero_cols)

    # Try to find linear dependencies among the rank terms
    # A term r is redundant if it can be expressed as a linear combination of others
    # We check by looking at the combined tensor structure
    # For each column c, check if U[:,c], V[:,c], W[:,c] are linearly dependent with others
    
    # Build a combined matrix where each column represents one rank term
    # The constraint is that the tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # must remain unchanged when we remove a column
    
    # Try removing each column and check if the tensor is preserved
    # This is expensive, so we only do it if we have time
    
    if time.time() < deadline - 1.0:
        # Build the full tensor T from the 49-term decomposition
        T_full = np.zeros((n, m, p))
        for r in range(rank):
            T_full += np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r]).reshape(n, m, p)
        
        # Try to remove each column and check if T is preserved
        # For efficiency, only try a subset
        removed = set()
        for c_try in range(rank):
            if c_try in removed:
                continue
            if time.time() > deadline - 0.5:
                break
            
            # Compute T without column c_try
            T_reduced = np.zeros((n, m, p))
            for r in range(rank):
                if r == c_try:
                    continue
                T_reduced += np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r]).reshape(n, m, p)
            
            if np.allclose(T_reduced, T_full, atol=1e-9):
                removed.add(c_try)
        
        if removed:
            keep_cols = [c for c in range(rank) if c not in removed]
            U_arr = U_arr[:, keep_cols]
            V_arr = V_arr[:, keep_cols]
            W_arr = W_arr[:, keep_cols]
            rank = len(keep_cols)

    # Convert back to lists
    U_out = U_arr.tolist()
    V_out = V_arr.tolist()
    W_out = W_arr.tolist()

    # Ensure integer values if possible
    U_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_out]
    V_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_out]
    W_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_out]

    return {"U": U_out, "V": V_out, "W": W_out}
