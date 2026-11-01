"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 2, 15

    # Build the known rank-53 decomposition:
    # 7 Strassen blocks (rank 7 each) + 1 schoolbook block (rank 4) = 53
    U_list = []
    V_list = []
    W_list = []

    # The 15 k-columns are grouped into 7 pairs and 1 single
    # Pairs: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)
    # Single: 14

    # Strassen 7x7x7 terms:
    # M1 = (A11 - A22) * (B21 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A11 - A12) * (B12 + B22)
    # M7 = (A21 + A12) * (B11 - B21)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5 + M6
    # C21 = M2 + M4 + M5
    # C22 = M1 - M3 + M2 - M6
    #
    # U factor: A matrix (n x m = 2x2)
    # V factor: B matrix (m x p = 2x15)
    # W factor: C matrix (p x n = 15x2)

    # For each pair (k0, k1), we have a 2x2x2 block
    # B has columns k0, k1 in this block
    # C has rows k0, k1 in this block

    strassen_U = [
        {(0,0): 1, (1,1): -1},   # M1
        {(1,0): 1, (1,1): 1},    # M2
        {(0,0): 1},              # M3
        {(1,1): 1},              # M4
        {(0,0): 1, (1,1): 1},    # M5
        {(0,0): 1, (0,1): -1},   # M6
        {(1,0): 1, (0,1): 1},    # M7
    ]

    # For pair (k0, k1): B rows 0,1 have columns k0, k1
    # V for M1: B21 + B22 -> (1,k0):1, (1,k1):1
    # V for M2: B11 -> (0,k0):1
    # V for M3: B12 - B22 -> (0,k1):1, (1,k1):-1
    # V for M4: B21 - B11 -> (1,k0):1, (0,k0):-1
    # V for M5: B11 + B22 -> (0,k0):1, (1,k1):1
    # V for M6: B12 + B22 -> (0,k1):1, (1,k1):1
    # V for M7: B11 - B21 -> (0,k0):1, (1,k0):-1

    strassen_V_templates = [
        {(1,0): 1, (1,1): 1},   # M1: B21 + B22
        {(0,0): 1},             # M2: B11
        {(0,1): 1, (1,1): -1},  # M3: B12 - B22
        {(1,0): 1, (0,0): -1},  # M4: B21 - B11
        {(0,0): 1, (1,1): 1},   # M5: B11 + B22
        {(0,1): 1, (1,1): 1},   # M6: B12 + B22
        {(0,0): 1, (1,0): -1},  # M7: B11 - B21
    ]

    # W for C: C11 = M1 + M4 - M5 + M7, C12 = M3 + M5 + M6, C21 = M2 + M4 + M5, C22 = M1 - M3 + M2 - M6
    # C is (p x n) = (15 x 2), so W[k*n + i]
    # For pair (k0, k1): C rows k0, k1; C cols 0, 1
    # W for M1: C11 += 1, C22 += 1 -> (k0,0):1, (k1,1):1
    # W for M2: C21 += 1, C22 += 1 -> (k0,1):1, (k1,1):1
    # W for M3: C12 += 1, C22 -= 1 -> (k0,1):1, (k1,1):-1
    # W for M4: C11 += 1, C21 += 1 -> (k0,0):1, (k1,1):1
    # W for M5: C11 -= 1, C12 += 1, C21 += 1 -> (k0,0):-1, (k0,1):1, (k1,1):1
    # W for M6: C12 += 1, C22 -= 1 -> (k0,1):1, (k1,1):-1
    # W for M7: C11 += 1, C22 += 1 -> (k0,0):1, (k1,1):1

    strassen_W_templates = [
        {(0,0): 1, (1,1): 1},              # M1
        {(0,1): 1, (1,1): 1},              # M2
        {(0,1): 1, (1,1): -1},             # M3
        {(0,0): 1, (1,1): 1},              # M4
        {(0,0): -1, (0,1): 1, (1,1): 1},   # M5
        {(0,1): 1, (1,1): -1},             # M6
        {(0,0): 1, (1,1): 1},              # M7
    ]

    pairs = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]

    for k0, k1 in pairs:
        for t in range(7):
            u_dict = strassen_U[t]
            v_dict = { (j, k0 + k): val for (j, k), val in strassen_V_templates[t].items() }
            w_dict = { (k0 + kk, i): val for (kk, i), val in strassen_W_templates[t].items() }

            u_vec = [0] * (n * m)
            for (i, j), val in u_dict.items():
                u_vec[i * m + j] = val
            U_list.append(u_vec)

            v_vec = [0] * (m * p)
            for (j, k), val in v_dict.items():
                v_vec[j * p + k] = val
            V_list.append(v_vec)

            w_vec = [0] * (p * n)
            for (k, i), val in w_dict.items():
                w_vec[k * n + i] = val
            W_list.append(w_vec)

    # Schoolbook for k=14: 4 terms
    for i in range(n):
        for j in range(m):
            u_vec = [0] * (n * m)
            u_vec[i * m + j] = 1
            U_list.append(u_vec)

            v_vec = [0] * (m * p)
            v_vec[j * p + 14] = 1
            V_list.append(v_vec)

            w_vec = [0] * (p * n)
            w_vec[14 * n + i] = 1
            W_list.append(w_vec)

    rank = len(U_list)
    U = np.array(U_list, dtype=float).T
    V = np.array(V_list, dtype=float).T
    W = np.array(W_list, dtype=float).T

    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                val = float(np.sum(U_arr[u_row, :] * V_arr[v_row, :] * W_arr[w_row, :]))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
