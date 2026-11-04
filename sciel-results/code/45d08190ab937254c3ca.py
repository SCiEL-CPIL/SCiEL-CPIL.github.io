import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Strassen 2x2*2x2 factors
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U_s[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i, m_idx] += c

    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Part 1: 4x4*4x4 Strassen (rank 49 = 7*7)
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            for i in range(4):
                I = i // 2
                i0 = i % 2
                for j in range(4):
                    J = j // 2
                    j0 = j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val

            for j in range(4):
                J = j // 2
                j0 = j % 2
                for k in range(4):
                    K = k // 2
                    k0 = k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val

            for k in range(4):
                K = k // 2
                k0 = k % 2
                for i in range(4):
                    I = i // 2
                    i0 = i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # Part 2: 4x4*4x2 (rank 28 = 4*7)
    block_products = [
        (0, 0, 0, 4, 0, 4),
        (0, 2, 2, 4, 0, 4),
        (2, 0, 0, 4, 2, 4),
        (2, 2, 2, 4, 2, 4),
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val

            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    # Compression: Merge duplicates and cancel negatives
    # We check all pairs of columns
    # If U[:, i] == U[:, j] and V[:, i] == V[:, j] and W[:, i] == W[:, j]:
    #   Merge j into i (scale i by 2, zero j)
    # If U[:, i] == -U[:, j] and V[:, i] == V[:, j] and W[:, i] == W[:, j]:
    #   Cancel both (zero i and j)
    
    # To be efficient, we can just do a few passes
    changed = True
    passes = 0
    while changed and passes < 10 and time.time() < deadline - 20:
        changed = False
        passes += 1
        for i in range(total_rank):
            if np.all(U[:, i] == 0) and np.all(V[:, i] == 0) and np.all(W[:, i] == 0):
                continue
            for j in range(i + 1, total_rank):
                if np.all(U[:, j] == 0) and np.all(V[:, j] == 0) and np.all(W[:, j] == 0):
                    continue
                
                # Check for exact match
                if np.allclose(U[:, i], U[:, j]) and np.allclose(V[:, i], V[:, j]) and np.allclose(W[:, i], W[:, j]):
                    # Merge j into i
                    U[:, i] = U[:, i] * 2
                    V[:, i] = V[:, i] * 2
                    W[:, i] = W[:, i] * 2
                    U[:, j] = 0
                    V[:, j] = 0
                    W[:, j] = 0
                    changed = True
                # Check for negative match (cancellation)
                elif np.allclose(U[:, i], -U[:, j]) and np.allclose(V[:, i], V[:, j]) and np.allclose(W[:, i], W[:, j]):
                    # Cancel both
                    U[:, i] = 0
                    V[:, i] = 0
                    W[:, i] = 0
                    U[:, j] = 0
                    V[:, j] = 0
                    W[:, j] = 0
                    changed = True
                    break # Move to next i

    # Compact non-zero columns
    keep_cols = []
    for c in range(total_rank):
        if not (np.all(U[:, c] == 0) and np.all(V[:, c] == 0) and np.all(W[:, c] == 0)):
            keep_cols.append(c)

    final_U = U[:, keep_cols]
    final_V = V[:, keep_cols]
    final_W = W[:, keep_cols]

    # Round to integers/half-integers
    final_U = np.round(final_U * 2) / 2
    final_V = np.round(final_V * 2) / 2
    final_W = np.round(final_W * 2) / 2

    def _clean_matrix(mat):
        out = []
        for row in mat:
            cleaned = []
            for entry in row:
                half = round(entry * 2) / 2
                if abs(entry - half) < 1e-9:
                    if half == int(half):
                        cleaned.append(int(half))
                    else:
                        cleaned.append(float(half))
                else:
                    cleaned.append(float(entry))
            out.append(cleaned)
        return out

    return {"U": _clean_matrix(final_U), "V": _clean_matrix(final_V), "W": _clean_matrix(final_W)}
