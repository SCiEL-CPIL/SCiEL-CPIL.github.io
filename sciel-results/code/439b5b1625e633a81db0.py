import numpy as np

def construct():
    n, m, p = 2, 3, 9

    # --- Factor 1: the (n, m, 1) tensor = matrix-vector product A * b ---
    # Rank of (2,3,1) tensor is n*m = 6.
    # U1 has shape (n*m, n*m), V1 has shape (m, n*m), W1 has shape (1, n*m).
    # For each (i, j): U1[i*m + j, i*m + j] = 1, V1[j, i*m + j] = 1, W1[0, i*m + j] = 1.
    r1 = n * m  # 6
    U1 = np.zeros((r1, r1), dtype=float)
    V1 = np.zeros((m, r1), dtype=float)
    W1 = np.zeros((1, r1), dtype=float)
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            U1[idx, idx] = 1.0
            V1[j, idx] = 1.0
            W1[0, idx] = 1.0

    # --- Factor 2: the (1, 1, p) tensor = column selection (identity on p columns) ---
    # Rank of (1,1,p) tensor is p = 9.
    # U2 has shape (1, p), V2 has shape (1, p), W2 has shape (p, p).
    # For each k: U2[0, k] = 1, V2[0, k] = 1, W2[k, k] = 1.
    r2 = p  # 9
    U2 = np.zeros((1, r2), dtype=float)
    V2 = np.zeros((1, r2), dtype=float)
    W2 = np.zeros((r2, r2), dtype=float)
    for k in range(p):
        U2[0, k] = 1.0
        V2[0, k] = 1.0
        W2[k, k] = 1.0

    # --- Kronecker product to combine the two factors ---
    # The combined tensor is T = T1 ⊗ T2.
    # Rank = r1 * r2 = 6 * 9 = 54.
    #
    # U_combined has shape (n*m * 1, r1 * r2) = (6, 54)
    #   = kron(U1, U2) but with appropriate reshaping.
    #   Actually: U_combined[a, b] where a = a1 (row of U1) * 1 + 0, b = b1 * r2 + b2
    #   U_combined[a1, b1*r2 + b2] = U1[a1, b1] * U2[0, b2]
    #
    # V_combined has shape (m * 1, r1 * r2) = (3, 54)
    #   V_combined[j, b1*r2 + b2] = V1[j, b1] * V2[0, b2]
    #
    # W_combined has shape (1 * p, r1 * r2) = (p, 54)  [output: p*n = 9*2 = 18? No...]
    #   Wait: output dimension is p * n = 9 * 2 = 18.
    #   W1 has shape (1, r1) and W2 has shape (p, r2).
    #   W_combined has shape (1 * p, r1 * r2) = (p, r1*r2) = (9, 54)? 
    #   But we need shape (p*n, rank) = (18, 54).
    #
    # Let me reconsider. The output index in the trace convention is W[k*n + i, r].
    # So the output has p*n = 18 entries.
    #
    # The (n, m, p) tensor: T[i, j, k] = delta (the multiplication tensor).
    # We can view this as: for fixed k, T[:, :, k] is the (n, m, 1) tensor.
    # So T = sum_k (T_{n,m,1} ⊗ E_k) where E_k is the k-th basis vector in R^p.
    #
    # More precisely: T[i, j, k] = T1[i, j, 0] * delta_{k, k}
    # This is a Kronecker product along the third mode.
    #
    # Let me build it directly:
    # U_combined[i*m + j, r] where r = r1_idx * r2 + r2_idx
    #   U_combined[i*m + j, r1_idx * r2 + r2_idx] = U1[i*m + j, r1_idx] * U2[0, r2_idx]
    #
    # V_combined[j*p + k, r] where r = r1_idx * r2 + r2_idx
    #   V_combined[j*p + k, r1_idx * r2 + r2_idx] = V1[j, r1_idx] * V2[0, r2_idx]
    #   But V1 has shape (m, r1) = (3, 6), and V2 has shape (1, r2) = (1, 9).
    #   V_combined row index: j*p + k. For fixed j, k ranges 0..p-1.
    #   V_combined[j*p + k, r1_idx * r2 + r2_idx] = V1[j, r1_idx] * (1 if r2_idx == k else 0)
    #   = V1[j, r1_idx] * V2[0, k] * delta_{r2_idx, k}
    #   Since V2[0, k] = 1 for all k, this is V1[j, r1_idx] when r2_idx == k.
    #
    # W_combined[k*n + i, r] where r = r1_idx * r2 + r2_idx
    #   W_combined[k*n + i, r1_idx * r2 + r2_idx] = W1[0, r1_idx] * W2[k, r2_idx]
    #   W1[0, r1_idx] = 1 for all r1_idx, W2[k, r2_idx] = delta_{k, r2_idx}
    #   So W_combined[k*n + i, r1_idx * r2 + k] = 1.

    rank = r1 * r2  # 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for i in range(n):
        for j in range(m):
            for r1_idx in range(r1):
                u_val = U1[i * m + j, r1_idx]
                if u_val != 0.0:
                    for k in range(p):
                        r = r1_idx * r2 + k
                        U[i * m + j, r] += u_val * U2[0, k]
                        V[j * p + k, r] += V1[j, r1_idx] * V2[0, k]
                        W[k * n + i, r] += W1[0, r1_idx] * W2[k, r2_idx]

    # Convert to lists of lists for the interface
    U_list = [[float(U[i][j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i][j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i][j]) for j in range(rank)] for i in range(p * n)]

    return {"U": U_list, "V": V_list, "W": W_list}
