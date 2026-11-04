"""Matrix multiplication 4x4x9 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit

    n, m, p = DIMENSIONS
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36

    # Build the target tensor structure
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))

    triple_set = set(triples)

    # Precompute target tensors
    T_U = np.zeros((U_rows, V_rows, W_rows))
    T_V = np.zeros((V_rows, U_rows, W_rows))
    T_W = np.zeros((W_rows, U_rows, V_rows))
    for (a, b, c) in triples:
        T_U[a, b, c] = 1.0
        T_V[b, a, c] = 1.0
        T_W[c, a, b] = 1.0

    T_U_flat = T_U.reshape(U_rows, V_rows * W_rows)
    T_V_flat = T_V.reshape(V_rows, U_rows * W_rows)
    T_W_flat = T_W.reshape(W_rows, U_rows * V_rows)

    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((U_rows, rank_school))
    V_school = np.zeros((V_rows, rank_school))
    W_school = np.zeros((W_rows, rank_school))
    for col, (a, b, c) in enumerate(triples):
        U_school[a, col] = 1
        V_school[b, col] = 1
        W_school[c, col] = 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    def check_exact(U, V, W, tol=1e-9):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_recon - T_U)) < tol

    def round_and_check(U, V, W, scale=2):
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        if check_exact(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None

    # Try to find a lower rank decomposition using combinatorial design
    # For 4x4x9, we can try to use the structure to reduce rank.
    # Key idea: for each (i,j), the 9 k-terms form a 4x9 matrix product.
    # If we can find a rank-r decomposition that works uniformly, total rank = 16*r.

    # Try ranks from 143 down to 128 (16*8=128)
    for target_rank in range(rank_school - 1, 127, -1):
        if time.time() > deadline - 20:
            break

        # Try a few seeds with ALS
        for seed in range(3):
            if time.time() > deadline - 20:
                break

            rng = np.random.RandomState(seed)
            U = rng.randn(U_rows, target_rank) * 0.01
            V = rng.randn(V_rows, target_rank) * 0.01
            W = rng.randn(W_rows, target_rank) * 0.01

            for it in range(30):
                if time.time() > deadline - 20:
                    break

                # Update U
                B_tensor = np.einsum('br,cr->bcr', V, W)
                B = B_tensor.transpose(2, 0, 1).reshape(target_rank, V_rows * W_rows)
                G = B @ B.T + 1e-10 * np.eye(target_rank)
                try:
                    G_inv = np.linalg.inv(G)
                except:
                    G_inv = np.linalg.pinv(G)
                U = (G_inv @ (B @ T_U_flat.T)).T

                # Update V
                C_tensor = np.einsum('ar,cr->acr', U, W)
                C = C_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * W_rows)
                Gv = C @ C.T + 1e-10 * np.eye(target_rank)
                try:
                    Gv_inv = np.linalg.inv(Gv)
                except:
                    Gv_inv = np.linalg.pinv(Gv)
                V = (Gv_inv @ (C @ T_V_flat.T)).T

                # Update W
                D_tensor = np.einsum('ar,br->abr', U, V)
                D = D_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * V_rows)
                Gw = D @ D.T + 1e-10 * np.eye(target_rank)
                try:
                    Gw_inv = np.linalg.inv(Gw)
                except:
                    Gw_inv = np.linalg.pinv(Gw)
                W = (Gw_inv @ (D @ T_W_flat.T)).T

            # Check if we found a valid solution
            if check_exact(U, V, W, tol=1e-5):
                rounded = round_and_check(U, V, W)
                if rounded is not None:
                    best_U, best_V, best_W = rounded
                    best_rank = target_rank
                    break

        if best_rank < rank_school:
            break

    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
