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

    def round_and_check_multi(U, V, W):
        """Try multiple rounding scales to find exact solution."""
        for scale in [2, 4, 8, 16]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if check_exact(U_r, V_r, W_r):
                return U_r, V_r, W_r
        # Also try with scale 1 (integers)
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if check_exact(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None

    def als_decompose(target_rank, seed, max_iters=50, tol=1e-6):
        """ALS decomposition with early stopping."""
        rng = np.random.RandomState(seed)
        U = rng.randn(U_rows, target_rank) * 0.05
        V = rng.randn(V_rows, target_rank) * 0.05
        W = rng.randn(W_rows, target_rank) * 0.05

        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            # Update U
            B_tensor = np.einsum('br,cr->bcr', V, W)
            B = B_tensor.transpose(2, 0, 1).reshape(target_rank, V_rows * W_rows)
            G = B @ B.T + 1e-12 * np.eye(target_rank)
            try:
                U = np.linalg.solve(G, B @ T_U_flat.T).T
            except:
                U = (np.linalg.pinv(G) @ (B @ T_U_flat.T)).T

            # Update V
            C_tensor = np.einsum('ar,cr->acr', U, W)
            C = C_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * W_rows)
            Gv = C @ C.T + 1e-12 * np.eye(target_rank)
            try:
                V = np.linalg.solve(Gv, C @ T_V_flat.T).T
            except:
                V = (np.linalg.pinv(Gv) @ (C @ T_V_flat.T)).T

            # Update W
            D_tensor = np.einsum('ar,br->abr', U, V)
            D = D_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * V_rows)
            Gw = D @ D.T + 1e-12 * np.eye(target_rank)
            try:
                W = np.linalg.solve(Gw, D @ T_W_flat.T).T
            except:
                W = (np.linalg.pinv(Gw) @ (D @ T_W_flat.T)).T

            # Check convergence
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T_U))
            if err < tol:
                break
            if abs(prev_err - err) < 1e-10:
                break
            prev_err = err

        return U, V, W, err

    # Try structured ranks based on matrix multiplication structure
    # Key structural ranks to try: 143, 142, 140, 136, 132, 128
    # Also try 143 first as it's closest to schoolbook
    structural_ranks = [143, 142, 141, 140, 138, 136, 132, 128]
    
    for target_rank in structural_ranks:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline - 15:
            break

        # Try multiple seeds
        for seed in range(5):
            if time.time() > deadline - 10:
                break

            U, V, W, err = als_decompose(target_rank, seed, max_iters=40, tol=1e-7)
            
            if err < 1e-5:
                rounded = round_and_check_multi(U, V, W)
                if rounded is not None:
                    best_U, best_V, best_W = rounded
                    best_rank = target_rank
                    break

        if best_rank < rank_school:
            break

    # If we haven't found improvement, try a few more ranks with more seeds
    if best_rank == rank_school and time.time() < deadline - 30:
        for target_rank in range(143, 127, -1):
            if target_rank >= best_rank:
                continue
            if time.time() > deadline - 15:
                break

            for seed in range(3):
                if time.time() > deadline - 10:
                    break

                U, V, W, err = als_decompose(target_rank, seed + 100, max_iters=30, tol=1e-6)
                
                if err < 1e-5:
                    rounded = round_and_check_multi(U, V, W)
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
