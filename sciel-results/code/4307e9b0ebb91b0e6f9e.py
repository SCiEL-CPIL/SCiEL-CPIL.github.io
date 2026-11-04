import numpy as np
import time

def construct():
    """
    Rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Kronecker product of two levels of Strassen (2x2x2 rank-7)
    to get a rank-49 decomposition for 4x4x4, then applies it independently
    to the two 4x4 column blocks of B, giving total rank 98.
    """
    start_time = time.time()
    DEADLINE = 150.0
    MARGIN = 5.0

    n, m, p = 4, 4, 8

    # Strassen 2x2x2 rank-7 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)

    # Verify 2x2x2 Strassen
    def verify_2x2x2():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0.0
                    for t in range(7):
                        total += U2[i*2+j, t] * V2[j*2+k, t] * W2[k*2+i, t]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    if not verify_2x2x2():
        return construct_naive(n, m, p)

    # Build 4x4x4 rank-49 decomposition via Kronecker product
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)

    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]

    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]

    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]

    # Verify 4x4x4 decomposition
    def verify_4x4x4():
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    total = 0.0
                    for c in range(49):
                        total += U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    if not verify_4x4x4():
        return construct_naive(n, m, p)

    # Build 4x4x8 rank-98 decomposition
    # Split B into two 4x4 blocks: B_left (cols 0-3) and B_right (cols 4-7)
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)   # 16 x 98
    V = np.zeros((m*p, rank), dtype=float)   # 32 x 98
    W = np.zeros((p*n, rank), dtype=float)   # 32 x 98

    # First block (output cols 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]

    # Second block (output cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]

    # Full verification
    def verify_full():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    if not verify_full():
        return construct_naive(n, m, p)

    # Deadline guard
    if time.time() > DEADLINE - MARGIN:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a lower-rank decomposition via optimization
    # We'll attempt to find a rank-R decomposition for R < 98
    # Start with R = 97 and work down, but only if time permits

    def try_optimization(target_rank, U_init, V_init, W_init, max_iter=200, tol=1e-8):
        """
        Try to optimize a decomposition of given rank using gradient-free method.
        This is a heuristic attempt to find a valid lower-rank decomposition.
        """
        U_opt = U_init.copy()
        V_opt = V_init.copy()
        W_opt = W_init.copy()

        # Simple iterative refinement using random perturbations
        best_err = float('inf')
        best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()

        def compute_error(U_mat, V_mat, W_mat):
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for r in range(target_rank):
                            total += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                        err += (total - 1.0) ** 2
            return err

        current_err = compute_error(U_opt, V_opt, W_opt)
        best_err = current_err

        # Gradient-free search: try small perturbations
        for iteration in range(max_iter):
            if time.time() > DEADLINE - MARGIN:
                break
            if current_err < tol:
                break

            # Random perturbation
            scale = max(0.01, 0.1 * np.sqrt(current_err))
            U_pert = U_opt + np.random.normal(0, scale, U_opt.shape)
            V_pert = V_opt + np.random.normal(0, scale, V_opt.shape)
            W_pert = W_opt + np.random.normal(0, scale, W_opt.shape)

            pert_err = compute_error(U_pert, V_pert, W_pert)

            if pert_err < current_err:
                U_opt, V_opt, W_opt = U_pert, V_pert, W_pert
                current_err = pert_err
                if current_err < best_err:
                    best_err = current_err
                    best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            else:
                # Reduce scale if we're not making progress
                scale *= 0.9

        return best_U, best_V, best_W, best_err

    # Try a few ranks below 98 if time permits
    # But this is very unlikely to succeed, so we'll be conservative
    if time.time() < DEADLINE - 30:  # Only if we have 30+ seconds left
        # Try rank 97 with random initialization based on the rank-98 solution
        for target_rank in [97, 96, 95]:
            if time.time() > DEADLINE - MARGIN:
                break

            # Initialize by truncating the rank-98 solution
            U_init = U[:, :target_rank].copy()
            V_init = V[:, :target_rank].copy()
            W_init = W[:, :target_rank].copy()

            # Add some random perturbation to escape local minima
            U_init += np.random.normal(0, 0.1, U_init.shape)
            V_init += np.random.normal(0, 0.1, V_init.shape)
            W_init += np.random.normal(0, 0.1, W_init.shape)

            max_iter = 50 if target_rank > 96 else 30
            U_opt, V_opt, W_opt, opt_err = try_optimization(target_rank, U_init, V_init, W_init, max_iter)

            if opt_err < 1e-6:
                # Found a valid lower-rank decomposition!
                # Round to nearest half-integer if possible
                def round_to_half_integer(mat):
                    return np.round(mat * 2) / 2

                U_round = round_to_half_integer(U_opt)
                V_round = round_to_half_integer(V_opt)
                W_round = round_to_half_integer(W_opt)

                # Verify the rounded version
                err_round = compute_error(U_round, V_round, W_round) if False else 0.0
                # Recompute error properly
                err_round = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for r in range(target_rank):
                                total += U_round[i*m+j, r] * V_round[j*p+k, r] * W_round[k*n+i, r]
                            err_round += (total - 1.0) ** 2

                if err_round < 1e-6:
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
