"""Matrix multiplication 3x3x8 rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 115.0

    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72

    # Build target tensor T[a, b, c] where:
    # a = i*m + j  (i in 0..n-1, j in 0..m-1)
    # b = j*p + k  (j in 0..m-1, k in 0..p-1)
    # c = k*n + i  (k in 0..p-1, i in 0..n-1)
    # T[a,b,c] = 1 iff consistent (j==j', k==k', i==i')
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Precompute flattened targets for each mode
    T_U = T.reshape(n * m, -1)  # (n*m, m*p*p*n)
    T_V = np.moveaxis(T, 1, 0).reshape(m * p, -1)  # (m*p, n*m*p*n)
    T_W = np.moveaxis(T, 2, 0).reshape(p * n, -1)  # (p*n, n*m*m*p)

    def compute_error(U, V, W):
        """Compute max absolute error of tensor reconstruction."""
        Pred = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(Pred - T))

    def round_to_half(mat):
        """Round matrix entries to nearest 0.5 if close."""
        r = np.round(mat * 2) / 2.0
        if np.allclose(mat, r, atol=1e-6):
            return r
        return mat

    def to_output(mat):
        """Convert numpy array to list of lists with int/float entries."""
        result = []
        for row in mat:
            row_list = []
            for x in row:
                x2 = x * 2.0
                if abs(x2 - round(x2)) < 1e-6:
                    val = round(x2) / 2.0
                    if abs(val - round(val)) < 1e-6:
                        row_list.append(int(round(val)))
                    else:
                        row_list.append(float(val))
                else:
                    row_list.append(float(x))
            result.append(row_list)
        return result

    # ALS solver with multiple restarts
    def als_solve(U_init, V_init, W_init, max_iter=200, tol=1e-10):
        """Run ALS to converge to a decomposition."""
        U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
        rank = U.shape[1]

        prev_err = float('inf')
        for it in range(max_iter):
            if time.time() > DEADLINE:
                break

            # Solve for U: fix V, W
            # H[(b,c), r] = V[b,r] * W[c,r]
            H = np.einsum('br,cr->bcr', V, W).reshape(-1, rank)
            U_new_T = np.linalg.lstsq(H, T_U.T, rcond=None)[0]
            U = U_new_T.T

            # Solve for V: fix U, W
            # K[(a,c), r] = U[a,r] * W[c,r]
            K = np.einsum('ar,cr->acr', U, W).reshape(-1, rank)
            V_new_T = np.linalg.lstsq(K, T_V.T, rcond=None)[0]
            V = V_new_T.T

            # Solve for W: fix U, V
            # M[(a,b), r] = U[a,r] * V[b,r]
            M = np.einsum('ar,br->abr', U, V).reshape(-1, rank)
            W_new_T = np.linalg.lstsq(M, T_W.T, rcond=None)[0]
            W = W_new_T.T

            # Check convergence
            if it % 5 == 0 or it == max_iter - 1:
                err = compute_error(U, V, W)
                if err < tol:
                    break
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err

        err = compute_error(U, V, W)
        return U, V, W, err

    # Try to find minimal rank decomposition
    best_rank = schoolbook_rank
    best_U = np.zeros((n * m, schoolbook_rank))
    best_V = np.zeros((m * p, schoolbook_rank))
    best_W = np.zeros((p * n, schoolbook_rank))

    # Initialize schoolbook as fallback
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1.0
                best_V[j * p + k, col] = 1.0
                best_W[k * n + i, col] = 1.0
                col += 1

    # Search ranks from lower to higher
    # Known results: 3x3x3 has rank 23, 3x3x4 likely ~30-35, 3x3x8 likely ~50-60
    ranks_to_try = list(range(45, 72))

    for target_rank in ranks_to_try:
        if time.time() > DEADLINE - 5:
            break

        found = False
        # Multiple random restarts
        for attempt in range(10):
            if time.time() > DEADLINE - 5:
                break

            np.random.seed(12345 + attempt * 7919 + target_rank * 31)

            # Try different initialization strategies
            if attempt % 3 == 0:
                # Small random
                scale = 0.1
            elif attempt % 3 == 1:
                # Medium random
                scale = 0.5
            else:
                # Sparse-like: mostly zeros with some ones
                scale = 1.0

            U_init = np.random.randn(n * m, target_rank) * scale
            V_init = np.random.randn(m * p, target_rank) * scale
            W_init = np.random.randn(p * n, target_rank) * scale

            U, V, W, err = als_solve(U_init, V_init, W_init, max_iter=300)

            if err < 1e-8:
                # Try rounding to half-integers
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                err_r = compute_error(U_r, V_r, W_r)

                if err_r < 1e-6:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break
                else:
                    # Use unrounded if it's valid
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break

        if found:
            continue

    # Final verification
    final_err = compute_error(best_U, best_V, best_W)
    if final_err > 1e-6:
        # Fallback to schoolbook
        best_rank = schoolbook_rank
        best_U = np.zeros((n * m, schoolbook_rank))
        best_V = np.zeros((m * p, schoolbook_rank))
        best_W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j, col] = 1.0
                    best_V[j * p + k, col] = 1.0
                    best_W[k * n + i, col] = 1.0
                    col += 1

    return {"U": to_output(best_U), "V": to_output(best_V), "W": to_output(best_W)}

# EVOLVE-BLOCK-END
