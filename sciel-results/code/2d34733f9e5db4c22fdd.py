import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 118  # safety margin

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-8):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def get_rank50():
        """Strassen-based rank-50 decomposition."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    # Get baseline
    U50, V50, W50 = get_rank50()
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # Structural approach: directly construct lower-rank decompositions
    # by searching for valid support patterns and solving linear systems.
    #
    # Key idea: A rank-r decomposition has r terms, each with support sets
    # S_U, S_V, S_W. The tensor entry T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r].
    # If we fix the support patterns, we get a linear system in the coefficients.
    #
    # For 2x4x7, we can exploit structure:
    # - Row 0 of A interacts with all 4 cols, Row 1 similarly
    # - We can group terms by which A-row and B-col they cover

    def try_construct_rank(target_rank):
        """
        Try to construct a valid rank-target_rank decomposition by:
        1. Starting from rank-50, identifying removable/redundant columns
        2. Using structured patterns based on Strassen sub-blocks
        """
        if time.time() > deadline - 5:
            return None, None, None, False

        # Strategy: modify the rank-50 by combining/eliminating terms
        # The 6 rank-1 tail terms (for column 6) can potentially be merged
        # with Strassen terms that already touch column 6.

        # Alternative: build from scratch with clever grouping
        # Group B-columns into: {0,1,2,3,4,5} handled by Strassen pairs,
        # and {6} handled separately. But we can try to absorb col 6
        # into the Strassen blocks.

        # Let's try a different tiling: use 2x2x3 Strassen-like blocks
        # A 2x2x3 block (2 A-rows, 2 A-cols/B-rows, 3 B-cols) needs rank ~9-10
        # instead of 7+3=10. Actually 2x2x2 is rank 7, 2x2x3 could be rank 10.

        # Better approach: try to find a rank-48 by removing 2 terms
        # and adjusting. Use the fact that some Strassen terms may be
        # linearly dependent when combined.

        # Direct construction attempt for rank 48:
        # Use 6 Strassen 2x2x2 blocks (42 terms) + 6 modified tail terms
        # where we merge the last 2 tail terms into the Strassen blocks.

        # Actually, let's try a completely different structural approach:
        # Decompose as sum of "row-wise" contributions.
        # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:] + A[0,2]*B[2,:] + A[0,3]*B[3,:]
        # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:] + A[1,2]*B[2,:] + A[1,3]*B[3,:]

        # Each row of C is a linear combination of 4 rows of B.
        # The tensor for row i is: sum_j A[i,j] * B[j,:] -> C[i,:]
        # This is a 1x4x7 tensor for each i, total 2 such tensors.

        # A 1x4x7 tensor has rank at most 7 (schoolbook).
        # But we can do better: 1x4x4 is rank 4, 1x4x3 is rank 3, total 7.
        # However, both rows share the same B structure, so we can overlap.

        # Let's try: use rank-4 for the first 4 cols of B, rank-3 for last 3.
        # For each of the 2 A-rows: 4 + 3 = 7 terms, total 14.
        # But that's just schoolbook for 2x4x7 restricted... no, that's 14 total.
        # Wait, that's not right. Each "1x4x7" needs rank 7, so 2 rows = 14.
        # But schoolbook is 56, so we're already at 14 which is way better.
        # The issue: the 14 terms don't share structure between rows.

        # Actually, the correct lower bound: matricization rank.
        T1 = T.reshape(n * m, -1)  # 8 x 392
        T2 = T.reshape(m * p, -1)  # 28 x 112
        T3 = T.reshape(p * n, -1)  # 14 x 224

        r1 = np.linalg.matrix_rank(T1)
        r2 = np.linalg.matrix_rank(T2)
        r3 = np.linalg.matrix_rank(T3)
        lower_bound = max(r1, r2, r3)

        # The true rank is at least lower_bound.
        # For 2x4x7, let's check what the lower bound is.

        # Try ALS from rank-50 with aggressive restarts for ranks below 50
        T1f = T.reshape(n * m, -1)
        T2f = T.reshape(m * p, -1)
        T3f = T.reshape(p * n, -1)

        def khatri_rao(A, B):
            M, R = A.shape
            N, _ = B.shape
            KR = np.zeros((M * N, R))
            for r in range(R):
                KR[:, r] = np.kron(A[:, r], B[:, r])
            return KR

        def als_one_shot(T1m, T2m, T3m, U_init, V_init, W_init, max_iters=200, tol=1e-13):
            R = U_init.shape[1]
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                KR_VW = khatri_rao(V, W)
                try:
                    U_new = T1m @ np.linalg.pinv(KR_VW)
                except:
                    break
                KR_WU = khatri_rao(W, U)
                try:
                    V_new = T2m @ np.linalg.pinv(KR_WU)
                except:
                    break
                KR_UV = khatri_rao(U, V)
                try:
                    W_new = T3m @ np.linalg.pinv(KR_UV)
                except:
                    break
                change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
                U, V, W = U_new, V_new, W_new
                if change < tol:
                    break
            return U, V, W

        def polish_decomposition(U, V, W, max_iters=200, tol=1e-14):
            """High-precision polish using alternating least squares."""
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                KR_VW = khatri_rao(V, W)
                try:
                    U_new = T1f @ np.linalg.pinv(KR_VW)
                except:
                    break
                KR_WU = khatri_rao(W, U)
                try:
                    V_new = T2f @ np.linalg.pinv(KR_WU)
                except:
                    break
                KR_UV = khatri_rao(U, V)
                try:
                    W_new = T3f @ np.linalg.pinv(KR_UV)
                except:
                    break
                change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
                U, V, W = U_new, V_new, W_new
                if change < tol:
                    break
            return U, V, W

        # Try reducing rank from 49 down to max(lower_bound, 30)
        min_try = max(lower_bound, 35)
        rng = np.random.default_rng(12345)

        for target_rank in range(min(best_rank - 1, 49), min_try - 1, -1):
            if time.time() > deadline - 8:
                break

            time_left = deadline - 6 - time.time()
            if time_left < 3:
                break

            n_restarts = max(5, min(20, int(time_left / 1.5)))

            for restart in range(n_restarts):
                if time.time() > deadline - 5:
                    break

                if restart == 0:
                    # Truncate reference with small noise
                    U_init = U50[:, :target_rank].copy() + rng.normal(0, 0.01, (n*m, target_rank))
                    V_init = V50[:, :target_rank].copy() + rng.normal(0, 0.01, (m*p, target_rank))
                    W_init = W50[:, :target_rank].copy() + rng.normal(0, 0.01, (p*n, target_rank))
                elif restart == 1:
                    # Pure random
                    U_init = rng.normal(0, 1, (n * m, target_rank))
                    V_init = rng.normal(0, 1, (m * p, target_rank))
                    W_init = rng.normal(0, 1, (p * n, target_rank))
                elif restart == 2:
                    # Larger perturbation of reference
                    U_init = U50[:, :target_rank].copy() + rng.normal(0, 0.05, (n*m, target_rank))
                    V_init = V50[:, :target_rank].copy() + rng.normal(0, 0.05, (m*p, target_rank))
                    W_init = W50[:, :target_rank].copy() + rng.normal(0, 0.05, (p*n, target_rank))
                elif restart == 3:
                    # Structured sparse: keep Strassen-like patterns
                    U_init = np.zeros((n * m, target_rank))
                    V_init = np.zeros((m * p, target_rank))
                    W_init = np.zeros((p * n, target_rank))
                    for r in range(target_rank):
                        # Random 2x2x2 Strassen-like support
                        i0 = rng.integers(0, n-1)
                        j0 = rng.integers(0, m-1)
                        k0 = rng.integers(0, p-1)
                        signs = rng.choice([-1, 1], (3,))
                        U_init[i0*m+j0, r] = signs[0]
                        U_init[(i0+1)*m+(j0+1), r] = signs[0]
                        V_init[j0*p+k0, r] = signs[1]
                        V_init[(j0+1)*p+(k0+1), r] = signs[1]
                        W_init[k0*n+i0, r] = signs[2]
                        W_init[(k0+1)*n+(i0+1), r] = signs[2]
                    # Add small noise to break symmetry
                    U_init += rng.normal(0, 0.01, (n*m, target_rank))
                    V_init += rng.normal(0, 0.01, (m*p, target_rank))
                    W_init += rng.normal(0, 0.01, (p*n, target_rank))
                elif restart == 4:
                    # Binary sparse init
                    U_init = rng.choice([-1, 0, 1], (n * m, target_rank)).astype(float)
                    V_init = rng.choice([-1, 0, 1], (m * p, target_rank)).astype(float)
                    W_init = rng.choice([-1, 0, 1], (p * n, target_rank)).astype(float)
                    # Ensure non-zero columns
                    for r in range(target_rank):
                        if np.all(U_init[:, r] == 0):
                            U_init[rng.integers(0, n*m), r] = 1
                        if np.all(V_init[:, r] == 0):
                            V_init[rng.integers(0, m*p), r] = 1
                        if np.all(W_init[:, r] == 0):
                            W_init[rng.integers(0, p*n), r] = 1
                elif restart == 5:
                    # Blend reference and random
                    U_init = 0.7 * U50[:, :target_rank] + 0.3 * rng.normal(0, 1, (n*m, target_rank))
                    V_init = 0.7 * V50[:, :target_rank] + 0.3 * rng.normal(0, 1, (m*p, target_rank))
                    W_init = 0.7 * W50[:, :target_rank] + 0.3 * rng.normal(0, 1, (p*n, target_rank))
                elif restart == 6:
                    # Rank-revealing: use SVD of truncated tensor matricization
                    T1_trunc = T1f[:, :target_rank * (m*p) // (n*m)] if target_rank * (m*p) // (n*m) > 0 else T1f[:, :1]
                    try:
                        U_init, _, _ = np.linalg.svd(T1_trunc, full_matrices=False)
                        if U_init.shape[1] < target_rank:
                            U_init = np.hstack([U_init, rng.normal(0, 0.1, (n*m, target_rank - U_init.shape[1]))])
                        else:
                            U_init = U_init[:, :target_rank]
                    except:
                        U_init = rng.normal(0, 1, (n * m, target_rank))
                    V_init = rng.normal(0, 1, (m * p, target_rank))
                    W_init = rng.normal(0, 1, (p * n, target_rank))
                elif restart == 7:
                    # Geometric scaling
                    scale = 1.0 / (target_rank ** 0.25)
                    U_init = rng.normal(0, scale, (n * m, target_rank))
                    V_init = rng.normal(0, 1/scale, (m * p, target_rank))
                    W_init = rng.normal(0, 1, (p * n, target_rank))
                elif restart == 8:
                    # Copy some columns from reference, rest random
                    n_copy = min(target_rank // 2, 50)
                    U_init = np.hstack([U50[:, :n_copy], rng.normal(0, 1, (n*m, target_rank - n_copy))])
                    V_init = np.hstack([V50[:, :n_copy], rng.normal(0, 1, (m*p, target_rank - n_copy))])
                    W_init = np.hstack([W50[:, :n_copy], rng.normal(0, 1, (p*n, target_rank - n_copy))])
                else:
                    # Varying scale random
                    U_init = rng.normal(0, 0.5 + 0.1*restart, (n * m, target_rank))
                    V_init = rng.normal(0, 1.0, (m * p, target_rank))
                    W_init = rng.normal(0, 0.8, (p * n, target_rank))

                U_sol, V_sol, W_sol = als_one_shot(T1f, T2f, T3f, U_init, V_init, W_init, max_iters=150)

                recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                residual = np.linalg.norm(recon - T)

                if residual < 1e-6:
                    # High-precision polish
                    U_sol, V_sol, W_sol = polish_decomposition(U_sol, V_sol, W_sol, max_iters=100)
                    if verify_decomp(U_sol, V_sol, W_sol):
                        return U_sol, V_sol, W_sol, True

        return None, None, None, False

    # Try to find lower rank, starting from 49 and going down
    for target_rank in range(49, max(35, best_rank - 15), -1):
        if time.time() > deadline - 8:
            break
        U_sol, V_sol, W_sol, success = try_construct_rank(target_rank)
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            # If we found a solution, try one more rank lower if time permits
            if time.time() < deadline - 10 and target_rank > 36:
                U_sol, V_sol, W_sol, success = try_construct_rank(target_rank - 1)
                if success:
                    best_rank = target_rank - 1
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
