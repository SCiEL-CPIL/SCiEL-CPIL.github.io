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

        # Try reducing rank from 49 down to max(lower_bound, 30)
        min_try = max(lower_bound, 30)
        rng = np.random.default_rng(12345)

        for target_rank in range(min(best_rank - 1, 49), min_try - 1, -1):
            if time.time() > deadline - 8:
                break

            time_left = deadline - 6 - time.time()
            if time_left < 3:
                break

            n_restarts = max(3, min(12, int(time_left / 2)))

            for restart in range(n_restarts):
                if time.time() > deadline - 5:
                    break

                if restart == 0:
                    # Truncate reference
                    U_init = U50[:, :target_rank].copy() + rng.normal(0, 0.02, (n*m, target_rank))
                    V_init = V50[:, :target_rank].copy() + rng.normal(0, 0.02, (m*p, target_rank))
                    W_init = W50[:, :target_rank].copy() + rng.normal(0, 0.02, (p*n, target_rank))
                elif restart == 1:
                    U_init = rng.normal(0, 1, (n * m, target_rank))
                    V_init = rng.normal(0, 1, (m * p, target_rank))
                    W_init = rng.normal(0, 1, (p * n, target_rank))
                elif restart == 2:
                    # Perturb more
                    U_init = U50[:, :target_rank].copy() + rng.normal(0, 0.1, (n*m, target_rank))
                    V_init = V50[:, :target_rank].copy() + rng.normal(0, 0.1, (m*p, target_rank))
                    W_init = W50[:, :target_rank].copy() + rng.normal(0, 0.1, (p*n, target_rank))
                elif restart == 3:
                    # Scale variations
                    U_init = rng.normal(0, 0.5, (n * m, target_rank))
                    V_init = rng.normal(0, 1.5, (m * p, target_rank))
                    W_init = rng.normal(0, 0.8, (p * n, target_rank))
                elif restart == 4:
                    # Sparse random init
                    U_init = rng.choice([-1, 0, 1], (n * m, target_rank)).astype(float)
                    V_init = rng.choice([-1, 0, 1], (m * p, target_rank)).astype(float)
                    W_init = rng.choice([-1, 0, 1], (p * n, target_rank)).astype(float)
                elif restart == 5:
                    # Mix of reference and random
                    U_init = 0.5 * U50[:, :target_rank] + 0.5 * rng.normal(0, 1, (n*m, target_rank))
                    V_init = 0.5 * V50[:, :target_rank] + 0.5 * rng.normal(0, 1, (m*p, target_rank))
                    W_init = 0.5 * W50[:, :target_rank] + 0.5 * rng.normal(0, 1, (p*n, target_rank))
                else:
                    # Different random seed pattern
                    U_init = rng.normal(0, 1 + restart*0.1, (n * m, target_rank))
                    V_init = rng.normal(0, 1, (m * p, target_rank))
                    W_init = rng.normal(0, 1, (p * n, target_rank))

                U_sol, V_sol, W_sol = als_one_shot(T1f, T2f, T3f, U_init, V_init, W_sol if False else W_init, max_iters=100)

                recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                residual = np.linalg.norm(recon - T)

                if residual < 1e-7:
                    # Polish
                    for _ in range(50):
                        if time.time() > deadline - 3:
                            break
                        KR_VW = khatri_rao(V_sol, W_sol)
                        U_sol = T1f @ np.linalg.pinv(KR_VW)
                        KR_WU = khatri_rao(W_sol, U_sol)
                        V_sol = T2f @ np.linalg.pinv(KR_WU)
                        KR_UV = khatri_rao(U_sol, V_sol)
                        W_sol = T3f @ np.linalg.pinv(KR_UV)

                    if verify_decomp(U_sol, V_sol, W_sol):
                        return U_sol, V_sol, W_sol, True

        return None, None, None, False

    # Try to find lower rank
    U_sol, V_sol, W_sol, success = try_construct_rank(49)
    if success:
        best_rank = 49
        best_U, best_V, best_W = U_sol, V_sol, W_sol

    # If we have time, try even lower
    if time.time() < deadline - 10:
        for tr in range(48, max(35, best_rank - 10), -1):
            if time.time() > deadline - 8:
                break
            U_sol, V_sol, W_sol, success = try_construct_rank(tr)
            if success:
                best_rank = tr
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
