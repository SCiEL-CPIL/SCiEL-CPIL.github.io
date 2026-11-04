import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 118

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

    # Rank 46 construction:
    # Split B (4x7) into three 4x2 blocks and one 4x1 vector
    # Block 1: columns 0,1 -> A[2x4] * B[4x2]
    #   Split A into A[0:2,0:2] and A[0:2,2:4]
    #   A[0:2,0:2] * B[0:2,0:2] uses Strassen rank 7
    #   A[0:2,2:4] * B[2:4,0:2] uses Strassen rank 7
    #   Total 14 for this block
    # Block 2: columns 2,3 -> same structure, 14 terms
    # Block 3: columns 4,5 -> same structure, 14 terms
    # Block 4: column 6 -> A[2x4] * B[4x1]
    #   Split into A[0:2,0:2]*B[0:2,0] + A[0:2,2:4]*B[2:4,0]
    #   Each is 2x2x1, rank 4 schoolbook
    #   Total 8 for this block
    # Grand total: 14+14+14+8 = 50? Wait, that's 50.
    # Let me recheck: 2x2x1 tensor has rank 4 (all 4 entries independent).
    # So 4+4=8 for the last column.
    # 14*3 + 8 = 50. That's the same as before.
    
    # Let me think differently. The issue is that 2x2x1 has rank 4, not 2.
    # Actually wait: for 2x2x1, the tensor is T[i,j,k] where k=0 only.
    # T[i,j,0] = delta_ij? No, it's the standard matrix-vector product.
    # C_i = sum_j A_ij * B_j
    # This is a bilinear map with 4 independent terms, so rank 4.
    
    # Alternative: Can we use a smarter decomposition for the last column?
    # Or can we merge some Strassen terms across blocks?
    
    # Let me try rank 48 instead: maybe we can save 2 terms by merging.
    # Actually, let's just implement the rank-50 and try to improve.
    
    def get_rank50():
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

        # Three 2x4x2 blocks, each using two 2x2x2 Strassen decompositions
        for k_start in [0, 2, 4]:
            # A[:, 0:2] * B[0:2, k_start:k_start+2]
            add_strassen(0, 0, k_start)
            # A[:, 2:4] * B[2:4, k_start:k_start+2]
            add_strassen(0, 2, k_start)

        # Last column: A[2x4] * B[4x1] at column 6
        # Schoolbook: 4 terms for A[0:2,0:2]*B[0:2,6], 4 terms for A[0:2,2:4]*B[2:4,6]
        for j in range(m):
            U[j, col] = 1.0
            V[j * p + 6, col] = 1.0
            W[6 * n + 0, col] = 1.0
            col += 1
        for j in range(m):
            U[4 + j, col] = 1.0
            V[(j + 2) * p + 6, col] = 1.0
            W[6 * n + 1, col] = 1.0
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    U50, V50, W50 = get_rank50()
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # Try to reduce rank using ALS with various initializations
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

    def als_one_shot(U_init, V_init, W_init, max_iters=200, tol=1e-14):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        for it in range(max_iters):
            if time.time() > deadline - 3:
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

    def polish(U, V, W, max_iters=300, tol=1e-15):
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            KR_VW = khatri_rao(V, W)
            U = T1f @ np.linalg.pinv(KR_VW)
            KR_WU = khatri_rao(W, U)
            V = T2f @ np.linalg.pinv(KR_WU)
            KR_UV = khatri_rao(U, V)
            W = T3f @ np.linalg.pinv(KR_UV)
        return U, V, W

    rng = np.random.default_rng(42)

    # Try ranks from 49 down to 40
    for target_rank in range(49, 39, -1):
        if time.time() > deadline - 10:
            break

        n_restarts = max(3, min(8, int((deadline - 10 - time.time()) / 2.0)))

        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break

            if restart == 0:
                U_init = U50[:, :target_rank] + rng.normal(0, 0.01, (n*m, target_rank))
                V_init = V50[:, :target_rank] + rng.normal(0, 0.01, (m*p, target_rank))
                W_init = W50[:, :target_rank] + rng.normal(0, 0.01, (p*n, target_rank))
            elif restart == 1:
                U_init = rng.normal(0, 1, (n * m, target_rank))
                V_init = rng.normal(0, 1, (m * p, target_rank))
                W_init = rng.normal(0, 1, (p * n, target_rank))
            elif restart == 2:
                U_init = rng.choice([-1, 0, 1], (n * m, target_rank)).astype(float)
                V_init = rng.choice([-1, 0, 1], (m * p, target_rank)).astype(float)
                W_init = rng.choice([-1, 0, 1], (p * n, target_rank)).astype(float)
            else:
                U_init = 0.5 * U50[:, :target_rank] + 0.5 * rng.normal(0, 1, (n*m, target_rank))
                V_init = 0.5 * V50[:, :target_rank] + 0.5 * rng.normal(0, 1, (m*p, target_rank))
                W_init = 0.5 * W50[:, :target_rank] + 0.5 * rng.normal(0, 1, (p*n, target_rank))

            U_sol, V_sol, W_sol = als_one_shot(U_init, V_init, W_init, max_iters=150)

            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)

            if residual < 1e-8:
                U_sol, V_sol, W_sol = polish(U_sol, V_sol, W_sol)
                if verify_decomp(U_sol, V_sol, W_sol):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
