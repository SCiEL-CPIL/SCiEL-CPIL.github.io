"""Matrix multiplication 2x4 * 4x5 tensor decomposition - rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # leave 20s margin before 150s kill

    # ---- Verification machinery (kept from current program) ----
    def verify(U, V, W, tol=1e-6):
        """Check if decomposition is valid using einsum."""
        r = U.shape[1]
        U_idx = U.reshape(n, m, r)
        V_idx = V.reshape(m, p, r)
        W_idx = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    # ---- Schoolbook fallback (rank 40) ----
    rank_school = n * m * p  # 40
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # ---- Structured Strassen-style decomposition (rank 36) ----
    # 2x4x5 = two 2x2x5 blocks (split j into j0 in {0,1} and j1 in {2,3}).
    # Each 2x2x5 = one 2x2x2 Strassen block (rank 7) + one 2x2x3 schoolbook (rank 6).
    # Total = 2 * (7 + 6) = 26? No: 2x2x3 schoolbook is 2*2*3=12. Let me recompute.
    # Actually 2x2x5: split k into k0 in {0,1} (2x2x2 Strassen, rank 7) and k1 in {2,3,4} (2x2x3 schoolbook, rank 12).
    # Per 2x2x5 block: 7 + 12 = 19. Two blocks: 38. Hmm, that's not 36.
    #
    # The measured 36 came from: each 2x2x5 = 2x2x2 Strassen (7) + 2x2x1 schoolbook (4)?? That only covers k=0.
    # Let me just build a correct, verified decomposition and trust verification.
    #
    # I'll build it by composing known sub-decompositions and verify. If the
    # structured one verifies, use it; otherwise fall back to schoolbook.

    def strassen_2x2x2():
        """Return (U,V,W) of shape (4,7),(4,7),(4,7) for 2x2x2 matmul tensor (all ones)."""
        # Strassen: M1=(a00+a11)(b00+b11), M2=(a10+a11)b00, M3=a00(b01-b11),
        # M4=a11(b10-b00), M5=(a00+a01)b11, M6=(a10-a00)(b00+b01), M7=(a01-a11)(b10+b11)
        # C00 = M1+M3-M5-M7 ; C01 = M4+M5 ; C10 = M2+M4 ; C11 = M1-M2+M6+M7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # index U[i*2+j], V[j*2+k], W[k*2+i]
        # M1: (a00+a11)(b00+b11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1   # contributes to C00
        # M2: (a10+a11)b00
        U[1, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1   # contributes to C10
        # M3: a00(b01-b11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1   # contributes to C00
        # M4: a11(b10-b00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1  # contributes to C01 and C10
        # M5: (a00+a01)b11
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1  # C00 -1, C01 +1
        # M6: (a10-a00)(b00+b01)
        U[1, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[1, 5] = 1   # contributes to C11
        # M7: (a01-a11)(b10+b11)
        U[2, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[1, 6] = 1  # C00 -1, C11 +1
        return U, V, W

    def build_structured():
        """Build rank-36 structured decomposition via two 2x2x5 blocks."""
        # Global indices: i in {0,1} (rows), j in {0,1,2,3}, k in {0,1,2,3,4}
        # Block A: j in {0,1}; Block B: j in {2,3}
        # For each block, split k: k in {0,1} Strassen (7), k in {2,3,4} schoolbook (2*2*3=12)
        # Per block: 7 + 12 = 19; total 38.
        # To reach 36, use k in {0,1} Strassen (7) and k in {2,3,4} but with a
        # 2x2x3 block that can be done in rank 10? Not guaranteed. So we accept 38
        # if it verifies, which still beats 40.
        R = 38
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for block in range(2):
            joff = block * 2  # 0 or 2
            # --- Strassen part for k in {0,1} ---
            sU, sV, sW = strassen_2x2x2()
            for c in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + (joff + j), col] += sU[i * 2 + j, c]
                for j in range(2):
                    for k in range(2):
                        V[(joff + j) * p + k, col] += sV[j * 2 + k, c]
                for k in range(2):
                    for i in range(2):
                        W[k * n + i, col] += sW[k * 2 + i, c]
                col += 1
            # --- Schoolbook part for k in {2,3,4} ---
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U[i * m + (joff + j), col] = 1
                        V[(joff + j) * p + (k + 2), col] = 1
                        W[(k + 2) * n + i, col] = 1
                        col += 1
        return U, V, W, col

    # ---- Polish: round to half-integers and re-verify ----
    def polish(U, V, W):
        # The structured decomposition uses only integer entries, so rounding is a no-op.
        # Try rounding to nearest 0.5 and verify; keep if valid.
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        if verify(U2, V2, W2, tol=1e-4):
            return U2, V2, W2
        return U, V, W

    # ---- Choose best verified decomposition ----
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()

    # Try structured
    if time.time() < deadline:
        sU, sV, sW, srank = build_structured()
        # Trim trailing zero columns in case col < R
        sU = sU[:, :srank]; sV = sV[:, :srank]; sW = sW[:, :srank]
        if verify(sU, sV, sW, tol=1e-6) and srank < best_rank:
            sU, sV, sW = polish(sU, sV, sW)
            best_rank = srank
            best_U, best_V, best_W = sU, sV, sW

    # ---- Attempt to shave rank with a quick ALS polish on the best found ----
    # (kept from current program family; only if time remains)
    if time.time() < deadline - 10 and best_rank < rank_school:
        def als_decompose(target_rank, max_iters=200, tol=1e-10, seed=42):
            rng = np.random.RandomState(seed)
            U = rng.randn(n * m, target_rank) * 0.5
            V = rng.randn(m * p, target_rank) * 0.5
            W = rng.randn(p * n, target_rank) * 0.5
            prev_err = float('inf')
            for iteration in range(max_iters):
                if time.time() > deadline - 5:
                    break
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except Exception:
                            pass
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                        except Exception:
                            pass
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                        except Exception:
                            pass
                U_idx = U.reshape(n, m, target_rank)
                V_idx = V.reshape(m, p, target_rank)
                W_idx = W.reshape(p, n, target_rank)
                T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
                err = np.max(np.abs(T - 1.0))
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
            return U, V, W, prev_err

        # Try to beat current best_rank by one or two
        for target_rank in range(max(20, best_rank - 2), best_rank):
            if time.time() > deadline - 5:
                break
            for seed in range(10):
                if time.time() > deadline - 5:
                    break
                U, V, W, err = als_decompose(target_rank, max_iters=150, seed=seed)
                if err < 1e-6:
                    U, V, W = polish(U, V, W)
                    if verify(U, V, W, tol=1e-6):
                        best_rank = target_rank
                        best_U, best_V, best_W = U, V, W
                        break

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
