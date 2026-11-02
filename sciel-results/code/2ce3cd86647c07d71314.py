import time
import numpy as np
import itertools

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit

    # Build the known rank-56 Strassen solution
    rank_56 = 56
    U56 = [[0] * rank_56 for _ in range(n * m)]
    V56 = [[0] * rank_56 for _ in range(m * p)]
    W56 = [[0] * rank_56 for _ in range(p * n)]

    u_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_vecs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            for li in range(4):
                if u_vecs[s][li] != 0:
                    U56[li][col] = u_vecs[s][li]
            v_indices = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                if v_vecs[s][li] != 0:
                    V56[v_indices[li]][col] = v_vecs[s][li]
            w_indices = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                if w_vecs[s][li] != 0:
                    W56[w_indices[li]][col] = w_vecs[s][li]

    # Build target tensor once
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[a, b, c] = 1.0

    def _verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        T_recon = np.einsum('as,bs,cs->abc', Ua, Va, Wa)
        return np.allclose(T_recon, T, atol=1e-8)

    # Verify the rank-56 solution
    if not _verify(U56, V56, W56):
        return {"U": U56, "V": V56, "W": W56}

    best_result = {"U": U56, "V": V56, "W": W56}
    best_rank = 56

    def _als_rank(r, U_init, V_init, W_init, max_iter=80, tol=1e-10):
        """ALS with pre-computed Khatri-Rao matrices for speed."""
        Ua = np.array(U_init, dtype=float).copy()
        Va = np.array(V_init, dtype=float).copy()
        Wa = np.array(W_init, dtype=float).copy()

        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None, None, None

            # Update U: for each a, T[a,:,:] = KR(V,W) @ U[a,:]
            KR_VW = np.einsum('bs,cs->bcs', Va, Wa).reshape(m*p*p*n, r)
            KR_VW_T = KR_VW.T
            KR_VW_T_KR_VW = KR_VW_T @ KR_VW
            try:
                KR_VW_pinv = np.linalg.pinv(KR_VW_T_KR_VW)
            except:
                KR_VW_pinv = np.linalg.pinv(KR_VW)
            for a in range(n*m):
                T_a = T[a].reshape(-1)
                Ua[a, :] = KR_VW_pinv @ KR_VW.T @ T_a

            # Update V: for each b, T[:,b,:] = KR(U,W) @ V[b,:]
            KR_UW = np.einsum('as,cs->acs', Ua, Wa).reshape(n*m*p*n, r)
            KR_UW_T = KR_UW.T
            KR_UW_T_KR_UW = KR_UW_T @ KR_UW
            try:
                KR_UW_pinv = np.linalg.pinv(KR_UW_T_KR_UW)
            except:
                KR_UW_pinv = np.linalg.pinv(KR_UW)
            for b in range(m*p):
                T_b = T[:, b, :].reshape(-1)
                Va[b, :] = KR_UW_pinv @ KR_UW.T @ T_b

            # Update W: for each c, T[:,:,c] = KR(U,V) @ W[c,:]
            KR_UV = np.einsum('as,bs->abs', Ua, Va).reshape(n*m*m*p, r)
            KR_UV_T = KR_UV.T
            KR_UV_T_KR_UV = KR_UV_T @ KR_UV
            try:
                KR_UV_pinv = np.linalg.pinv(KR_UV_T_KR_UV)
            except:
                KR_UV_pinv = np.linalg.pinv(KR_UV)
            for c in range(p*n):
                T_c = T[:, :, c].reshape(-1)
                Wa[c, :] = KR_UV_pinv @ KR_UV.T @ T_c

            # Check convergence
            T_recon = np.einsum('as,bs,cs->abc', Ua, Va, Wa)
            err = np.linalg.norm(T_recon - T)
            if err < tol:
                return Ua, Va, Wa

        T_recon = np.einsum('as,bs,cs->abc', Ua, Va, Wa)
        err = np.linalg.norm(T_recon - T)
        if err < 1e-8:
            return Ua, Va, Wa
        return None, None, None

    def _try_rank(r, max_iter=60):
        """Try to find rank-r decomposition by removing one column at a time from rank-56."""
        U56a = np.array(U56, dtype=float)
        V56a = np.array(V56, dtype=float)
        W56a = np.array(W56, dtype=float)

        # Try removing each single column
        for col_to_remove in range(56):
            if time.time() > deadline - 10:
                break

            mask = [i for i in range(56) if i != col_to_remove]
            U_init = U56a[:, mask]
            V_init = V56a[:, mask]
            W_init = W56a[:, mask]

            Ua, Va, Wa = _als_rank(r, U_init, V_init, W_init, max_iter=max_iter)
            if Ua is not None:
                return Ua, Va, Wa

        return None, None, None

    # Try rank 55
    if time.time() < deadline - 20:
        U55, V55, W55 = _try_rank(55, max_iter=40)
        if U55 is not None:
            # Try to snap to integer/half-integer values
            U55_snap = np.round(U55 * 2) / 2
            V55_snap = np.round(V55 * 2) / 2
            W55_snap = np.round(W55 * 2) / 2
            if _verify(U55_snap.tolist(), V55_snap.tolist(), W55_snap.tolist()):
                best_result = {"U": U55_snap.tolist(), "V": V55_snap.tolist(), "W": W55_snap.tolist()}
                best_rank = 55
            else:
                # Use the floating point version if it verifies
                if _verify(U55.tolist(), V55.tolist(), W55.tolist()):
                    best_result = {"U": U55.tolist(), "V": V55.tolist(), "W": W55.tolist()}
                    best_rank = 55

    # Try rank 54 if time permits
    if time.time() < deadline - 30 and best_rank == 56:
        U56a = np.array(U56, dtype=float)
        V56a = np.array(V56, dtype=float)
        W56a = np.array(W56, dtype=float)

        # Try removing pairs of columns
        for c1, c2 in itertools.combinations(range(56), 2):
            if time.time() > deadline - 10:
                break
            mask = [i for i in range(56) if i != c1 and i != c2]
            U_init = U56a[:, mask]
            V_init = V56a[:, mask]
            W_init = W56a[:, mask]

            Ua, Va, Wa = _als_rank(54, U_init, V_init, W_init, max_iter=30)
            if Ua is not None:
                U54_snap = np.round(Ua * 2) / 2
                V54_snap = np.round(Va * 2) / 2
                W54_snap = np.round(Wa * 2) / 2
                if _verify(U54_snap.tolist(), V54_snap.tolist(), W54_snap.tolist()):
                    best_result = {"U": U54_snap.tolist(), "V": V54_snap.tolist(), "W": W54_snap.tolist()}
                    best_rank = 54
                elif _verify(Ua.tolist(), Va.tolist(), Wa.tolist()):
                    best_result = {"U": Ua.tolist(), "V": Va.tolist(), "W": Wa.tolist()}
                    best_rank = 54
                break

    return best_result
# EVOLVE-BLOCK-END
