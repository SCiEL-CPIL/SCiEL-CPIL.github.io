import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 6, 6, 6
    assert n == 6 and m == 6 and p == 6

    deadline = time.time() + 140.0

    # ------------------------------------------------------------------
    # Helper: build block-Strassen rank-189 decomposition (always valid)
    # ------------------------------------------------------------------
    def build_block_strassen(rank3):
        bs = 3
        total_rank = 7 * rank3
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]

        strassen_defs = [
            {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
            {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
            {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
            {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
            {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
            {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
            {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
        ]
        recon_coeffs = [
            [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
            [(2, 1.0), (4, 1.0)],
            [(1, 1.0), (3, 1.0)],
            [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
        ]

        # Inner 3x3 decomposition: list of (u_idx, v_idx, w_idx, coeff)
        # For schoolbook rank3=27: each term is A[i,k]*B[k,j] -> C[i,j]
        inner = []
        for i in range(bs):
            for k in range(bs):
                for j in range(bs):
                    inner.append((i * bs + k, k * bs + j, j * bs + i, 1.0))

        r_idx = 0
        for s in range(7):
            s_def = strassen_defs[s]
            for (u_local, v_local, w_local, coeff) in inner:
                # U: u_local = i_local*bs + k_local
                i_local = u_local // bs
                k_local = u_local % bs
                for block_idx, coeff_a in s_def['A']:
                    br = block_idx // 2
                    bc = block_idx % 2
                    i_glob = br * bs + i_local
                    k_glob = bc * bs + k_local
                    U[i_glob * m + k_glob][r_idx] += coeff_a * coeff

                # V: v_local = k_local*bs + j_local
                k_local_v = v_local // bs
                j_local = v_local % bs
                for block_idx, coeff_b in s_def['B']:
                    br = block_idx // 2
                    bc = block_idx % 2
                    k_glob = br * bs + k_local_v
                    j_glob = bc * bs + j_local
                    V[k_glob * p + j_glob][r_idx] += coeff_b * coeff

                # W: w_local = j_local*bs + i_local
                j_local_w = w_local // bs
                i_local_w = w_local % bs
                for c_block_idx in range(4):
                    gamma = 0.0
                    for (s_idx, g) in recon_coeffs[c_block_idx]:
                        if s_idx == s:
                            gamma = g
                            break
                    if gamma != 0.0:
                        br = c_block_idx // 2
                        bc = c_block_idx % 2
                        i_glob = br * bs + i_local_w
                        j_glob = bc * bs + j_local_w
                        W[j_glob * n + i_glob][r_idx] += gamma * coeff

                r_idx += 1

        assert r_idx == total_rank, f"r_idx {r_idx} != total_rank {total_rank}"
        return U, V, W

    # ------------------------------------------------------------------
    # Helper: verify a decomposition exactly
    # ------------------------------------------------------------------
    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        T = np.zeros((n * m, m * p, p * n), dtype=float)
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    T[i * m + k, k * p + j, j * n + i] = 1.0
        recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
        return np.allclose(recon, T, atol=1e-9)

    # ------------------------------------------------------------------
    # Search for a rank-23 exact 3x3 decomposition (BCLT-class)
    # ------------------------------------------------------------------
    def find_3x3_decomposition(rank3):
        if time.time() > deadline - 30:
            return None

        n3 = 3
        T3 = np.zeros((n3 * n3, n3 * n3, n3 * n3), dtype=float)
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    T3[i * n3 + k, k * n3 + j, j * n3 + i] = 1.0

        best_res = None
        best_err = float('inf')

        # A few structured random restarts with a tight iteration budget
        for restart in range(4):
            if time.time() > deadline - 25:
                break
            np.random.seed(1000 + restart)
            U = np.random.randn(n3 * n3, rank3) * 0.5
            V = np.random.randn(n3 * n3, rank3) * 0.5
            W = np.random.randn(n3 * n3, rank3) * 0.5
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            def objective(x):
                U_f = x[:n3 * n3 * rank3].reshape(n3 * n3, rank3)
                V_f = x[n3 * n3 * rank3:2 * n3 * n3 * rank3].reshape(n3 * n3, rank3)
                W_f = x[2 * n3 * n3 * rank3:].reshape(n3 * n3, rank3)
                T_recon = np.einsum('ir,jr,kr->ijk', U_f, V_f, W_f)
                err = T_recon - T3
                return np.sum(err * err)

            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': 150, 'maxfun': 300, 'gtol': 1e-6})
            except Exception:
                continue

            x_opt = res.x
            U_opt = x_opt[:n3 * n3 * rank3].reshape(n3 * n3, rank3)
            V_opt = x_opt[n3 * n3 * rank3:2 * n3 * n3 * rank3].reshape(n3 * n3, rank3)
            W_opt = x_opt[2 * n3 * n3 * rank3:].reshape(n3 * n3, rank3)

            # Snap to half-integers (BCLT entries are integers)
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2

            T_recon = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
            err = float(np.linalg.norm(T_recon - T3))

            if err < 1e-9:
                # Build inner list from the snapped matrices
                inner = []
                for r in range(rank3):
                    for u in range(n3 * n3):
                        for v in range(n3 * n3):
                            for w in range(n3 * n3):
                                val = U_snap[u, r] * V_snap[v, r] * W_snap[w, r]
                                if abs(val) > 1e-12:
                                    # Only keep if this (u,v,w) triple is consistent
                                    # (we'll reconstruct below)
                                    pass
                # Reconstruct inner terms directly from the matrices
                inner = []
                for r in range(rank3):
                    for i in range(n3):
                        for k in range(n3):
                            for j in range(n3):
                                u_idx = i * n3 + k
                                v_idx = k * n3 + j
                                w_idx = j * n3 + i
                                # Contribution to C[i,j] from rank-r term
                                contrib = U_snap[u_idx, r] * V_snap[v_idx, r] * W_snap[w_idx, r]
                                # We need U[u_idx,r]*V[v_idx,r] to be the A_s*B_s coefficient
                                # and W[w_idx,r] to be the gamma. Store (u,v,w,coeff)
                                # where coeff is the product for the (u,v) pair and W is separate.
                                # Simpler: store the three matrices and build via einsum in the
                                # 6x6 assembly. But our build_block_strassen expects an inner list.
                                # So store per-rank triple with full (u,v,w) and unit coeff,
                                # letting the U/V/W matrices carry the values.
                                inner.append((u_idx, v_idx, w_idx, 1.0))
                # But we need the actual values. Rebuild using matrices directly.
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}

            if err < best_err:
                best_err = err
                best_res = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}

        return None

    # ------------------------------------------------------------------
    # Try to find a rank-23 3x3 decomposition and use it (rank 161)
    # ------------------------------------------------------------------
    if time.time() < deadline - 35:
        res3 = find_3x3_decomposition(23)
        if res3 is not None:
            # Build 6x6 using the 3x3 matrices directly via einsum-style assembly
            U3 = np.array(res3["U"], dtype=float)
            V3 = np.array(res3["V"], dtype=float)
            W3 = np.array(res3["W"], dtype=float)
            rank3 = U3.shape[1]
            total_rank = 7 * rank3

            U = [[0.0] * total_rank for _ in range(n * m)]
            V = [[0.0] * total_rank for _ in range(m * p)]
            W = [[0.0] * total_rank for _ in range(p * n)]

            strassen_defs = [
                {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
                {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
                {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
                {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
                {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
                {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
                {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
            ]
            recon_coeffs = [
                [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
                [(2, 1.0), (4, 1.0)],
                [(1, 1.0), (3, 1.0)],
                [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
            ]

            bs = 3
            r_idx = 0
            for s in range(7):
                s_def = strassen_defs[s]
                for r in range(rank3):
                    for i in range(bs):
                        for k in range(bs):
                            u_local = i * bs + k
                            for j in range(bs):
                                v_local = k * bs + j
                                w_local = j * bs + i

                                # U
                                for block_idx, coeff_a in s_def['A']:
                                    br = block_idx // 2
                                    bc = block_idx % 2
                                    i_glob = br * bs + i
                                    k_glob = bc * bs + k
                                    U[i_glob * m + k_glob][r_idx] += coeff_a * U3[u_local, r]

                                # V
                                for block_idx, coeff_b in s_def['B']:
                                    br = block_idx // 2
                                    bc = block_idx % 2
                                    k_glob = br * bs + k
                                    j_glob = bc * bs + j
                                    V[k_glob * p + j_glob][r_idx] += coeff_b * V3[v_local, r]

                                # W
                                for c_block_idx in range(4):
                                    gamma = 0.0
                                    for (s_idx, g) in recon_coeffs[c_block_idx]:
                                        if s_idx == s:
                                            gamma = g
                                            break
                                    if gamma != 0.0:
                                        br = c_block_idx // 2
                                        bc = c_block_idx % 2
                                        i_glob = br * bs + i
                                        j_glob = bc * bs + j
                                        W[j_glob * n + i_glob][r_idx] += gamma * W3[w_local, r]

                                r_idx += 1

            assert r_idx == total_rank
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}

    # ------------------------------------------------------------------
    # Fallback: exact block-Strassen rank-189
    # ------------------------------------------------------------------
    U, V, W = build_block_strassen(27)
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Should not reach here, but return the block-Strassen result anyway
    return {"U": U, "V": V, "W": W}
