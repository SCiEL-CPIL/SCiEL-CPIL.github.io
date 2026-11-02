"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = 2, 4, 4
    rank_28 = 28

    # Build the rank-28 Strassen decomposition using clean block structure
    U = np.zeros((n * m, rank_28), dtype=float)  # (8, 28)
    V = np.zeros((m * p, rank_28), dtype=float)  # (16, 28)
    W = np.zeros((p * n, rank_28), dtype=float)  # (8, 28)

    # Strassen 2x2 terms: each is (u_coeffs, v_coeffs, w_coeffs)
    # Local 2x2 index mapping: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (A10+A11)B00
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        # M3 = A00(B01-B11)
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        # M4 = A11(B10-B00)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        # M5 = (A00+A01)B11
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

    # Global index mappings
    # A (2x4): row-major, index = i*m + j = i*4 + j
    u_map_L = [0, 1, 4, 5]   # A cols 0,1
    u_map_R = [2, 3, 6, 7]   # A cols 2,3

    # B (4x4): row-major, index = i*p + j = i*4 + j
    v_map_TL = [0, 1, 4, 5]   # B rows 0,1 cols 0,1
    v_map_TR = [2, 3, 6, 7]   # B rows 0,1 cols 2,3
    v_map_BL = [8, 9, 12, 13] # B rows 2,3 cols 0,1
    v_map_BR = [10, 11, 14, 15] # B rows 2,3 cols 2,3

    # C (2x4): W[k*n + i] = W[k*2 + i]
    # C_L (cols 0,1): local (i,j) -> W[j*2+i]
    w_map_L = [0, 2, 1, 3]
    # C_R (cols 2,3): local (i,j) -> W[(j+2)*2+i]
    w_map_R = [4, 6, 5, 7]

    col = 0
    # 1. A_L * B_TL -> C_L
    for u_t, v_t, w_t in strassen_terms:
        for idx, val in u_t:
            U[u_map_L[idx], col] += val
        for idx, val in v_t:
            V[v_map_TL[idx], col] += val
        for idx, val in w_t:
            W[w_map_L[idx], col] += val
        col += 1

    # 2. A_R * B_BL -> C_L
    for u_t, v_t, w_t in strassen_terms:
        for idx, val in u_t:
            U[u_map_R[idx], col] += val
        for idx, val in v_t:
            V[v_map_BL[idx], col] += val
        for idx, val in w_t:
            W[w_map_L[idx], col] += val
        col += 1

    # 3. A_L * B_TR -> C_R
    for u_t, v_t, w_t in strassen_terms:
        for idx, val in u_t:
            U[u_map_L[idx], col] += val
        for idx, val in v_t:
            V[v_map_TR[idx], col] += val
        for idx, val in w_t:
            W[w_map_R[idx], col] += val
        col += 1

    # 4. A_R * B_BR -> C_R
    for u_t, v_t, w_t in strassen_terms:
        for idx, val in u_t:
            U[u_map_R[idx], col] += val
        for idx, val in v_t:
            V[v_map_BR[idx], col] += val
        for idx, val in w_t:
            W[w_map_R[idx], col] += val
        col += 1

    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Verify that the decomposition exactly represents matrix multiplication."""
        rng = np.random.RandomState(42)
        for _ in range(5):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()
            C_computed = np.zeros((n, p))
            for i in range(n):
                for j in range(p):
                    s = 0.0
                    for r in range(U_mat.shape[1]):
                        u_val = np.dot(U_mat[:, r], A_flat)
                        v_val = np.dot(V_mat[:, r], B_flat)
                        s += u_val * v_val * W_mat[j * n + i, r]
                    C_computed[i, j] = s
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    if not verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a rank-27 decomposition
    # Approach: Check if any single term is redundant (the remaining 27 span the tensor)
    # and also try linear dependency analysis via SVD of the tensor vectors.

    T_full = np.zeros((n * m, m * p, p * n))
    for r in range(rank_28):
        T_full += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])

    best_rank = 28
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Quick check: for each term, see if removing it still gives the full tensor
    for elim_r in range(rank_28):
        if time.time() > deadline - 10:
            break
        U_sub = np.delete(U, elim_r, axis=1)
        V_sub = np.delete(V, elim_r, axis=1)
        W_sub = np.delete(W, elim_r, axis=1)

        T_sub = np.zeros((n * m, m * p, p * n))
        for r in range(27):
            T_sub += np.einsum('i,j,k->ijk', U_sub[:, r], V_sub[:, r], W_sub[:, r])

        if np.allclose(T_sub, T_full, atol=1e-10):
            best_rank = 27
            best_result = {
                "U": U_sub.tolist(),
                "V": V_sub.tolist(),
                "W": W_sub.tolist()
            }
            break

    # If no simple elimination works, try SVD-based null space search
    if best_rank == 28 and time.time() < deadline - 15:
        vecs = np.zeros((n * m * m * p * p * n, rank_28))
        for r in range(rank_28):
            vecs[:, r] = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r]).flatten()

        try:
            _, s_vals, Vt = np.linalg.svd(vecs, full_matrices=False)
            rank_mat = np.sum(s_vals > 1e-10)
            if rank_mat < 28:
                null_vecs = Vt[rank_mat:]
                for nv in null_vecs:
                    if time.time() > deadline - 10:
                        break
                    max_idx = np.argmax(np.abs(nv))
                    if abs(nv[max_idx]) > 1e-10:
                        U_sub = np.delete(U, max_idx, axis=1)
                        V_sub = np.delete(V, max_idx, axis=1)
                        W_sub = np.delete(W, max_idx, axis=1)
                        T_sub = np.zeros((n * m, m * p, p * n))
                        for r in range(27):
                            T_sub += np.einsum('i,j,k->ijk', U_sub[:, r], V_sub[:, r], W_sub[:, r])
                        if np.allclose(T_sub, T_full, atol=1e-10):
                            best_rank = 27
                            best_result = {
                                "U": U_sub.tolist(),
                                "V": V_sub.tolist(),
                                "W": W_sub.tolist()
                            }
                            break
        except Exception:
            pass

    return best_result
# EVOLVE-BLOCK-END
