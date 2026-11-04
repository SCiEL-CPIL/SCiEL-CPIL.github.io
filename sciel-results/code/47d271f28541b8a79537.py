import time
import numpy as np

def construct():
    _t0 = time.time()
    deadline = _t0 + 140.0  # 10s safety margin

    n, m, p = 3, 4, 6  # C is n x p, A is n x m, B is m x p

    RANK = 66
    U = [[0] * RANK for _ in range(n * m)]  # (12, 66)
    V = [[0] * RANK for _ in range(m * p)]  # (24, 66)
    W = [[0] * RANK for _ in range(p * n)]  # (18, 66)

    col = 0

    # Indexing:
    #   A_{i,k}  -> U row  i * m + k       (i in [0,n), k in [0,m))
    #   B_{k,j}  -> V row  k * p + j       (k in [0,m), j in [0,p))
    #   C_{i,j}  -> W row  j * n + i       (j in [0,p), i in [0,n))  (trace convention)

    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """1x2 * 2x2 block, rank 4 (schoolbook).
        A[ar:ar+1, ac:ac+2] * B[br:br+2, bc:bc+2]
        -> C[cr, cc], C[cr, cc+1]
        """
        nonlocal col
        A0 = _idx_u(a_r, a_c)
        A1 = _idx_u(a_r, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C0 = _idx_w(c_c, c_r)
        C1 = _idx_w(c_c + 1, c_r)

        # M1 = A0 * B00 -> C0
        U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1
        col += 1
        # M2 = A1 * B10 -> C0
        U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1
        col += 1
        # M3 = A0 * B01 -> C1
        U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1
        col += 1
        # M4 = A1 * B11 -> C1
        U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1
        col += 1

    def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """2x2 * 2x2 block, rank 7 (Strassen).
        A[ar:ar+2, ac:ac+2] * B[br:br+2, bc:bc+2]
        -> C[cr:cr+2, cc:cc+2]
        """
        nonlocal col
        A00 = _idx_u(a_r, a_c)
        A01 = _idx_u(a_r, a_c + 1)
        A10 = _idx_u(a_r + 1, a_c)
        A11 = _idx_u(a_r + 1, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C00 = _idx_w(c_c, c_r)
        C01 = _idx_w(c_c + 1, c_r)
        C10 = _idx_w(c_c, c_r + 1)
        C11 = _idx_w(c_c + 1, c_r + 1)

        # M1 = (A00+A11)(B00+B11)
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1

        # M3 = A00(B01-B11)
        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1

        # M4 = A11(B10-B00)
        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1
        col += 1

    # ---- Build the 66-term decomposition ----
    # Partition: A rows 1+2, cols 2+2; B rows 2+2, cols 2+2+2
    # C[0, j*2:j*2+2] uses 1x2*2x2 (rank 4 each, x2 blocks = 8 per col pair)
    # C[1:3, j*2:j*2+2] uses 2x2*2x2 Strassen (rank 7 each, x2 blocks = 14 per col pair)
    # Total: 3*(8+14) = 66

    for j in range(3):  # C column block index (B col offset 2*j)
        bc = 2 * j  # B column start

        # --- Row 0 of C (A row 0, 1x2 blocks) ---
        _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
        _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)

        # --- Rows 1-2 of C (A rows 1:3, 2x2 blocks) ---
        _add_strassen(1, 0, 0, bc, 1, 2 * j)
        _add_strassen(1, 2, 2, bc, 1, 2 * j)

        if time.time() > deadline:
            break

    assert col == RANK, f"Expected {RANK} terms, got {col}"

    # ---- Try to reduce rank via local search ----
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Precompute expected tensor for fast verification
    T_expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T_expected[a, b, c] = 1.0

    def _verify_fast(Ua, Va, Wa):
        """Vectorized verification of the tensor decomposition."""
        T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
        if not np.allclose(T, T_expected, atol=1e-9):
            return False
        return True

    def _try_drop(Ua, Va, Wa, r):
        """Try dropping term r. Returns (success, new_U, new_V, new_W) or (False, None, None, None)."""
        Ua_new = np.delete(Ua, r, axis=1)
        Va_new = np.delete(Va, r, axis=1)
        Wa_new = np.delete(Wa, r, axis=1)
        if _verify_fast(Ua_new, Va_new, Wa_new):
            return True, Ua_new, Va_new, Wa_new
        return False, None, None, None

    # Try to drop terms one by one
    improved = False
    for r in range(RANK):
        if time.time() > deadline:
            break
        success, Ua_new, Va_new, Wa_new = _try_drop(U_np, V_np, W_np, r)
        if success:
            U_np = Ua_new
            V_np = Va_new
            W_np = Wa_new
            improved = True
            RANK -= 1
            r = -1  # restart from beginning since indices shifted
            if time.time() > deadline:
                break

    # ---- Additional refinement: try pairwise term merging via linear dependence ----
    # If two terms are linearly dependent (one is a scalar multiple of the other in all three matrices),
    # we can drop one. Check for exact duplicates or scalar multiples.
    if RANK > 2 and time.time() < deadline - 10:
        rank_cur = U_np.shape[1]
        for r in range(rank_cur):
            if time.time() > deadline - 5:
                break
            for s in range(r + 1, rank_cur):
                # Check if column r and s are proportional in U, V, W
                Ur = U_np[:, r]
                Us = U_np[:, s]
                Vr = V_np[:, r]
                Vs = V_np[:, s]
                Wr = W_np[:, r]
                Ws = W_np[:, s]

                # Find scale factor from U (use first non-zero element)
                nonzero_r = np.nonzero(Ur)[0]
                if len(nonzero_r) == 0:
                    continue
                idx = nonzero_r[0]
                if Ur[idx] == 0:
                    continue
                scale = Us[idx] / Ur[idx]

                # Check proportionality
                if (np.allclose(Ur * scale, Us, atol=1e-12) and
                    np.allclose(Vr * scale, Vs, atol=1e-12) and
                    np.allclose(Wr * scale, Ws, atol=1e-12)):
                    # Drop column s
                    U_np = np.delete(U_np, s, axis=1)
                    V_np = np.delete(V_np, s, axis=1)
                    W_np = np.delete(W_np, s, axis=1)
                    RANK -= 1
                    improved = True
                    s = r  # re-check same r with new s
                    break

    # ---- Final verification and cleanup ----
    if not _verify_fast(U_np, V_np, W_np):
        # Fallback: return the original 66-rank decomposition
        U = [[0] * 66 for _ in range(n * m)]
        V = [[0] * 66 for _ in range(m * p)]
        W = [[0] * 66 for _ in range(p * n)]
        col = 0

        for j in range(3):
            bc = 2 * j
            _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
            _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)
            _add_strassen(1, 0, 0, bc, 1, 2 * j)
            _add_strassen(1, 2, 2, bc, 1, 2 * j)

        return {"U": U, "V": V, "W": W}

    # Convert to lists and round to nearest half-integer
    U_out = U_np.tolist()
    V_out = V_np.tolist()
    W_out = W_np.tolist()

    U_out = [[round(x * 2) / 2 for x in row] for row in U_out]
    V_out = [[round(x * 2) / 2 for x in row] for row in V_out]
    W_out = [[round(x * 2) / 2 for x in row] for row in W_out]

    return {"U": U_out, "V": V_out, "W": W_out}
