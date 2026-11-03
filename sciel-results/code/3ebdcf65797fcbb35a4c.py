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

    def _verify(Ua, Va, Wa):
        """Check if the decomposition exactly computes matrix multiplication."""
        # For each (i,j) in C, check: sum_k A[i,k]*B[k,j] = sum_r U[i*m+k, r]*V[k*p+j, r]*W[j*n+i, r]
        for i in range(n):
            for j in range(p):
                expected = 0.0
                for k in range(m):
                    expected += 1.0  # A[i,k]*B[k,j] = 1*1 = 1 for the tensor
                # Actually, the tensor T[i*m+k, k*p+j, j*n+i] = 1
                # So sum_r U[i*m+k, r]*V[k*p+j, r]*W[j*n+i, r] should = 1 for each (i,k,j)
                # Wait, the tensor is T[a_idx, b_idx, c_idx] = 1 iff a_idx = i*m+k, b_idx = k*p+j, c_idx = j*n+i
                # So for fixed (i,j), sum_k T[i*m+k, k*p+j, j*n+i] = m = 4
                # And sum_r (sum_k U[i*m+k,r]*V[k*p+j,r]) * W[j*n+i,r] should = 4
                lhs = 0.0
                for k in range(m):
                    for r in range(len(Ua[0])):
                        lhs += Ua[i * m + k, r] * Va[k * p + j, r] * Wa[j * n + i, r]
                if abs(lhs - m) > 1e-9:
                    return False
        return True

    def _try_drop(Ua, Va, Wa, r):
        """Try dropping term r. Returns (success, new_U, new_V, new_W) or (False, None, None, None)."""
        if not _verify(Ua, Va, Wa):
            return False, None, None, None
        # Try removing term r
        Ua_new = np.delete(Ua, r, axis=1)
        Va_new = np.delete(Va, r, axis=1)
        Wa_new = np.delete(Wa, r, axis=1)
        if _verify(Ua_new, Va_new, Wa_new):
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

    if improved:
        # Convert back to lists
        U = U_np.tolist()
        V = V_np.tolist()
        W = W_np.tolist()
        # Round to nearest integer or half-integer
        U = [[round(x * 2) / 2 for x in row] for row in U]
        V = [[round(x * 2) / 2 for x in row] for row in V]
        W = [[round(x * 2) / 2 for x in row] for row in W]

    return {"U": U, "V": V, "W": W}
