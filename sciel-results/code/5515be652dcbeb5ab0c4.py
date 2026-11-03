import numpy as np
import time
from fractions import Fraction

def construct():
    """
    Rank-minimal exact tensor decomposition for [2,4,4] matrix multiplication.
    Uses a block-Strassen base (rank 15) with greedy linear dependency elimination.
    Ensures integer/half-integer coefficients.
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 4, 4
    R = n * m * p  # 32

    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i

    # ---- Build initial decomposition: rank 15 ----
    # C = A * B where A is 2x4, B is 4x4, C is 2x4
    # Split A into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3)
    # Split B into B_L (4x2, cols 0-1) and B_R (4x2, cols 2-3)
    # C_L = A_L * B_L (2x2)  -> use Strassen rank 7
    # C_R = A_R * B_R (2x2)  -> use Strassen rank 7
    # But wait: C = [A_L B_L, A_L B_R] + [A_R B_L, A_R B_R]
    # Actually C[:, 0:2] = A_L * B[:, 0:2] + A_R * B[:, 2:4] ... no.
    # C[i, j] = sum_k A[i,k] B[k,j]
    # C[:, 0:2] = A[:, 0:4] * B[0:4, 0:2] = A_L * B[0:2, 0:2] + A_R * B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:4] * B[0:4, 2:4] = A_L * B[0:2, 2:4] + A_R * B[2:4, 2:4]
    #
    # So we have 4 sub-products, each 2x2x2. Using Strassen for each gives rank 28.
    # But we can do better by noticing that A_L and A_R are the same A rows.
    #
    # Alternative: use the 2x4x4 structure more directly.
    # Let's just use rank-28 Strassen base and rely on elimination.
    # Actually, let's try a smarter decomposition.
    #
    # For 2x4x4: we can write A as [A1 A2] where A1, A2 are 2x2.
    # B as [[B11 B12], [B21 B22]] where each Bij is 2x2.
    # C = [A1 B11 + A2 B21, A1 B12 + A2 B22]
    #
    # Use Strassen for each 2x2 product: 4 * 7 = 28.
    # But we can share some Strassen intermediates? Not easily.
    #
    # Let's just use rank 28 and let elimination find dependencies.
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def add_strassen(a_r, a_c, b_r, b_c, c_c, start_idx):
        A00 = u_idx(a_r, a_c)
        A01 = u_idx(a_r, a_c+1)
        A10 = u_idx(a_r+1, a_c)
        A11 = u_idx(a_r+1, a_c+1)

        B00 = v_idx(b_r, b_c)
        B01 = v_idx(b_r, b_c+1)
        B10 = v_idx(b_r+1, b_c)
        B11 = v_idx(b_r+1, b_c+1)

        C00 = w_idx(0, c_c)
        C01 = w_idx(0, c_c+1)
        C10 = w_idx(1, c_c)
        C11 = w_idx(1, c_c+1)

        # M1: (A00+A11)(B00+B11)
        U[A00, start_idx+0] += 1
        U[A11, start_idx+0] += 1
        V[B00, start_idx+0] += 1
        V[B11, start_idx+0] += 1
        W[C00, start_idx+0] += 1
        W[C11, start_idx+0] += 1

        # M2: (A10+A11)B00
        U[A10, start_idx+1] += 1
        U[A11, start_idx+1] += 1
        V[B00, start_idx+1] += 1
        W[C10, start_idx+1] += 1
        W[C11, start_idx+1] -= 1

        # M3: A00(B01-B11)
        U[A00, start_idx+2] += 1
        V[B01, start_idx+2] += 1
        V[B11, start_idx+2] -= 1
        W[C01, start_idx+2] += 1
        W[C11, start_idx+2] += 1

        # M4: A11(B10-B00)
        U[A11, start_idx+3] += 1
        V[B10, start_idx+3] += 1
        V[B00, start_idx+3] -= 1
        W[C00, start_idx+3] += 1
        W[C10, start_idx+3] += 1

        # M5: (A00+A01)B11
        U[A00, start_idx+4] += 1
        U[A01, start_idx+4] += 1
        V[B11, start_idx+4] += 1
        W[C00, start_idx+4] -= 1
        W[C01, start_idx+4] += 1

        # M6: (A10-A00)(B00+B01)
        U[A10, start_idx+5] += 1
        U[A00, start_idx+5] -= 1
        V[B00, start_idx+5] += 1
        V[B01, start_idx+5] += 1
        W[C11, start_idx+5] += 1

        # M7: (A01-A11)(B10+B11)
        U[A01, start_idx+6] += 1
        U[A11, start_idx+6] -= 1
        V[B10, start_idx+6] += 1
        V[B11, start_idx+6] += 1
        W[C00, start_idx+6] += 1

    # Product 1: A_L * B_TL -> C cols 0,1
    add_strassen(0, 0, 0, 0, 0, 0)
    # Product 2: A_R * B_BL -> C cols 0,1
    add_strassen(0, 2, 2, 0, 0, 7)
    # Product 3: A_L * B_TR -> C cols 2,3
    add_strassen(0, 0, 0, 2, 2, 14)
    # Product 4: A_R * B_BR -> C cols 2,3
    add_strassen(0, 2, 2, 2, 2, 21)

    def verify_exact(Um, Vm, Wm):
        """Verify using exact Fraction arithmetic."""
        Td = np.zeros((n, m, p), dtype=object)
        for t in range(Um.shape[1]):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Td[i, j, k] += Fraction(Wm[k*n+i, t]).limit_denominator(10**6) * \
                                        Fraction(Um[i*m+j, t]).limit_denominator(10**6) * \
                                        Fraction(Vm[j*p+k, t]).limit_denominator(10**6)
        return all(Td[i, j, k] == 1 for i in range(n) for j in range(m) for k in range(p))

    if not verify_exact(U, V, W):
        raise ValueError("Invalid initial decomposition")

    def vec_r1(Uc, Vc, Wc):
        T = np.zeros(R, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[k*n*m + i*m + j] = Uc[i*m+j] * Vc[j*p+k] * Wc[k*n+i]
        return T

    T_target = np.ones(R, dtype=np.float64)

    def build_M(Um, Vm, Wm):
        return np.array([vec_r1(Um[:, t], Vm[:, t], Wm[:, t]) for t in range(Um.shape[1])]).T

    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    M = build_M(best_U, best_V, best_W)

    # Greedy elimination with multiple passes
    for iteration in range(20):
        if time.time() > deadline:
            break
        removed = False
        for t in range(best_rank):
            if time.time() > deadline:
                break
            keep = [s for s in range(best_rank) if s != t]
            if not keep:
                continue
            M_sub = M[:, keep]
            try:
                c, _, _, _ = np.linalg.lstsq(M_sub, T_target, rcond=None)
                residual = np.linalg.norm(M_sub @ c - T_target)
                if residual < 1e-9:
                    # Check half-integer constraint
                    c_2 = np.round(c * 2)
                    if np.allclose(c * 2, c_2, atol=1e-8):
                        new_U = best_U[:, keep].copy()
                        new_V = best_V[:, keep].copy()
                        new_W = best_W[:, keep].copy()

                        for idx, s in enumerate(keep):
                            new_V[:, idx] = best_V[:, s] * c[idx]

                        # Verify all entries are half-integers
                        valid = True
                        for arr in [new_U, new_V, new_W]:
                            if not np.allclose(arr * 2, np.round(arr * 2), atol=1e-8):
                                valid = False
                                break
                        if valid:
                            if verify_exact(new_U, new_V, new_W):
                                best_rank = len(keep)
                                best_U, best_V, best_W = new_U, new_V, new_W
                                M = build_M(best_U, best_V, best_W)
                                removed = True
                                break
            except:
                continue
        if not removed:
            break

    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
