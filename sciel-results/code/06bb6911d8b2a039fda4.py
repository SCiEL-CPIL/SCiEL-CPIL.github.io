import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26

    def A_idx(i, j):
        return i * m + j

    def B_idx(j, k):
        return j * p + k

    def C_idx(i, k):
        return k * n + i

    def verify(U, V, W, tol=1e-8):
        target_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_indices.append((i * m + j, j * p + k, k * n + i))
        target_arr = np.array(target_indices)
        a_idx = target_arr[:, 0]
        b_idx = target_arr[:, 1]
        c_idx = target_arr[:, 2]
        U_t = U[a_idx, :]
        V_t = V[b_idx, :]
        W_t = W[c_idx, :]
        prod = U_t * V_t * W_t
        sums = np.sum(prod, axis=1)
        return np.all(np.abs(sums - 1.0) < tol)

    def build_strassen_block(k0, U, V, W, col_offset):
        """Add a standard Strassen block for 2x2 * 2x2 (columns k0, k0+1)."""
        prods = [
            ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
            ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
            ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}),
            ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}),
            ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
            ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}),
            ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}),
        ]
        for t, (a_coeff, b_coeff) in enumerate(prods):
            col = col_offset + t
            for (i, j), c in a_coeff.items():
                U[A_idx(i, j), col] += c
            for (j, kl), c in b_coeff.items():
                V[B_idx(j, k0 + kl), col] += c
        c_combos = [
            (0, k0,     [(0, 1), (3, 1), (4, -1), (6, 1)]),
            (0, k0 + 1, [(2, 1), (4, 1)]),
            (1, k0,     [(1, 1), (3, 1)]),
            (1, k0 + 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
        ]
        for (i, k, terms) in c_combos:
            for (t, coeff) in terms:
                W[C_idx(i, k), col_offset + t] += coeff
        return 7

    def build_rank46():
        """6 Strassen blocks + 1 schoolbook block for k=12."""
        R = 46
        U = np.zeros((nm, R), dtype=float)
        V = np.zeros((mp, R), dtype=float)
        W = np.zeros((pn, R), dtype=float)
        col = 0
        for k0 in range(0, 12, 2):
            col += build_strassen_block(k0, U, V, W, col)
        # Schoolbook for k=12
        prods = [
            ((0, 0), (0, 12)),
            ((0, 1), (1, 12)),
            ((1, 0), (0, 12)),
            ((1, 1), (1, 12)),
        ]
        for t, ((i, j), (jj, kk)) in enumerate(prods):
            col_idx = col + t
            U[A_idx(i, j), col_idx] = 1.0
            V[B_idx(jj, kk), col_idx] = 1.0
        W[C_idx(0, 12), col + 0] = 1.0
        W[C_idx(0, 12), col + 1] = 1.0
        W[C_idx(1, 12), col + 2] = 1.0
        W[C_idx(1, 12), col + 3] = 1.0
        col += 4
        assert col == R, f"Expected rank {R}, got {col}"
        return U, V, W

    def solve_2x2x3_rank8():
        """
        Find a rank-8 decomposition for the 2x2 * 2x3 multiplication tensor.
        Returns (U8, V8, W8) of shapes (4,8), (6,8), (6,8) or None if not found.
        """
        # Target tensor T8 of shape (4, 6, 6)
        T8 = np.zeros((4, 6, 6))
        # A is 2x2: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B is 2x3: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
        # C is 2x3: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
        # Monomials: A[i,j] * B[j,k] -> C[i,k]
        monomials = [
            (0, 0, 0),  # A[0,0]*B[0,0] -> C[0,0]
            (1, 3, 0),  # A[0,1]*B[1,0] -> C[0,0]
            (0, 1, 1),  # A[0,0]*B[0,1] -> C[0,1]
            (1, 4, 1),  # A[0,1]*B[1,1] -> C[0,1]
            (0, 2, 2),  # A[0,0]*B[0,2] -> C[0,2]
            (1, 5, 2),  # A[0,1]*B[1,2] -> C[0,2]
            (2, 0, 3),  # A[1,0]*B[0,0] -> C[1,0]
            (3, 3, 3),  # A[1,1]*B[1,0] -> C[1,0]
            (2, 1, 4),  # A[1,0]*B[0,1] -> C[1,1]
            (3, 4, 4),  # A[1,1]*B[1,1] -> C[1,1]
            (2, 2, 5),  # A[1,0]*B[0,2] -> C[1,2]
            (3, 5, 5),  # A[1,1]*B[1,2] -> C[1,2]
        ]
        for (ai, bi, ci) in monomials:
            T8[ai, bi, ci] = 1.0

        r8 = 8
        best_loss = float('inf')
        best_U8, best_V8, best_W8 = None, None, None

        # Try multiple random restarts with ALS
        for restart in range(50):
            if time.time() > deadline - 30:
                break

            np.random.seed(12345 + restart * 7919)
            U8 = np.random.randn(4, r8) * 0.3
            V8 = np.random.randn(6, r8) * 0.3
            W8 = np.random.randn(6, r8) * 0.3

            for step in range(300):
                if time.time() > deadline - 35:
                    break

                # Solve for U8
                M_VW = np.zeros((r8, 36))
                for t in range(r8):
                    for b in range(6):
                        for c in range(6):
                            M_VW[t, b * 6 + c] = V8[b, t] * W8[c, t]
                T8_reshaped = T8.reshape(4, 36)
                try:
                    M_VW_T = M_VW.T
                    U8 = T8_reshaped @ M_VW_T @ np.linalg.inv(M_VW @ M_VW_T)
                except np.linalg.LinAlgError:
                    pass

                # Solve for V8
                M_UW = np.zeros((r8, 24))
                for t in range(r8):
                    for a in range(4):
                        for c in range(6):
                            M_UW[t, a * 6 + c] = U8[a, t] * W8[c, t]
                T8_reshaped2 = T8.reshape(6, 24)
                try:
                    M_UW_T = M_UW.T
                    V8 = T8_reshaped2 @ M_UW_T @ np.linalg.inv(M_UW @ M_UW_T)
                except np.linalg.LinAlgError:
                    pass

                # Solve for W8
                M_UV = np.zeros((r8, 24))
                for t in range(r8):
                    for a in range(4):
                        for b in range(6):
                            M_UV[t, a * 6 + b] = U8[a, t] * V8[b, t]
                T8_reshaped3 = T8.reshape(6, 24)
                try:
                    M_UV_T = M_UV.T
                    W8 = T8_reshaped3 @ M_UV_T @ np.linalg.inv(M_UV @ M_UV_T)
                except np.linalg.LinAlgError:
                    pass

                T_cur = np.einsum('at,bt,ct->abc', U8, V8, W8)
                loss = np.sum((T_cur - T8) ** 2)

                if loss < best_loss:
                    best_loss = loss
                    best_U8, best_V8, best_W8 = U8.copy(), V8.copy(), W8.copy()

                if loss < 1e-12:
                    break

            if best_loss < 1e-12:
                break

        if best_U8 is None:
            return None

        T_check = np.einsum('at,bt,ct->abc', best_U8, best_V8, best_W8)
        if np.allclose(T_check, T8, atol=1e-6):
            # Polish with more iterations
            U8_final, V8_final, W8_final = best_U8.copy(), best_V8.copy(), best_W8.copy()
            for step in range(100):
                M_VW = np.zeros((r8, 36))
                for t in range(r8):
                    for b in range(6):
                        for c in range(6):
                            M_VW[t, b * 6 + c] = V8_final[b, t] * W8_final[c, t]
                T8_reshaped = T8.reshape(4, 36)
                try:
                    U8_final = T8_reshaped @ M_VW.T @ np.linalg.inv(M_VW @ M_VW_T)
                except:
                    pass
                M_UW = np.zeros((r8, 24))
                for t in range(r8):
                    for a in range(4):
                        for c in range(6):
                            M_UW[t, a * 6 + c] = U8_final[a, t] * W8_final[c, t]
                T8_reshaped2 = T8.reshape(6, 24)
                try:
                    V8_final = T8_reshaped2 @ M_UW.T @ np.linalg.inv(M_UW @ M_UW_T)
                except:
                    pass
                M_UV = np.zeros((r8, 24))
                for t in range(r8):
                    for a in range(4):
                        for b in range(6):
                            M_UV[t, a * 6 + b] = U8_final[a, t] * V8_final[b, t]
                T8_reshaped3 = T8.reshape(6, 24)
                try:
                    W8_final = T8_reshaped3 @ M_UV.T @ np.linalg.inv(M_UV @ M_UV_T)
                except:
                    pass
                T_cur = np.einsum('at,bt,ct->abc', U8_final, V8_final, W8_final)
                if np.sum((T_cur - T8) ** 2) < 1e-14:
                    break

            # Try to snap to half-integers
            U8_rounded = np.round(U8_final * 2) / 2
            V8_rounded = np.round(V8_final * 2) / 2
            W8_rounded = np.round(W8_final * 2) / 2
            T_check2 = np.einsum('at,bt,ct->abc', U8_rounded, V8_rounded, W8_rounded)

            if np.allclose(T_check2, T8, atol=1e-6):
                return U8_rounded, V8_rounded, W8_rounded
            else:
                return U8_final, V8_final, W8_final

        return None

    def build_rank43():
        """
        5 Strassen blocks (for k=0..9) + a rank-8 block for k=10,11,12.
        """
        R = 5 * 7 + 8  # 43
        U = np.zeros((nm, R), dtype=float)
        V = np.zeros((mp, R), dtype=float)
        W = np.zeros((pn, R), dtype=float)
        col = 0
        for k0 in range(0, 10, 2):
            col += build_strassen_block(k0, U, V, W, col)

        # Solve for the 2x2x3 block
        result = solve_2x2x3_rank8()
        if result is None:
            return None

        U8, V8, W8 = result
        r8 = 8

        # Add to main matrices
        for t in range(r8):
            U[A_idx(0, 0), col + t] += U8[0, t]
            U[A_idx(0, 1), col + t] += U8[1, t]
            U[A_idx(1, 0), col + t] += U8[2, t]
            U[A_idx(1, 1), col + t] += U8[3, t]
            for bj in range(6):
                if bj < 3:
                    V[B_idx(0, 10 + bj), col + t] += V8[bj, t]
                else:
                    V[B_idx(1, 10 + bj - 3), col + t] += V8[bj, t]
            for cj in range(6):
                c_i = cj // 3
                c_k = 10 + cj % 3
                W[C_idx(c_i, c_k), col + t] += W8[cj, t]
        col += r8
        assert col == R, f"Expected {R}, got {col}"
        return U, V, W

    # Build rank-46 solution first as fallback
    U46, V46, W46 = build_rank46()
    best_U, best_V, best_W = U46.copy(), V46.copy(), W46.copy()
    best_rank = 46

    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook
        R0 = n * m * p
        U0 = np.zeros((nm, R0), dtype=float)
        V0 = np.zeros((mp, R0), dtype=float)
        W0 = np.zeros((pn, R0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col_idx = i * m * p + j * p + k
                    U0[i * m + j, col_idx] = 1.0
                    V0[j * p + k, col_idx] = 1.0
                    W0[k * n + i, col_idx] = 1.0
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    # Try rank-43
    result43 = build_rank43()
    if result43 is not None:
        U43, V43, W43 = result43
        if verify(U43, V43, W43):
            best_U, best_V, best_W = U43, V43, W43
            best_rank = 43

    # Snap to half-integers
    best_U = np.round(best_U * 2) / 2
    best_V = np.round(best_V * 2) / 2
    best_W = np.round(best_W * 2) / 2

    # Final verification
    if not verify(best_U, best_V, best_W):
        # If rounding broke it, use unrounded
        U46, V46, W46 = build_rank46()
        if verify(U46, V46, W46):
            return {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}
        else:
            # Ultimate fallback: schoolbook
            R0 = n * m * p
            U0 = np.zeros((nm, R0), dtype=float)
            V0 = np.zeros((mp, R0), dtype=float)
            W0 = np.zeros((pn, R0), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col_idx = i * m * p + j * p + k
                        U0[i * m + j, col_idx] = 1.0
                        V0[j * p + k, col_idx] = 1.0
                        W0[k * n + i, col_idx] = 1.0
            return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
