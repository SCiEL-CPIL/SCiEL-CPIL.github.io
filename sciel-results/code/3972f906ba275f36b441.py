import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 145

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

    def solve_2x2x3_rank8(max_time=None):
        """
        Find a rank-8 decomposition for the 2x2 * 2x3 multiplication tensor.
        Uses structured initialization inspired by Strassen + schoolbook hybrid.
        """
        if max_time is None:
            max_time = deadline - 25

        T8 = np.zeros((4, 6, 6))
        monomials = [
            (0, 0, 0), (1, 3, 0),
            (0, 1, 1), (1, 4, 1),
            (0, 2, 2), (1, 5, 2),
            (2, 0, 3), (3, 3, 3),
            (2, 1, 4), (3, 4, 4),
            (2, 2, 5), (3, 5, 5),
        ]
        for (ai, bi, ci) in monomials:
            T8[ai, bi, ci] = 1.0

        r8 = 8
        best_loss = float('inf')
        best_U8, best_V8, best_W8 = None, None, None

        # Structured initializations
        def make_structured_init(seed):
            np.random.seed(seed)
            # Start from a perturbed Strassen-like structure
            U8 = np.zeros((4, r8))
            V8 = np.zeros((6, r8))
            W8 = np.zeros((6, r8))
            # First 7 columns: Strassen-like for 2x2x2 part (cols 0,1)
            # Col 0: (a00+a11)(b00+b11) -> c00+c11
            U8[0,0]=1; U8[3,0]=1; V8[0,0]=1; V8[4,0]=1; W8[0,0]=1; W8[4,0]=1
            # Col 1: (a10+a11)b00 -> c10-c11
            U8[2,1]=1; U8[3,1]=1; V8[0,1]=1; W8[3,1]=1; W8[4,1]=-1
            # Col 2: a00(b01-b11) -> c01+c11
            U8[0,2]=1; V8[1,2]=1; V8[5,2]=-1; W8[1,2]=1; W8[4,2]=1
            # Col 3: a11(b10-b00) -> c00+c10
            U8[3,3]=1; V8[3,3]=1; V8[0,3]=-1; W8[0,3]=1; W8[3,3]=1
            # Col 4: (a00+a01)b11 -> -c00+c01
            U8[0,4]=1; U8[1,4]=1; V8[5,4]=1; W8[0,4]=-1; W8[1,4]=1
            # Col 5: (a10-a00)(b00+b01) -> c11
            U8[2,5]=1; U8[0,5]=-1; V8[0,5]=1; V8[1,5]=1; W8[4,5]=1
            # Col 6: (a01-a11)(b10+b11) -> c00
            U8[1,6]=1; U8[3,6]=-1; V8[3,6]=1; V8[5,6]=1; W8[0,6]=1
            # Col 7: schoolbook for column 2
            U8[0,7]=1; V8[2,7]=1; W8[2,7]=1
            # Add small random perturbation
            U8 += np.random.randn(4, r8) * 0.1
            V8 += np.random.randn(6, r8) * 0.1
            W8 += np.random.randn(6, r8) * 0.1
            return U8, V8, W8

        def make_random_init(seed):
            np.random.seed(seed)
            U8 = np.random.randn(4, r8) * 0.3
            V8 = np.random.randn(6, r8) * 0.3
            W8 = np.random.randn(6, r8) * 0.3
            return U8, V8, W8

        def als_step(U8, V8, W8, T8, r8):
            # Solve for U8
            M_VW = np.zeros((r8, 36))
            for t in range(r8):
                for b in range(6):
                    for c in range(6):
                        M_VW[t, b * 6 + c] = V8[b, t] * W8[c, t]
            T8_r = T8.reshape(4, 36)
            try:
                M_VW_T = M_VW.T
                U8_new = T8_r @ M_VW_T @ np.linalg.inv(M_VW @ M_VW_T)
            except np.linalg.LinAlgError:
                U8_new = U8

            # Solve for V8
            M_UW = np.zeros((r8, 24))
            for t in range(r8):
                for a in range(4):
                    for c in range(6):
                        M_UW[t, a * 6 + c] = U8_new[a, t] * W8[c, t]
            T8_r2 = T8.reshape(6, 24)
            try:
                M_UW_T = M_UW.T
                V8_new = T8_r2 @ M_UW_T @ np.linalg.inv(M_UW @ M_UW_T)
            except np.linalg.LinAlgError:
                V8_new = V8

            # Solve for W8
            M_UV = np.zeros((r8, 24))
            for t in range(r8):
                for a in range(4):
                    for b in range(6):
                        M_UV[t, a * 6 + b] = U8_new[a, t] * V8_new[b, t]
            T8_r3 = T8.reshape(6, 24)
            try:
                M_UV_T = M_UV.T
                W8_new = T8_r3 @ M_UV_T @ np.linalg.inv(M_UV @ M_UV_T)
            except np.linalg.LinAlgError:
                W8_new = W8

            return U8_new, V8_new, W8_new

        # Try structured initializations first
        for restart in range(30):
            if time.time() > max_time:
                break

            if restart < 10:
                U8, V8, W8 = make_structured_init(1000 + restart * 100)
            else:
                U8, V8, W8 = make_random_init(5000 + restart * 7919)

            for step in range(200):
                if time.time() > max_time:
                    break

                U8, V8, W8 = als_step(U8, V8, W8, T8, r8)
                T_cur = np.einsum('at,bt,ct->abc', U8, V8, W8)
                loss = np.sum((T_cur - T8) ** 2)

                if loss < best_loss:
                    best_loss = loss
                    best_U8, best_V8, best_W8 = U8.copy(), V8.copy(), W8.copy()

                if loss < 1e-14:
                    break

            if best_loss < 1e-12:
                break

        if best_U8 is None:
            return None

        # Polish
        U8_final, V8_final, W8_final = best_U8.copy(), best_V8.copy(), best_W8.copy()
        for step in range(200):
            if time.time() > max_time:
                break
            U8_final, V8_final, W8_final = als_step(U8_final, V8_final, W8_final, T8, r8)
            T_cur = np.einsum('at,bt,ct->abc', U8_final, V8_final, W8_final)
            if np.sum((T_cur - T8) ** 2) < 1e-16:
                break

        T_check = np.einsum('at,bt,ct->abc', U8_final, V8_final, W8_final)
        if np.allclose(T_check, T8, atol=1e-6):
            # Try to snap to half-integers
            U8_rounded = np.round(U8_final * 2) / 2
            V8_rounded = np.round(V8_final * 2) / 2
            W8_rounded = np.round(W8_final * 2) / 2
            T_check2 = np.einsum('at,bt,ct->abc', U8_rounded, V8_rounded, W8_rounded)
            if np.allclose(T_check2, T8, atol=1e-6):
                return U8_rounded, V8_rounded, W8_rounded
            return U8_final, V8_final, W8_final
        return None

    def build_rank43():
        """
        5 Strassen blocks (for k=0..9) + a rank-8 block for k=10,11,12.
        """
        if time.time() > deadline - 20:
            return None

        R = 5 * 7 + 8  # 43
        U = np.zeros((nm, R), dtype=float)
        V = np.zeros((mp, R), dtype=float)
        W = np.zeros((pn, R), dtype=float)
        col = 0
        for k0 in range(0, 10, 2):
            col += build_strassen_block(k0, U, V, W, col)

        result = solve_2x2x3_rank8()
        if result is None:
            return None

        U8, V8, W8 = result
        r8 = 8

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

    # Build rank-46 solution first as guaranteed fallback
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
    if time.time() < deadline - 25:
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
        U46, V46, W46 = build_rank46()
        if verify(U46, V46, W46):
            return {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}
        else:
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
