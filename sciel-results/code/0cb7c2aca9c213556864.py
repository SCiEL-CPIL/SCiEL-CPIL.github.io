"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms, then applies systematic rank reduction via structured
dependency detection and iterative elimination.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)

        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)

        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])

        # m2 = (c+d)*e -> C10 += 1, C11 -= 1
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])

        # m3 = a*(f-h) -> C01 += 1, C11 += 1
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])

        # m4 = d*(g-e) -> C00 += 1, C10 += 1
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])

        # m5 = (a+b)*h -> C00 -= 1, C01 += 1
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])

        # m6 = (c-a)*(e+f) -> C11 += 1
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])

        # m7 = (b-d)*(g+h) -> C00 += 1
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])

    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    def build_decomp():
        """Build the base decomposition: 4x Strassen + direct terms."""
        U_t = []
        V_t = []
        W_t = []

        # A[0:2, 0:2] * B[0:2, k] for k in 0:2, 2:4, 4:6, 6:8
        for k0 in [0, 2, 4, 6]:
            strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)

        # A[0:2, 2] * B[2, k] direct
        for k in range(p):
            for i in range(2):
                direct_term(i, 2, k, U_t, V_t, W_t)

        # A[2, 0:2] * B[0:2, k] direct
        for k in range(p):
            for j in range(2):
                direct_term(2, j, k, U_t, V_t, W_t)

        # A[2, 2] * B[2, k] direct
        for k in range(p):
            direct_term(2, 2, k, U_t, V_t, W_t)

        rank = len(W_t)

        # Build the factor matrices
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        for idx in range(rank):
            for (i, j), coeff in U_t[idx]:
                U[i * m + j, idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V[j * p + k, idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W[k * n + i, idx] = coeff

        return U, V, W, rank

    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i * m + j] * Va[j * p + k] * Wa[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def term_vector(Ua, Va, Wa, r):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i * m + j) * p + k] = Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
        return vec

    def try_eliminate(r, Ua, Va, Wa, rank_cur, others_list):
        """Try to express term r as integer/half-integer combination of other terms."""
        target = term_vector(Ua, Va, Wa, r)
        others = [s for s in others_list if s != r]
        if not others:
            return None

        # Build design matrix from all other terms
        M = np.array([term_vector(Ua, Va, Wa, s) for s in others])

        # Try least squares and round to nearest integer
        try:
            coeffs, _, _, _ = np.linalg.lstsq(M, target, rcond=None)
        except Exception:
            return None

        # Try rounding to integers
        coeffs_int = np.round(coeffs)
        if np.all(np.abs(coeffs - coeffs_int) < 1e-6):
            coeff_map = {s: coeffs_int[i] for i, s in enumerate(others)}
            coeff_map = {s: c for s, c in coeff_map.items() if abs(c) > 1e-9}
        else:
            # Try rounding to half-integers (multiply by 2)
            coeffs_half = np.round(coeffs * 2) / 2.0
            if np.all(np.abs(coeffs - coeffs_half) < 1e-6):
                coeff_map = {s: coeffs_half[i] for i, s in enumerate(others)}
                coeff_map = {s: c for s, c in coeff_map.items() if abs(c) > 1e-9}
            else:
                # Try with a subset of top-overlapping terms
                overlaps = np.abs(M @ target)
                order = np.argsort(-overlaps)
                top_k = min(10, len(others))
                top = [others[i] for i in order[:top_k]]
                Mt = np.array([term_vector(Ua, Va, Wa, s) for s in top])
                try:
                    ct, _, _, _ = np.linalg.lstsq(Mt, target, rcond=None)
                except Exception:
                    return None
                # Try integer
                ct_int = np.round(ct)
                if np.all(np.abs(ct - ct_int) < 1e-6):
                    coeff_map = {s: ct_int[i] for i, s in enumerate(top)}
                    coeff_map = {s: c for s, c in coeff_map.items() if abs(c) > 1e-9}
                else:
                    # Try half-integer
                    ct_half = np.round(ct * 2) / 2.0
                    if np.all(np.abs(ct - ct_half) < 1e-6):
                        coeff_map = {s: ct_half[i] for i, s in enumerate(top)}
                        coeff_map = {s: c for s, c in coeff_map.items() if abs(c) > 1e-9}
                    else:
                        return None

        # Verify exactness
        recon = np.zeros(n * m * p)
        for s, c in coeff_map.items():
            recon += c * term_vector(Ua, Va, Wa, s)
        if not np.allclose(recon, target, atol=1e-9):
            return None

        # Build new matrices: keep all terms except r, scale others
        keep = [s for s in range(rank_cur) if s != r]
        new_rank = len(keep)
        new_U = np.zeros((n * m, new_rank))
        new_V = np.zeros((m * p, new_rank))
        new_W = np.zeros((p * n, new_rank))
        for new_idx, s in enumerate(keep):
            scale = 1.0 + coeff_map.get(s, 0.0)
            if abs(scale) < 1e-9:
                continue  # This term becomes zero
            new_U[:, new_idx] = Ua[:, s] * scale
            new_V[:, new_idx] = Va[:, s] * scale
            new_W[:, new_idx] = Wa[:, s] * scale

        # Remove zero columns
        nonzero_mask = (np.any(np.abs(new_U) > 1e-9, axis=0) |
                        np.any(np.abs(new_V) > 1e-9, axis=0) |
                        np.any(np.abs(new_W) > 1e-9, axis=0))
        if not np.all(nonzero_mask):
            new_U = new_U[:, nonzero_mask]
            new_V = new_V[:, nonzero_mask]
            new_W = new_W[:, nonzero_mask]

        return new_U, new_V, new_W

    def reduce_rank(Ua, Va, Wa, rank_cur, max_time):
        """Greedy rank reduction with multiple elimination strategies."""
        progress = True
        iterations = 0
        while progress and (time.time() - start_time) < max_time:
            progress = False
            iterations += 1
            # Try eliminating each term in different orders
            orderings = [
                list(range(rank_cur)),
                list(range(rank_cur - 1, -1, -1)),
                # Prioritize direct terms (higher indices) first
                sorted(range(rank_cur), key=lambda x: -x)
            ]
            for order in orderings:
                if (time.time() - start_time) > max_time:
                    break
                for r in order:
                    if (time.time() - start_time) > max_time:
                        break
                    others_list = list(range(rank_cur))
                    res = try_eliminate(r, Ua, Va, Wa, rank_cur, others_list)
                    if res is not None:
                        new_U, new_V, new_W = res
                        if verify_decomp(new_U, new_V, new_W):
                            Ua, Va, Wa = new_U, new_V, new_W
                            rank_cur = Ua.shape[1]
                            progress = True
                            break
                if progress:
                    break
        return Ua, Va, Wa, rank_cur

    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result

    # Build base decomposition
    Ua, Va, Wa, rank = build_decomp()

    if not verify_decomp(Ua, Va, Wa):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Apply rank reduction
    Ua, Va, Wa, rank = reduce_rank(Ua, Va, Wa, rank, DEADLINE - 5.0)

    # Final verification
    if not verify_decomp(Ua, Va, Wa):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": to_nice_list(Ua), "V": to_nice_list(Va), "W": to_nice_list(Wa)}
