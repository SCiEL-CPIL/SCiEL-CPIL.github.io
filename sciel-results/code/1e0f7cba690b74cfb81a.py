"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms, then applies systematic rank reduction via rational linear
dependency detection.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

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

    def build_decomp(partition):
        """Build decomposition based on partition strategy."""
        U_t = []
        V_t = []
        W_t = []

        if partition == 0:
            # Strassen for A[0:2,0:2]*B[0:2,k] for each 2-block of k
            for k0 in [0, 2, 4, 6]:
                strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)
            # Direct terms for remaining
            for i in range(2):
                for k in range(p):
                    direct_term(i, 2, k, U_t, V_t, W_t)
            for j in range(2):
                for k in range(p):
                    direct_term(2, j, k, U_t, V_t, W_t)
            for k in range(p):
                direct_term(2, 2, k, U_t, V_t, W_t)
        elif partition == 1:
            # Alternative: use Strassen on different blocks
            # Try A[0:2, 0:2] * B[0:2, :] with Strassen, and handle row 2 separately
            for k0 in [0, 2, 4, 6]:
                strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)
            # Row 2 of A times all of B
            for j in range(m):
                for k in range(p):
                    direct_term(2, j, k, U_t, V_t, W_t)
            # Column 2 of A (rows 0,1) times B
            for i in range(2):
                for k in range(p):
                    direct_term(i, 2, k, U_t, V_t, W_t)
        elif partition == 2:
            # Schoolbook fallback
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        direct_term(i, j, k, U_t, V_t, W_t)
        elif partition == 3:
            # Try using Strassen on A[0:2,0:2]*B[0:2,0:2] only,
            # and direct for everything else
            strassen_decomp(0, 0, 0, 0, U_t, V_t, W_t)
            # A[0:2,0:2]*B[0:2,2:8] direct
            for i in range(2):
                for j in range(2):
                    for k in range(2, p):
                        direct_term(i, j, k, U_t, V_t, W_t)
            # A[0:2,2]*B[2,:] direct
            for i in range(2):
                for k in range(p):
                    direct_term(i, 2, k, U_t, V_t, W_t)
            # A[2,:]*B[:,0:2] direct
            for j in range(2):
                for k in range(2):
                    direct_term(2, j, k, U_t, V_t, W_t)
            # A[2,2]*B[2,2:8] direct
            for k in range(2, p):
                direct_term(2, 2, k, U_t, V_t, W_t)
            # A[2,0:2]*B[0:2,2:8] direct
            for j in range(2):
                for k in range(2, p):
                    direct_term(2, j, k, U_t, V_t, W_t)
            # A[0:2,0:2]*B[2,0:2] direct
            for i in range(2):
                for k in range(2):
                    direct_term(i, 2, k, U_t, V_t, W_t)

        rank = len(W_t)

        # Build the factor matrices
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        for idx in range(rank):
            for (i, j), coeff in U_t[idx]:
                U[i * m + j][idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V[j * p + k][idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W[k * n + i][idx] = coeff

        return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float), rank

    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def term_vector(Ua, Va, Wa, r):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i*m+j)*p + k] = Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
        return vec

    def try_eliminate_rational(r, Ua, Va, Wa, rank_cur, max_denom=6):
        """Try to express term r as rational combination of other terms."""
        target = term_vector(Ua, Va, Wa, r)
        others = [s for s in range(rank_cur) if s != r]
        if not others:
            return None

        # Build design matrix
        M = np.array([term_vector(Ua, Va, Wa, s) for s in others])

        # First try integer solution
        try:
            coeffs, _, _, _ = np.linalg.lstsq(M, target, rcond=None)
        except Exception:
            return None

        # Check if integer solution works
        coeffs_int = np.round(coeffs)
        if np.all(np.abs(coeffs - coeffs_int) < 1e-6):
            coeff_map = {s: int(coeffs_int[i]) for i, s in enumerate(others) if abs(coeffs_int[i]) > 1e-9}
            if _verify_elimination(r, coeff_map, others, Ua, Va, Wa, target):
                return coeff_map

        # Try rational solutions with small denominators
        for denom in range(2, max_denom + 1):
            # Scale target and try to find integer solution for scaled problem
            # If target = sum(c_s * v_s) where c_s = a_s/denom
            # Then denom * target = sum(a_s * v_s)
            scaled_target = denom * target
            try:
                coeffs_scaled, _, _, _ = np.linalg.lstsq(M, scaled_target, rcond=None)
            except Exception:
                continue

            coeffs_int_scaled = np.round(coeffs_scaled)
            if np.all(np.abs(coeffs_scaled - coeffs_int_scaled) < 1e-6):
                coeff_map = {}
                for i, s in enumerate(others):
                    val = coeffs_int_scaled[i] / denom
                    if abs(val) > 1e-9:
                        coeff_map[s] = val
                if _verify_elimination(r, coeff_map, others, Ua, Va, Wa, target):
                    return coeff_map

        # Try subset-based approach with top overlapping terms
        overlaps = np.abs(M @ target)
        order = np.argsort(-overlaps)
        for top_k in [4, 6, 8, 10]:
            if top_k >= len(others):
                break
            top = [others[i] for i in order[:top_k]]
            Mt = np.array([term_vector(Ua, Va, Wa, s) for s in top])
            try:
                ct, _, _, _ = np.linalg.lstsq(Mt, target, rcond=None)
            except Exception:
                continue

            # Try integer
            ct_int = np.round(ct)
            if np.all(np.abs(ct - ct_int) < 1e-6):
                coeff_map = {s: int(ct_int[i]) for i, s in enumerate(top) if abs(ct_int[i]) > 1e-9}
                if _verify_elimination(r, coeff_map, others, Ua, Va, Wa, target):
                    return coeff_map

            # Try rational
            for denom in range(2, max_denom + 1):
                scaled_t = denom * target
                try:
                    ct_s, _, _, _ = np.linalg.lstsq(Mt, scaled_t, rcond=None)
                except Exception:
                    continue
                ct_s_int = np.round(ct_s)
                if np.all(np.abs(ct_s - ct_s_int) < 1e-6):
                    coeff_map = {}
                    for i, s in enumerate(top):
                        val = ct_s_int[i] / denom
                        if abs(val) > 1e-9:
                            coeff_map[s] = val
                    if _verify_elimination(r, coeff_map, others, Ua, Va, Wa, target):
                        return coeff_map

        return None

    def _verify_elimination(r, coeff_map, others, Ua, Va, Wa, target):
        """Verify that the elimination is exact."""
        recon = np.zeros(n * m * p)
        for s, c in coeff_map.items():
            recon += c * term_vector(Ua, Va, Wa, s)
        return np.allclose(recon, target, atol=1e-9)

    def apply_elimination(r, coeff_map, Ua, Va, Wa, rank_cur):
        """Apply elimination of term r using coeff_map."""
        keep = [s for s in range(rank_cur) if s != r]
        new_rank = len(keep)
        new_U = np.zeros((n * m, new_rank))
        new_V = np.zeros((m * p, new_rank))
        new_W = np.zeros((p * n, new_rank))
        for new_idx, s in enumerate(keep):
            scale = 1.0 + coeff_map.get(s, 0.0)
            if abs(scale) < 1e-9:
                continue
            new_U[:, new_idx] = Ua[:, s] * scale
            new_V[:, new_idx] = Va[:, s] * scale
            new_W[:, new_idx] = Wa[:, s] * scale

        # Remove zero columns
        nonzero_mask = np.any(np.abs(new_U) > 1e-9, axis=0) | \
                       np.any(np.abs(new_V) > 1e-9, axis=0) | \
                       np.any(np.abs(new_W) > 1e-9, axis=0)
        if not np.all(nonzero_mask):
            new_U = new_U[:, nonzero_mask]
            new_V = new_V[:, nonzero_mask]
            new_W = new_W[:, nonzero_mask]

        return new_U, new_V, new_W

    def reduce_rank(Ua, Va, Wa, rank_cur, max_time):
        """Greedy rank reduction with rational coefficient search."""
        progress = True
        while progress and (time.time() - start_time) < max_time:
            progress = False
            for r in range(rank_cur):
                if (time.time() - start_time) > max_time:
                    break
                res = try_eliminate_rational(r, Ua, Va, Wa, rank_cur, max_denom=6)
                if res is not None:
                    new_U, new_V, new_W = apply_elimination(r, res, Ua, Va, Wa, rank_cur)
                    if verify_decomp(new_U, new_V, new_W):
                        Ua, Va, Wa = new_U, new_V, new_W
                        rank_cur = Ua.shape[1]
                        progress = True
                        break
        return Ua, Va, Wa, rank_cur

    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                # Try to find a nice rational representation
                found = False
                for denom in [1, 2, 3, 4, 6, 8, 12]:
                    val_scaled = val * denom
                    if abs(val_scaled - round(val_scaled)) < 1e-8:
                        nice_val = round(val_scaled) / denom
                        if abs(nice_val - round(nice_val)) < 1e-9:
                            nice_val = int(round(nice_val))
                        nice_row.append(nice_val)
                        found = True
                        break
                if not found:
                    # Try larger denominators
                    for denom in [16, 24, 32, 48]:
                        val_scaled = val * denom
                        if abs(val_scaled - round(val_scaled)) < 1e-8:
                            nice_val = round(val_scaled) / denom
                            if abs(nice_val - round(nice_val)) < 1e-9:
                                nice_val = int(round(nice_val))
                            nice_row.append(nice_val)
                            found = True
                            break
                if not found:
                    nice_row.append(val)
            result.append(nice_row)
        return result

    # Try multiple partition strategies and keep the best
    best_result = None
    best_rank = float('inf')

    partitions = [0, 1, 3, 2]  # Try Strassen-based first, schoolbook last

    for partition in partitions:
        if (time.time() - start_time) > DEADLINE - 15.0:
            break

        Ua, Va, Wa, rank = build_decomp(partition)

        if not verify_decomp(Ua, Va, Wa):
            continue

        # Apply rank reduction
        Ua, Va, Wa, rank = reduce_rank(Ua, Va, Wa, rank, DEADLINE - 10.0)

        if rank < best_rank:
            best_rank = rank
            best_result = (Ua.copy(), Va.copy(), Wa.copy(), rank)

        if (time.time() - start_time) > DEADLINE - 10.0:
            break

    if best_result is None:
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

    Ua, Va, Wa, rank = best_result
    return {"U": to_nice_list(Ua), "V": to_nice_list(Va), "W": to_nice_list(Wa)}
