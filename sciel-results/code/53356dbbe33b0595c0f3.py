"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms, then applies systematic rank reduction via integer linear
dependency detection.
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

    def build_decomp(partition):
        """Build decomposition based on partition strategy."""
        U_t = []
        V_t = []
        W_t = []

        if partition == 0:
            # Original: 2x2 Strassen for A[0:2,0:2]*B[0:2,k], direct for rest
            for k0 in [0, 2, 4, 6]:
                strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)
            for k0 in [0, 2, 4, 6]:
                for i in range(2):
                    for k in range(k0, k0 + 2):
                        direct_term(i, 2, k, U_t, V_t, W_t)
                for j in range(2):
                    for k in range(k0, k0 + 2):
                        direct_term(2, j, k, U_t, V_t, W_t)
                for k in range(k0, k0 + 2):
                    direct_term(2, 2, k, U_t, V_t, W_t)
        elif partition == 1:
            # Try different: 2x2 Strassen for A[0:2,0:2]*B[0:2,k] and A[2,0:2]*B[0:2,k]
            # Actually, let's try using Strassen on different 2x2 blocks
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
        elif partition == 2:
            # Use 1x3 and 3x1 blocks with direct, but try to find structure
            # Actually, let's try a completely different approach:
            # Use the fact that we can write the 3x3 as [2x2; 1x3] and [3x2; 1x1]
            # and find a decomposition that shares factors
            # For now, fall back to schoolbook if this doesn't help
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        direct_term(i, j, k, U_t, V_t, W_t)

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

    def try_eliminate(r, Ua, Va, Wa, rank_cur):
        """Try to express term r as integer combination of other terms."""
        target = term_vector(Ua, Va, Wa, r)
        others = [s for s in range(rank_cur) if s != r]
        if not others:
            return None

        # Build design matrix
        M = np.array([term_vector(Ua, Va, Wa, s) for s in others])

        # Try least squares and round
        try:
            coeffs, _, _, _ = np.linalg.lstsq(M, target, rcond=None)
        except Exception:
            return None

        coeffs_int = np.round(coeffs)
        if not np.all(np.abs(coeffs - coeffs_int) < 1e-6):
            # Try with a subset of top-overlapping terms
            overlaps = np.abs(M @ target)
            order = np.argsort(-overlaps)
            top_k = min(8, len(others))
            top = [others[i] for i in order[:top_k]]
            Mt = np.array([term_vector(Ua, Va, Wa, s) for s in top])
            try:
                ct, _, _, _ = np.linalg.lstsq(Mt, target, rcond=None)
            except Exception:
                return None
            ct_int = np.round(ct)
            if not np.all(np.abs(ct - ct_int) < 1e-6):
                return None
            coeff_map = {s: 0.0 for s in others}
            for s, cv in zip(top, ct_int):
                coeff_map[s] = cv
        else:
            coeff_map = {s: coeffs_int[i] for i, s in enumerate(others)}
            coeff_map = {s: c for s, c in coeff_map.items() if abs(c) > 1e-9}

        # Verify exactness
        recon = np.zeros(n * m * p)
        for s, c in coeff_map.items():
            recon += c * term_vector(Ua, Va, Wa, s)
        if not np.allclose(recon, target, atol=1e-9):
            return None

        # Build new matrices
        keep = [s for s in range(rank_cur) if s != r]
        new_rank = len(keep)
        new_U = np.zeros((n * m, new_rank))
        new_V = np.zeros((m * p, new_rank))
        new_W = np.zeros((p * n, new_rank))
        for new_idx, s in enumerate(keep):
            scale = 1.0 + coeff_map.get(s, 0.0)
            if abs(scale) < 1e-9:
                # This term becomes zero, skip it
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
        """Rank reduction via null space analysis."""
        progress = True
        while progress and (time.time() - start_time) < max_time:
            progress = False
            # Precompute term vectors
            R = rank_cur
            T = np.zeros((R, n * m * p))
            for r in range(R):
                T[r, :] = term_vector(Ua, Va, Wa, r)
            
            # Find null space of T
            # We want integer vector c such that T c = 0
            # Use SVD to find null space
            try:
                U_svd, s, Vt = np.linalg.svd(T, full_matrices=False)
            except Exception:
                break
            
            # Determine rank
            tol = 1e-10 * max(s) if len(s) > 0 else 1e-10
            rank_svd = np.sum(s > tol)
            
            if rank_svd < R:
                # Null space is spanned by rows of Vt[rank_svd:]
                null_space = Vt[rank_svd:]
                # Try to find an integer vector in the null space
                # We can iterate over the null space basis vectors and check for rational/integer relations
                # Or, simpler: try to find a short integer vector in the null space.
                # Since entries are small integers, we can try small integer combinations of the null space basis.
                
                # A robust way: check if any column of Vt[rank_svd:] is "close" to an integer vector (after scaling)
                # Or better: use the fact that T has integer entries. The null space is a rational subspace.
                # We can find a basis for the null space with rational entries using Gaussian elimination on T.
                
                # Let's try a simpler heuristic first: 
                # Check if any single term is a linear combination of others with small integer coefficients.
                # This is what the old code did, but it was slow. 
                # With the null space basis, we can see if there are simple dependencies.
                
                # If null space dimension is 1, the vector is unique (up to scale).
                if R - rank_svd == 1:
                    v = null_space[0]
                    # Scale v to have integer entries
                    non_zero = v[np.abs(v) > 1e-9]
                    if len(non_zero) > 0:
                        min_val = np.min(np.abs(non_zero))
                        v_scaled = v / min_val
                        v_int = np.round(v_scaled)
                        if np.allclose(v_scaled, v_int, atol=1e-6):
                            c = v_int
                            elim_idx = -1
                            for i in range(R-1, -1, -1):
                                if abs(c[i]) > 1e-9:
                                    elim_idx = i
                                    break
                            
                            if elim_idx != -1:
                                c_r = c[elim_idx]
                                if abs(c_r) > 1e-9:
                                    keep = [s for s in range(R) if s != elim_idx]
                                    new_rank = len(keep)
                                    new_U = np.zeros((n * m, new_rank))
                                    new_V = np.zeros((m * p, new_rank))
                                    new_W = np.zeros((p * n, new_rank))
                                    
                                    for new_idx, s in enumerate(keep):
                                        scale = 1.0 + c[s] / c_r
                                        new_U[:, new_idx] = Ua[:, s] * scale
                                        new_V[:, new_idx] = Va[:, s] * scale
                                        new_W[:, new_idx] = Wa[:, s] * scale
                                    
                                    nonzero_mask = np.any(np.abs(new_U) > 1e-9, axis=0) | \
                                                   np.any(np.abs(new_V) > 1e-9, axis=0) | \
                                                   np.any(np.abs(new_W) > 1e-9, axis=0)
                                    if not np.all(nonzero_mask):
                                        new_U = new_U[:, nonzero_mask]
                                        new_V = new_V[:, nonzero_mask]
                                        new_W = new_W[:, nonzero_mask]
                                        
                                    if verify_decomp(new_U, new_V, new_W):
                                        Ua, Va, Wa = new_U, new_V, new_W
                                        rank_cur = Ua.shape[1]
                                        progress = True
                                        break
                else:
                    # Higher dimensional null space.
                    # We can try to find a short integer vector in the null space.
                    # For now, skip or try a simple heuristic.
                    pass
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

    # Try multiple partition strategies and keep the best
    best_result = None
    best_rank = float('inf')

    for partition in [0, 1]:
        if (time.time() - start_time) > DEADLINE - 10.0:
            break

        Ua, Va, Wa, rank = build_decomp(partition)

        if not verify_decomp(Ua, Va, Wa):
            continue

        # Apply rank reduction
        Ua, Va, Wa, rank = reduce_rank(Ua, Va, Wa, rank, DEADLINE - 5.0)

        if rank < best_rank:
            best_rank = rank
            best_result = (Ua, Va, Wa, rank)

        if (time.time() - start_time) > DEADLINE - 5.0:
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
