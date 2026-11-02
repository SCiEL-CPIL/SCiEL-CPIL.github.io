import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Strassen decomposition for 2x2x2
    strassen_decomp = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # M4
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),         # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})          # M7
    ]

    def _build_strassen_3x4x8():
        """Block-Strassen decomposition: rank 88. Split n=3 into 2+1."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j1 in range(2):
            for l in range(4):
                for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                    for li, coeff in u_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        U[li_i * m + (2 * j1 + li_j)][col] += coeff
                    for li, coeff in v_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += coeff
                    for li, coeff in w_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        W[(2 * l + li_j) * n + li_i][col] += coeff
                    col += 1
            for k in range(8):
                for j2 in range(2):
                    U[2 * m + (2 * j1 + j2)][col] = 1
                    V[(2 * j1 + j2) * p + k][col] = 1
                    W[k * n + 2][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _build_strassen_alt_split():
        """Alternative split: n=3 into 1+2. Top schoolbook, bottom Strassen."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j in range(4):
            for k in range(8):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k * n][col] = 1
                col += 1
        for j1 in range(2):
            for l in range(4):
                for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                    for li, coeff in u_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        U[(1 + li_i) * m + (2 * j1 + li_j)][col] += coeff
                    for li, coeff in v_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += coeff
                    for li, coeff in w_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        W[(2 * l + li_j) * n + (1 + li_i)][col] += coeff
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _verify(U, V, W, tol=1e-9):
        """Verify exact tensor decomposition (trace convention)."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _verify_half_integer(U, V, W):
        """Check that all entries are integers or half-integers."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True

    def _prune_columns(U, V, W, tol=1e-9):
        """
        Structural refinement: greedily remove linearly dependent / redundant
        rank terms. A column (rank term) is redundant if it can be expressed as
        a linear combination of the others while preserving the tensor identity.
        We test by checking, for each candidate removal, whether the resulting
        reduced decomposition still verifies. We prioritize removing columns
        that are linear combinations of others (small residual after projection).
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]

        # Precompute the "signature" of each column: the vector of values it
        # contributes to each (i,j,k) entry.
        # signature[r] = U[:,r] * V[:,r] * W[:,r] reshaped over (n*m, m*p, p*n)
        # Actually the contribution to entry (i,j,k) is U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
        # We'll build a matrix S where S[(i,j,k), r] = that product.
        idx = np.array([(i, j, k) for i in range(n) for j in range(m) for k in range(p)])
        S = np.zeros((n * m * p, rank), dtype=float)
        for t, (i, j, k) in enumerate(idx):
            S[t, :] = U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i]

        # The identity requires sum over r of S[t, r] == 1 for all t.
        # A column r is removable if there exist coefficients c_s (s != r) such that
        # sum_s c_s S[:, s] == S[:, r]  (i.e., S[:, r] in span of others).
        # Greedy: repeatedly find a column closest to being in the span of the rest,
        # remove it if residual small enough, else stop.
        keep = list(range(rank))
        changed = True
        while changed and len(keep) > 1:
            changed = False
            S_keep = S[:, keep]
            # For each column, compute residual of projecting onto others.
            best_r = -1
            best_res = tol * 10
            for pos, r in enumerate(keep):
                others = [q for q in keep if q != r]
                if len(others) == 0:
                    continue
                S_others = S[:, others]
                # Solve least squares: find c s.t. S_others @ c ≈ S[:, r]
                # Use lstsq.
                try:
                    c, _, _, _ = np.linalg.lstsq(S_others, S[:, r], rcond=None)
                    residual = np.linalg.norm(S[:, r] - S_others @ c)
                except np.linalg.LinAlgError:
                    continue
                if residual < best_res:
                    best_res = residual
                    best_r = r
            if best_r >= 0 and best_res < tol * 10:
                # Remove best_r
                keep = [q for q in keep if q != best_r]
                changed = True
            else:
                break

        U_red = [[row[r] for r in keep] for row in U]
        V_red = [[row[r] for r in keep] for row in V]
        W_red = [[row[r] for r in keep] for row in W]
        return U_red, V_red, W_red

    def _prune_by_rank(U, V, W, tol=1e-9):
        """
        Alternative refinement: find the maximum set of columns whose signatures
        span the same space as all columns, by incremental linear-independence
        test. This gives the minimal rank (dimension of the span of signatures).
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]

        idx = np.array([(i, j, k) for i in range(n) for j in range(m) for k in range(p)])
        S = np.zeros((n * m * p, rank), dtype=float)
        for t, (i, j, k) in enumerate(idx):
            S[t, :] = U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i]

        # Greedy basis: keep columns that add to the column-space of S.
        # This finds a minimal generating set (a basis) of the column space.
        keep = []
        basis = []
        for r in range(rank):
            col = S[:, r]
            # Check if col is in span of basis
            if not basis:
                keep.append(r)
                basis.append(col.copy())
                continue
            B = np.stack(basis, axis=1)
            c, _, _, _ = np.linalg.lstsq(B, col, rcond=None)
            residual = np.linalg.norm(col - B @ c)
            if residual > tol * 10:
                keep.append(r)
                basis.append(col.copy())
        return keep

    # Build candidate decompositions
    best_result = None
    best_rank = float('inf')

    candidates = []

    # Candidate 1: original Strassen split (n: 2+1)
    c1 = _build_strassen_3x4x8()
    if _verify(c1['U'], c1['V'], c1['W']) and _verify_half_integer(c1['U'], c1['V'], c1['W']):
        candidates.append((88, c1))

    # Candidate 2: alternative split (n: 1+2)
    if time.time() < deadline:
        c2 = _build_strassen_alt_split()
        if _verify(c2['U'], c2['V'], c2['W']) and _verify_half_integer(c2['U'], c2['V'], c2['W']):
            candidates.append((88, c2))

    # Try to reduce rank via linear-dependence pruning on each candidate
    for cand_rank, cand in list(candidates):
        if time.time() > deadline - 5:
            break
        # Prune by rank (basis extraction)
        try:
            keep = _prune_by_rank(cand['U'], cand['V'], cand['W'])
            if len(keep) < cand_rank:
                U_red = [[row[r] for r in keep] for row in cand['U']]
                V_red = [[row[r] for r in keep] for row in cand['V']]
                W_red = [[row[r] for r in keep] for row in cand['W']]
                if _verify(U_red, V_red, W_red) and _verify_half_integer(U_red, V_red, W_red):
                    candidates.append((len(keep), {"U": U_red, "V": V_red, "W": W_red}))
        except Exception:
            pass

    # Select best candidate
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_rank, best_result = candidates[0]
    else:
        # Fallback
        best_result = _build_strassen_3x4x8()
        best_rank = 88

    return best_result
# EVOLVE-BLOCK-END
