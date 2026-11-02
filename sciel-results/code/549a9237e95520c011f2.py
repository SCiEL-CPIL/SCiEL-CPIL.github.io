"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8

    # Standard 7-term Strassen decomposition for 2x2x2
    # Each term: (u_entries, v_entries, w_entries)
    # u_entries: (i, j, val) for A[i,j]
    # v_entries: (j, k, val) for B[j,k]
    # w_entries: (k, i, val) for C[k,i] (trace convention W[k*n+i])
    s = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],         [(0,1,1),(1,1,-1)]),
        ([(0,0,1)],         [(0,1,1),(1,1,-1)],[(1,0,1),(1,1,1)]),
        ([(1,1,1)],         [(1,0,1),(0,0,-1)],[(0,0,1),(0,1,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)],         [(1,0,1),(0,0,-1)]),
        ([(1,0,1),(0,0,-1)],[(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)],[(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    def build_rank14():
        """Build the standard rank-14 Strassen decomposition."""
        rank = 14
        U = np.zeros((u_dim, rank), dtype=float)
        V = np.zeros((v_dim, rank), dtype=float)
        W = np.zeros((w_dim, rank), dtype=float)

        for b in range(2):
            for r_local in range(7):
                col = b * 7 + r_local
                ue, ve, we = s[r_local]
                for (i, j, val) in ue:
                    U[i * m + j, col] += val
                for (j, k, val) in ve:
                    kk = k + 2 * b
                    V[j * p + kk, col] += val
                for (k, i, val) in we:
                    kk = k + 2 * b
                    W[kk * n + i, col] += val
        return U, V, W

    def verify(U, V, W):
        """Verify that the decomposition correctly represents matrix multiplication."""
        # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] should equal 1 iff j matches
        # For matmul: T[(a,b),(b,c),(c,a)] = 1
        T_expected = np.zeros((u_dim, v_dim, w_dim), dtype=float)
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i = a * m + b
                    j = b * p + c
                    k = c * n + a
                    T_expected[i, j, k] = 1.0
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_recon, T_expected, atol=1e-10)

    # Build rank-14 decomposition
    U14, V14, W14 = build_rank14()

    # Verify rank-14
    if not verify(U14, V14, W14):
        # Fallback to schoolbook rank 16
        rank_fb = n * m * p
        U_fb = np.zeros((u_dim, rank_fb), dtype=float)
        V_fb = np.zeros((v_dim, rank_fb), dtype=float)
        W_fb = np.zeros((w_dim, rank_fb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Attempt rank-13 search: check if any one of the 14 Strassen vectors
    # can be expressed as a linear combination of the others, allowing
    # elimination of one rank term.
    #
    # The 14 vectors come in two groups of 7 (block 0 and block 1).
    # Within each block, the 7 vectors are linearly independent (Strassen
    # is rank-minimal for 2x2x2). Across blocks, they operate on disjoint
    # B-column and C-row indices, so they are also independent.
    #
    # However, we check systematically: for each candidate column to remove,
    # try to find coefficients that express it as a combination of the rest.
    # This is a linear algebra check on the tensor structure.

    # Build the full tensor T from the rank-14 decomposition
    T = np.einsum('ir,jr,kr->ijk', U14, V14, W14)

    # For rank-13, we need to find U13, V13, W13 with 13 columns such that
    # sum_r U13[:,r]*V13[:,r]*W13[:,r] = T
    # This is a hard problem. Instead, we check a necessary condition:
    # if we remove column c from the rank-14 decomposition, the residual
    # tensor T_res = T - U14[:,c]*V14[:,c]*W14[:,c] must be expressible
    # with rank <= 12 (the remaining 13 columns minus the one we're trying
    # to re-express). Actually, we need the residual to be rank <= 12 so
    # that together with the 13 other columns we get rank 13 total.
    #
    # More precisely: if column c can be written as a linear combination
    # of the other 13 columns in all three modes simultaneously, we can
    # reduce rank. This requires:
    #   U14[:,c] = sum_{r != c} alpha_r * U14[:,r]
    #   V14[:,c] = sum_{r != c} beta_r * V14[:,r]
    #   W14[:,c] = sum_{r != c} gamma_r * W14[:,r]
    # AND the tensor contribution matches.
    #
    # The simplest check: see if U14[:,c] is in the span of {U14[:,r] : r != c}
    # AND V14[:,c] is in the span of {V14[:,r] : r != c}
    # AND W14[:,c] is in the span of {W14[:,r] : r != c}
    # This is necessary but not sufficient.

    rank13_found = False
    for c in range(14):
        # Check if U column c is in span of others
        U_others = np.delete(U14, c, axis=1)
        U_c = U14[:, c]
        # Check rank
        if np.linalg.matrix_rank(np.column_stack([U_others, U_c])) == np.linalg.matrix_rank(U_others):
            continue
        # Check V
        V_others = np.delete(V14, c, axis=1)
        V_c = V14[:, c]
        if np.linalg.matrix_rank(np.column_stack([V_others, V_c])) == np.linalg.matrix_rank(V_others):
            continue
        # Check W
        W_others = np.delete(W14, c, axis=1)
        W_c = W14[:, c]
        if np.linalg.matrix_rank(np.column_stack([W_others, W_c])) == np.linalg.matrix_rank(W_others):
            continue

        # All three conditions met - try to find the actual combination
        # Solve for coefficients
        try:
            # U: U_c = U_others @ alpha
            alpha, _, _, _ = np.linalg.lstsq(U_others, U_c, rcond=None)
            if not np.allclose(U_others @ alpha, U_c, atol=1e-10):
                continue
            # V: V_c = V_others @ beta
            beta, _, _, _ = np.linalg.lstsq(V_others, V_c, rcond=None)
            if not np.allclose(V_others @ beta, V_c, atol=1e-10):
                continue
            # W: W_c = W_others @ gamma
            gamma, _, _, _ = np.linalg.lstsq(W_others, W_c, rcond=None)
            if not np.allclose(W_others @ gamma, W_c, atol=1e-10):
                continue

            # Now verify the tensor contribution matches
            # T_c = U_c * V_c * W_c (outer product)
            T_c = np.einsum('i,j,k->ijk', U_c, V_c, W_c)
            # T_others_combined = sum_r (U_others[:,r] * V_others[:,r] * W_others[:,r])
            # But we need: T_c = sum_r alpha_r * (U_others[:,r] * V_others[:,r] * W_others[:,r])
            # This is NOT generally true even if the individual vectors are in the span,
            # because the tensor is a 3-way outer product.
            #
            # Actually, if U_c = U_others @ alpha, V_c = V_others @ beta, W_c = W_others @ gamma,
            # then U_c * V_c * W_c = sum_{r,s,t} alpha_r * beta_s * gamma_t * U_others[:,r] * V_others[:,s] * W_others[:,t]
            # This equals sum_r (alpha_r * beta_r * gamma_r) * U_others[:,r] * V_others[:,r] * W_others[:,r]
            # ONLY if alpha, beta, gamma are all supported on the same indices and proportional.
            #
            # Let's just check directly:
            T_others = np.einsum('ir,jr,kr->ijk', U_others, V_others, W_others)
            # We need T_c to be in the span of the 13 outer products
            # Flatten and check
            T_c_flat = T_c.flatten()
            T_others_flat = T_others.reshape(-1, 13)
            # Check if T_c_flat is in the column space of T_others_flat
            rank_before = np.linalg.matrix_rank(T_others_flat)
            rank_after = np.linalg.matrix_rank(np.column_stack([T_others_flat, T_c_flat]))
            if rank_after == rank_before:
                # T_c is in the span - rank 13 is achievable!
                # Find coefficients
                coeffs, _, _, _ = np.linalg.lstsq(T_others_flat, T_c_flat, rcond=None)
                # Build rank-13 decomposition
                U13 = U_others.copy()
                V13 = V_others.copy()
                W13 = W_others.copy()
                # Add the contribution of column c to the remaining columns
                for r in range(13):
                    if abs(coeffs[r]) > 1e-12:
                        U13[:, r] += coeffs[r] * U_others[:, r]
                        V13[:, r] += coeffs[r] * V_others[:, r]
                        W13[:, r] += coeffs[r] * W_others[:, r]

                # Verify
                if verify(U13, V13, W13):
                    return {"U": U13.tolist(), "V": V13.tolist(), "W": W13.tolist()}
                rank13_found = True
        except Exception:
            continue

    # Rank-13 not found (as expected), return rank-14
    return {"U": U14.tolist(), "V": V14.tolist(), "W": W14.tolist()}

# EVOLVE-BLOCK-END
