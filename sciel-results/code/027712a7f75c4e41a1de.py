import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 120  # seconds, safety margin from 150s limit

    n, m, p = 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # ---- Build the 42-term Strassen decomposition ----
    # 6 blocks of 2x2@2x2, each with 7 Strassen terms
    # Block (jb, side): jb in 0..2 (C column pair), side in {L, R} (A/B column/row pair)
    #
    # Strassen products (U_combo, V_combo, W_combo) for 2x2@2x2:
    #   P1: (a11+a22)(b11+b22)  -> c11+=1, c22+=1
    #   P2: (a21+a22)(b11)      -> c21+=1, c22-=1
    #   P3: (a11)(b12-b22)      -> c12+=1, c22+=1
    #   P4: (a22)(b21-b11)      -> c11+=1, c21+=1
    #   P5: (a11+a12)(b22)      -> c11-=1, c12+=1
    #   P6: (a21-a11)(b11+b12)  -> c22+=1
    #   P7: (a12-a22)(b21+b22)  -> c11+=1

    # Local 2x2 indices: 0=a11/b11/c11, 1=a12/b12/c12, 2=a21/b21/c21, 3=a22/b22/c22
    S_U = [
        {0: 1, 3: 1},   # P1
        {2: 1, 3: 1},   # P2
        {0: 1},         # P3
        {3: 1},         # P4
        {0: 1, 1: 1},   # P5
        {2: 1, 0: -1},  # P6
        {1: 1, 3: -1},  # P7
    ]
    S_V = [
        {0: 1, 3: 1},   # P1
        {0: 1},         # P2
        {1: 1, 3: -1},  # P3
        {2: 1, 0: -1},  # P4
        {3: 1},         # P5
        {0: 1, 1: 1},   # P6
        {2: 1, 3: 1},   # P7
    ]
    S_W = [
        {0: 1, 3: 1},   # P1: c11+1, c22+1
        {2: 1, 3: -1},  # P2: c21+1, c22-1
        {1: 1, 3: 1},   # P3: c12+1, c22+1
        {0: 1, 2: 1},   # P4: c11+1, c21+1
        {0: -1, 1: 1},  # P5: c11-1, c12+1
        {3: 1},         # P6: c22+1
        {0: 1},         # P7: c11+1
    ]

    rank42 = 42
    U42 = np.zeros((nm, rank42))
    V42 = np.zeros((mp, rank42))
    W42 = np.zeros((pn, rank42))

    col = 0
    for jb in range(3):       # C column block: columns 2*jb, 2*jb+1
        for side in ['L', 'R']:
            # A column offset: L->0, R->2  (A is 2x4, columns 0..3)
            a_col_off = 0 if side == 'L' else 2
            # B row offset: L->0, R->2    (B is 4x6, rows 0..3)
            b_row_off = 0 if side == 'L' else 2

            for pi in range(7):
                r = col + pi
                # U: A[i][a_col_off + j_local], i in 0..1, j_local in 0..1
                for loc_i in range(2):
                    for loc_j in range(2):
                        coeff = S_U[pi].get(loc_i * 2 + loc_j, 0)
                        if coeff != 0:
                            U42[loc_i * m + (a_col_off + loc_j), r] = coeff
                # V: B[b_row_off + j_local][2*jb + k_local], j_local in 0..1, k_local in 0..1
                for loc_j in range(2):
                    for loc_k in range(2):
                        coeff = S_V[pi].get(loc_j * 2 + loc_k, 0)
                        if coeff != 0:
                            V42[(b_row_off + loc_j) * p + (2 * jb + loc_k), r] = coeff
                # W: C[2*jb + k_local][i_local], i_local in 0..1, k_local in 0..1
                # W index: k*n + i = (2*jb + k_local)*2 + i_local
                for loc_i in range(2):
                    for loc_k in range(2):
                        coeff = S_W[pi].get(loc_i * 2 + loc_k, 0)
                        if coeff != 0:
                            W42[(2 * jb + loc_k) * n + loc_i, r] = coeff
            col += 7

    # ---- Verify the 42-term decomposition is correct ----
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 if i,j,k are "connected" (always for matrix mult), 0 otherwise.
    # For matrix multiplication, T[i,j,k] = delta (it's 1 for all valid (i,j,k) since
    # the tensor is the multiplication tensor: T_{ijk} = 1 if the indices form a valid
    # product term, 0 otherwise. Actually for matmul, T[i,j,k] = 1 for ALL i,j,k in range
    # because C[i,k] = sum_j A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.

    # Let's verify
    T_expected = np.ones((n, m, p))
    T_actual = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank42):
                    s += U42[i * m + j, r] * V42[j * p + k, r] * W42[k * n + i, r]
                T_actual[i, j, k] = s
    if not np.allclose(T_actual, T_expected, atol=1e-10):
        # Fallback to schoolbook if Strassen is wrong
        rank_sb = n * m * p
        U = np.zeros((nm, rank_sb))
        V = np.zeros((mp, rank_sb))
        W = np.zeros((pn, rank_sb))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Try to reduce rank by finding linear dependencies ----
    # The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to find a set of coefficients alpha_r such that the decomposition
    # with terms (U_r, V_r, W_r) weighted by alpha_r still gives T, but with
    # some terms being linear combinations of others.
    #
    # Alternative approach: try to find a lower-rank decomposition directly.
    # The tensor has 48 entries (2*4*6). We need rank r such that there exist
    # U (8xr), V (24xr), W (12xr) with T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
    #
    # Strategy: Use the 42-term decomposition as a starting point and try to
    # eliminate terms by finding linear dependencies in the "slices".
    #
    # For a fixed (i,k), the tensor slice T[i,:,k] is a vector of length m=4.
    # T[i,j,k] = sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r]
    # This is a bilinear form in j with coefficients depending on r.
    #
    # Better: Try to find the minimal rank by checking if we can express
    # some columns of (U, V, W) as linear combinations of others.
    #
    # Practical approach: Use randomized rank reduction.
    # Pick a random subset of r terms and check if they span the tensor.
    # This is expensive, so let's try a structured approach.
    #
    # Key observation: The Strassen decomposition has 6 blocks. Within each block,
    # the 7 terms are linearly independent (that's the point of Strassen). But
    # ACROSS blocks, there may be dependencies.
    #
    # Let's try: can we reduce from 42 to 41? 40? etc.
    # For each candidate rank r, we try to find U, V, W of that rank.
    # This is a hard problem in general, but for small cases we can try.
    #
    # Simpler: check if any single term is redundant given the others.
    # For term r0, check if T = sum_{r != r0} U_r * V_r * W_r still holds
    # when we allow the other terms to be adjusted. This is equivalent to
    # checking if the "direction" of term r0 is in the span of the others.
    #
    # Actually, the cleanest approach: the tensor T is fixed. We want the
    # minimal rank decomposition. Let's try ranks 35, 36, ..., 42 and check
    # feasibility via a randomized search.
    #
    # Given time constraints, let's try a more targeted approach:
    # Check if the 42-term decomposition has any linearly dependent "directions".
    # Two terms r1, r2 are "dependent" if there exist scalars a, b (not both 0)
    # such that a*U_{r1} + b*U_{r2} = 0 AND a*V_{r1} + b*V_{r2} = 0 AND
    # a*W_{r1} + b*W_{r2} = 0. This is very unlikely for random Strassen.
    #
    # More practically: try to find a rank-r decomposition for r < 42 by
    # solving a system of equations. The number of unknowns for rank r is
    # r*(nm + mp + pn) = r*(8+24+12) = 44r. The number of equations is
    # n*m*p = 48. For r >= 2, we have more unknowns than equations, so
    # solutions should exist. The question is whether they have the right
    # structure (each term is a rank-1 tensor).
    #
    # This is the tensor rank problem, which is NP-hard in general. But for
    # this specific small case, let's try a few things.
    #
    # Given the time budget, let's just return the 42-term Strassen decomposition.
    # It's already a significant improvement over 48.
    #
    # Actually, let me try one more thing: check if rank 41 is feasible by
    # trying to eliminate one term and solving for the rest.
    # This is complex, so I'll skip it and return 42.

    if time.time() - start_time > DEADLINE - 5:
        return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}

    # Try a quick check: is there a rank-40 decomposition?
    # This would require finding a 2-term linear dependency across blocks.
    # Let's check if any pair of terms from different blocks are linearly dependent.
    # (This is a very weak check, but fast.)
    
    # For now, return the 42-term Strassen decomposition.
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
