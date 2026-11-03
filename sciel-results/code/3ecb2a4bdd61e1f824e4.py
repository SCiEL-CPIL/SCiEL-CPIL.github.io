"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    import random

    _deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 4, 6

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    def build_rank66():
        U66 = np.zeros((n * m, 66))
        V66 = np.zeros((m * p, 66))
        W66 = np.zeros((p * n, 66))
        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            U66[A00, col] += 1; U66[A11, col] += 1
            V66[B00, col] += 1; V66[B11, col] += 1
            W66[C00, col] += 1; W66[C11, col] += 1
            col += 1

            U66[A10, col] += 1; U66[A11, col] += 1
            V66[B00, col] += 1
            W66[C10, col] += 1; W66[C11, col] -= 1
            col += 1

            U66[A00, col] += 1
            V66[B01, col] += 1; V66[B11, col] -= 1
            W66[C01, col] += 1; W66[C11, col] += 1
            col += 1

            U66[A11, col] += 1
            V66[B10, col] += 1; V66[B00, col] -= 1
            W66[C00, col] += 1; W66[C10, col] += 1
            col += 1

            U66[A00, col] += 1; U66[A01, col] += 1
            V66[B11, col] += 1
            W66[C00, col] -= 1; W66[C01, col] += 1
            col += 1

            U66[A10, col] += 1; U66[A00, col] -= 1
            V66[B00, col] += 1; V66[B01, col] += 1
            W66[C11, col] += 1
            col += 1

            U66[A01, col] += 1; U66[A11, col] -= 1
            V66[B10, col] += 1; V66[B11, col] += 1
            W66[C00, col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            U66[A0, col] += 1; V66[B00, col] += 1; W66[C0, col] += 1
            col += 1
            U66[A1, col] += 1; V66[B10, col] += 1; W66[C0, col] += 1
            col += 1
            U66[A0, col] += 1; V66[B01, col] += 1; W66[C1, col] += 1
            col += 1
            U66[A1, col] += 1; V66[B11, col] += 1; W66[C1, col] += 1
            col += 1

        for t0 in [0, 2, 4]:
            add_1x2_2x2(0, 0, 0, t0, 0, t0)
            add_1x2_2x2(0, 2, 2, t0, 0, t0)
        for t0 in [0, 2, 4]:
            add_strassen(1, 0, 0, t0, 1, t0)
            add_strassen(1, 2, 2, t0, 1, t0)

        return U66, V66, W66

    U66, V66, W66 = build_rank66()

    if not verify(U66, V66, W66, 66):
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find linear dependencies to reduce rank below 66
    # Flatten the rank-1 tensors and check rank of the 3D array
    r0 = 66
    # Build the tensor as a matrix: each rank-1 term is a vector in R^(n*m*p)
    # Check if these 66 vectors are linearly dependent
    tensor_vecs = []
    for c in range(r0):
        vec = (U66[:, c].reshape(n, m) * 1.0)  # placeholder
        # Actually, the rank-1 tensor for column c is:
        # T_c[i,j,k] = U66[i*m+j, c] * V66[j*p+k, c] * W66[k*n+i, c]
        # We need to check if sum_c alpha_c * T_c = 0 has non-trivial solutions
        # This is equivalent to checking if the 66 vectors in R^(n*m*p) are linearly dependent
        # But that's 72-dimensional, so max rank is 72, and 66 < 72, so they could be independent
        # Instead, let's try to find a dependency by solving a linear system
        
    # Alternative: try to find if any column can be expressed as a combination of others
    # by checking the tensor equation
    # For a given set of columns S to keep, check if the remaining columns can be
    # linearly combined to reproduce the tensor
    
    # Simpler approach: try removing one column at a time and see if we can adjust
    # the remaining columns to still match the tensor
    
    best_rank = 66
    best_U, best_V, best_W = U66.copy(), V66.copy(), W66.copy()
    
    # Try to find a rank-65 decomposition by finding a linear dependency
    # The tensor equation is: sum_c U_c ⊗ V_c ⊗ W_c = T (all-ones tensor)
    # If we have a dependency sum_c alpha_c * (U_c ⊗ V_c ⊗ W_c) = 0, we can remove one term
    
    # Let's try a random perturbation approach: add small random coefficients to all terms
    # and see if we can find a lower-rank representation
    
    # Actually, let's try a structured approach: use a different partition
    # Split columns as (3, 3) instead of (2, 2, 2)
    # Rows 1-2 with cols 0-2: 2x3 block, rows 1-2 with cols 3-5: 2x3 block
    # Row 0 with cols 0-2: 1x3, row 0 with cols 3-5: 1x3
    
    # For 2x3: use rank 6 (direct) or try to do better
    # For 1x3: rank 3 (direct)
    # Total: 2*6 + 2*3 = 18? No, that's not right.
    
    # Let me think differently. The known lower bound for 3x4x6 is not well established.
    # Let's try to use a 4x6 = (2+2) x (3+3) structure
    # Rows 1-2 (2x4) x Cols 0-2 (3x3): 2x3x3
    # Rows 1-2 (2x4) x Cols 3-5 (3x3): 2x3x3
    # Row 0 (1x4) x Cols 0-2 (3x3): 1x3x3
    # Row 0 (1x4) x Cols 3-5 (3x3): 1x3x3
    
    # This doesn't immediately help. Let's stick with the 66 and try to find dependencies.
    
    # Try: can we combine the two Strassen blocks in a way that shares terms?
    # For example, if we have Strassen on (rows 1-2, cols 0-1) and (rows 1-2, cols 2-3),
    # the A-side factors are different, so no direct sharing.
    
    # Let's try a completely different partition:
    # Use a 3x4x6 = (1+2) x 4 x (2+2+2) structure
    # But handle the 2x2x2 blocks with rank 7 (Strassen) and the 1x2x2 with rank 4
    # That's what we have.
    
    # Try to find if there's a rank-65 by solving for a dependency
    # Set up: find alpha_0, ..., alpha_65 not all zero such that
    # sum_c alpha_c * U_c ⊗ V_c ⊗ W_c = 0
    
    # This is a system of 72 equations (n*m*p = 3*4*6 = 72) in 66 unknowns
    # So there are 66 variables and 72 equations - generically no non-trivial solution
    # But the structure might allow it
    
    # Let's build the 72x66 matrix where entry (ijk, c) = U66[i*m+j, c] * V66[j*p+k, c] * W66[k*n+i, c]
    M = np.zeros((n * m * p, r0))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j) * p + k  # or some consistent ordering
                # Actually, let's use a flat index
                flat_idx = i * m * p + j * p + k
                for c in range(r0):
                    M[flat_idx, c] = U66[i * m + j, c] * V66[j * p + k, c] * W66[k * n + i, c]
    
    # Check the rank of M
    # If rank(M) < 66, then there's a dependency
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-9)
    except:
        rank_M = 66
    
    if rank_M < 66:
        # Find the null space
        try:
            null_space = np.linalg.null_space(M)
            if null_space.shape[1] > 0:
                # We can reduce the rank
                # Find a column to remove
                # The null space vector gives coefficients
                # For simplicity, try to remove the last column
                # and adjust the others
                # This is complex; let's just try rank 65 with a specific structure
                pass
        except:
            pass
    
    # Try a different approach: use a 4x4x4 sub-problem
    # 3x4x6 contains a 2x4x4 sub-problem (rows 1-2, cols 0-3)
    # Rank of 2x4x4: known to be at most 10 (using Strassen twice: 7+7=14, but can do better)
    # Actually, 2x4x4 = 2x(2+2)x4 = two 2x2x4 blocks
    # Each 2x2x4: rank at most 7 (Strassen on 2x2x2 + 4 direct for 2x2x2) = 11
    # Or use a better decomposition
    
    # Let's try: 2x4x4 with rank 10
    # This would save 4 ranks compared to 2*7=14 for the two 2x2x2 blocks
    # Plus we still have the remaining parts
    
    # For now, return the rank-66 solution as a baseline
    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}
# EVOLVE-BLOCK-END
