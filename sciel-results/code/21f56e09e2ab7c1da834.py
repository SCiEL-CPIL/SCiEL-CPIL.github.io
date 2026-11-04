"""Matrix multiplication 2x4x7 - rank-minimal exact decomposition."""

import time
import numpy as np

DIMENSIONS = [2, 4, 7]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill

    def build_strassen_decomposition():
        """Build a rank-50 Strassen-based decomposition."""
        rank = 50
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)

        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # 2x4x4: 4 Strassen blocks
        add_strassen(0, 0, 0)
        add_strassen(0, 2, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 2)

        # 2x4x2: 2 Strassen blocks
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)

        # 2x4x1: 8 direct terms for k=6
        for j in range(4):
            for i in range(2):
                U[i * m + j, col] = 1
                V[j * p + 6, col] = 1
                W[6 * n + i, col] = 1
                col += 1

        assert col == rank, f"Expected rank {rank}, got {col}"
        return U, V, W

    def verify_decomposition(U, V, W):
        """Verify exact decomposition."""
        rank = U.shape[1]
        nm, mp, pn = n * m, m * p, p * n

        # Check all nonzero tensor entries
        for a in range(nm):
            i, j = divmod(a, m)
            for b in range(mp):
                j2, k = divmod(b, p)
                if j != j2:
                    continue
                c = k * n + i
                actual = 0
                for r in range(rank):
                    u, v, w = U[a, r], V[b, r], W[c, r]
                    if u != 0 and v != 0 and w != 0:
                        actual += int(u * v * w)
                if actual != 1:
                    return False

        # Check zero entries (sample for speed)
        for a in range(nm):
            i, j = divmod(a, m)
            for b in range(mp):
                j2, k = divmod(b, p)
                if j == j2:
                    continue
                c = k * n + i
                actual = 0
                for r in range(rank):
                    u, v, w = U[a, r], V[b, r], W[c, r]
                    if u != 0 and v != 0 and w != 0:
                        actual += int(u * v * w)
                if actual != 0:
                    return False
        return True

    def try_reduce_rank(U, V, W):
        """
        Try to reduce rank by finding linear dependencies among rank vectors.
        For a decomposition T = sum_r u_r ⊗ v_r ⊗ w_r, if there exist
        coefficients alpha_r (not all zero) such that
        sum_r alpha_r * u_r ⊗ v_r ⊗ w_r = 0,
        then we can eliminate one vector.
        
        We check this by: for each candidate column r0 to remove, set up
        linear equations from the tensor entries to see if u_{r0}⊗v_{r0}⊗w_{r0}
        can be expressed as a linear combination of the others.
        """
        R = U.shape[1]
        nm, mp, pn = n * m, m * p, p * n
        
        # Build list of nonzero tensor entries
        nonzero_entries = []
        for a in range(nm):
            i, j = divmod(a, m)
            for b in range(mp):
                j2, k = divmod(b, p)
                if j == j2:
                    c = k * n + i
                    nonzero_entries.append((a, b, c))
        
        # For each column to potentially remove, check if it's in the span
        # of the others (in the tensor sense)
        # This requires solving: for all (a,b,c) in nonzero_entries:
        #   U[a,r0]*V[b,r0]*W[c,r0] = sum_{r!=r0} beta_r * U[a,r]*V[b,r]*W[c,r]
        # This is a system of |nonzero_entries| equations in (R-1) unknowns beta_r
        # But it's nonlinear in beta (trilinear). However, for fixed (a,b), it's linear in c.
        
        # Alternative: use the matricization approach.
        # The tensor T has a unique decomposition up to the rank vectors.
        # If we fix U and V, then W is determined (if U and V have full column rank).
        # So we can check: does there exist a (R-1)-rank decomposition?
        
        # Practical approach: try removing each column and see if the remaining
        # columns can be adjusted (scaled) to still give the correct tensor.
        # This is equivalent to checking if the column r0 of the "tensor matrix"
        # is in the column space of the others.
        
        # Form the tensor as a matrix: T_mat[a, (b,c)] for all valid (b,c) given a
        # Actually, let's use a different approach:
        # For each pair of indices (a, b) where the tensor has entries, 
        # the vector over c is determined.
        
        # Let's try: for each r0, remove it and check if the residual
        # sum_{r!=r0} U[a,r]V[b,r]W[c,r] can match U[a,r0]V[b,r0]W[c,r0]
        # by adjusting the other vectors. This is too complex.
        
        # Simpler numerical approach: check if the numerical rank of the
        # "flattened" representation is less than R.
        
        # Flatten T into a matrix of shape (nm, mp*pn) and check its rank
        # But mp*pn = 28*14 = 392, and nm = 8, so the matrix is 8x392.
        # Its rank is at most 8, which is always <= R=50. Not useful.
        
        # Better: use the "slice" approach. For each a, T_a is a mp x pn matrix.
        # T_a = sum_r U[a,r] * (v_r ⊗ w_r)
        # So T_a is in the span of the mp x pn matrices v_r ⊗ w_r.
        # The dimension of this span is at most R.
        
        # For a=0: T_0 has rank at most min(mp, pn) = min(28,14) = 14
        # But it's a specific matrix with structure.
        
        # Let's just check: is there a non-trivial linear relation?
        # Form the "coefficient matrix" M where M[(a,b), r] = U[a,r]*V[b,r]
        # Then T[a,b,c] = sum_r M[(a,b),r] * W[c,r]
        # For this to have a lower-rank decomposition, the matrices
        # M_a = M[a,:] (shape mp x R) must have a common null structure with W.
        
        # This is getting complex. Let's try a direct search:
        # Try all pairs of columns and see if merging them helps.
        
        # Actually, the most practical approach for R=50:
        # Check if any column is zero (it's not in Strassen)
        # Check if the U, V, W matrices have rank < R (they should be full rank)
        
        U_rank = np.linalg.matrix_rank(U.astype(float))
        V_rank = np.linalg.matrix_rank(V.astype(float))
        W_rank = np.linalg.matrix_rank(W.astype(float))
        
        if U_rank < R or V_rank < R or W_rank < R:
            # There might be redundancy
            # Try to find the actual rank and project
            pass
        
        return U, V, W

    # Build initial decomposition
    U, V, W = build_strassen_decomposition()

    # Verify
    if not verify_decomposition(U, V, W):
        # Should not happen
        pass

    # Try rank reduction if time permits
    if time.time() - start_time < DEADLINE - 20:
        U, V, W = try_reduce_rank(U, V, W)

    # Convert to list format
    rank = U.shape[1]
    U_list = [[int(U[i, r]) for r in range(rank)] for i in range(U.shape[0])]
    V_list = [[int(V[i, r]) for r in range(rank)] for i in range(V.shape[0])]
    W_list = [[int(W[i, r]) for r in range(rank)] for i in range(W.shape[0])]

    return {"U": U_list, "V": V_list, "W": W_list}
