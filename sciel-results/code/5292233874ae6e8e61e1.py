import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    # n=2, m=4, p=7

    # We will build a decomposition using a hybrid approach:
    # Split the m=4 dimension into two 2x2 blocks.
    # For each 2x2 block, split p=7 into [2,2,2,1].
    # Use Strassen (rank 7) for the 2x2x2 parts and Schoolbook (rank 4) for the 2x2x1 part.
    # Total rank = 2 * (3*7 + 4) = 2 * 25 = 50.

    rank_total = 50

    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)

    col = 0

    def add_strassen_block(i0, j0, k0, offset):
        """
        Add a 2x2x2 Strassen decomposition.
        i0: starting row index in A
        j0: starting col index in A (which is row index in B)
        k0: starting col index in B (which is row index in C)
        offset: column offset in the global rank matrices
        Returns new column index
        """
        nonlocal col
        if col + 7 > rank_total:
            return col

        # Indices in U (A matrix, size n x m)
        a00 = i0 * m + j0
        a01 = i0 * m + (j0 + 1)
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + (j0 + 1)

        # Indices in V (B matrix, size m x p)
        b00 = j0 * p + k0
        b01 = j0 * p + (k0 + 1)
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + (k0 + 1)

        # Indices in W (C matrix, size p x n, trace convention W[k*n+i])
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        r = col
        # r=0: M1 = (A00+A11)*(B00+B11)
        U[a00, r] = 1
        U[a11, r] = 1
        V[b00, r] = 1
        V[b11, r] = 1
        W[c00, r] = 1
        W[c11, r] = 1
        col += 1

        # r=1: M2 = (A10+A11)*B00
        U[a10, r] = 1
        U[a11, r] = 1
        V[b00, r] = 1
        W[c10, r] = 1
        W[c11, r] = -1
        col += 1

        # r=2: M3 = A00*(B10+B11)
        U[a00, r] = 1
        V[b10, r] = 1
        V[b11, r] = -1
        W[c01, r] = 1
        W[c11, r] = 1
        col += 1

        # r=3: M4 = A11*(B01-B00)
        U[a11, r] = 1
        V[b01, r] = 1
        V[b00, r] = -1
        W[c00, r] = 1
        W[c10, r] = 1
        col += 1

        # r=4: M5 = (A00+A01)*B11
        U[a00, r] = 1
        U[a01, r] = 1
        V[b11, r] = 1
        W[c00, r] = -1
        W[c01, r] = 1
        col += 1

        # r=5: M6 = (A10-A00)*(B00+B01)
        U[a10, r] = 1
        U[a00, r] = -1
        V[b00, r] = 1
        V[b01, r] = 1
        W[c11, r] = 1
        col += 1

        # r=6: M7 = (A01-A11)*(B10+B11)
        U[a01, r] = 1
        U[a11, r] = -1
        V[b10, r] = 1
        V[b11, r] = 1
        W[c00, r] = 1
        col += 1

        return col

    def add_schoolbook_block(i0, j0, k0, offset):
        """
        Add a 2x2x1 Schoolbook decomposition.
        """
        nonlocal col
        if col + 4 > rank_total:
            return col

        a00 = i0 * m + j0
        a01 = i0 * m + (j0 + 1)
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + (j0 + 1)

        b00 = j0 * p + k0
        b10 = (j0 + 1) * p + k0

        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        r = col
        U[a00, r] = 1
        V[b00, r] = 1
        W[c00, r] = 1
        col += 1

        U[a01, r] = 1
        V[b10, r] = 1
        W[c00, r] = 1
        col += 1

        U[a10, r] = 1
        V[b00, r] = 1
        W[c10, r] = 1
        col += 1

        U[a11, r] = 1
        V[b10, r] = 1
        W[c10, r] = 1
        col += 1

        return col

    # Process two 2x4 blocks of A
    for j_block in [0, 2]:
        # Split p=7 into [2, 2, 2, 1]
        k_splits = [0, 2, 4, 6]
        sizes = [2, 2, 2, 1]
        
        for idx, (k0, size) in enumerate(zip(k_splits, sizes)):
            if time.time() > deadline:
                break
            if size == 2:
                col = add_strassen_block(0, j_block, k0, col)
            else:
                col = add_schoolbook_block(0, j_block, k0, col)

    # Verify the decomposition
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                # Sum U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                # Vectorized dot product
                rec = np.dot(np.dot(U[a_idx, :], V[b_idx, :]), W[c_idx, :])
                # Actually, the tensor contraction is sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # Let's do it element-wise for clarity and safety
                rec = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                if abs(rec - 1.0) > max_err:
                    max_err = abs(rec - 1.0)

    if max_err > 1e-9:
        # Fallback: schoolbook rank 56
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, r] = 1
                    V_fb[j * p + k, r] = 1
                    W_fb[k * n + i, r] = 1
                    r += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
