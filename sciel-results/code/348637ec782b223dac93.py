"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Combines block decomposition with Strassen and optimized sub-block decompositions.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses block decomposition with optimized sub-block decompositions.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        """Add a single rank-1 tensor term to the decomposition."""
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)

        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_222(i0, j0, k0):
        """
        Strassen decomposition for 2x2 * 2x2 block starting at (i0, j0, k0).
        Adds 7 rank-1 terms.
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1: (A11+A22)*(B11+B22)
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])

        # M2: (A21+A22)*B11
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])

        # M3: A11*(B12-B22)
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])

        # M4: A22*(B21-B11)
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])

        # M5: (A11+A12)*B22
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])

        # M6: (A21-A11)*(B11+B12)
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])

        # M7: (A12-A22)*(B21+B22)
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def m227_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x7 block.
        Split p=7 as 2+2+2+1: Strassen + Strassen + Strassen + schoolbook = 7+7+7+4 = 25
        """
        strassen_222(i0, j0, k0)
        strassen_222(i0, j0, k0 + 2)
        strassen_222(i0, j0, k0 + 4)
        # Schoolbook for k = k0+6
        k6 = k0 + 6
        for i in range(2):
            for j in range(2):
                ii = i0 + i
                jj = j0 + j
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k6], [1.0],
                               [k6*n+ii], [1.0])

    def m127_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x7 block.
        Schoolbook = rank 14.
        """
        for j in range(2):
            for k in range(7):
                jj = j0 + j
                kk = k0 + k
                add_rank1_term([i0*m+jj], [1.0],
                               [jj*p+kk], [1.0],
                               [kk*n+i0], [1.0])

    def m327_decomp(i0, j0, k0):
        """
        Decomposition for 3x2 * 2x7 block.
        Split n=3 as 2+1:
          M_{2,2,7}: 25 (Strassen x3 + schoolbook)
          M_{1,2,7}: 14 (schoolbook)
        Total: 39
        """
        m227_decomp(i0, j0, k0)
        m127_decomp(i0 + 2, j0, k0)

    def m347_decomp(i0, j0, k0):
        """
        Decomposition for 3x4 * 4x7 block.
        Split m=4 as 2+2:
          M_{3,2,7}: 39
          M_{3,2,7}: 39
        Total: 78
        """
        m327_decomp(i0, j0, k0)
        m327_decomp(i0, j0 + 2, k0)

    # Build the full decomposition
    m347_decomp(0, 0, 0)

    rank = len(U_vecs)

    # Convert to matrix format
    U = np.array(U_vecs).T  # shape (n*m, rank)
    V = np.array(V_vecs).T  # shape (m*p, rank)
    W = np.array(W_vecs).T  # shape (p*n, rank)

    # Verify correctness
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B

    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(rank):
                    val += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_computed[i, k] = val

    error = np.max(np.abs(C_computed - C_expected))
    if error > 1e-9:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_sb = [[0] * rank_school for _ in range(n * m)]
        V_sb = [[0] * rank_school for _ in range(m * p)]
        W_sb = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Try to reduce rank by finding and removing linearly dependent columns
    # Use SVD-based approach: check if any column can be expressed as linear combination of others
    def try_reduce_rank(U, V, W, max_iterations=20):
        """
        Attempt to reduce rank by finding and removing linearly dependent columns.
        For each column r, check if removing it still gives valid decomposition.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        start_reduce = time.time()

        for iteration in range(max_iterations):
            if time.time() - start_reduce > 30.0 or time.time() - start_time > deadline - 5.0:
                break

            reduced = False
            for r in range(rank - 1, -1, -1):
                if time.time() - start_reduce > 30.0 or time.time() - start_time > deadline - 5.0:
                    break

                # Temporarily remove column r
                mask = [i for i in range(rank) if i != r]
                U_test = U[:, mask]
                V_test = V[:, mask]
                W_test = W[:, mask]

                # Quick verification with a few random matrices
                test_passed = True
                for _ in range(5):
                    A_test = np.random.randn(n, m)
                    B_test = np.random.randn(m, p)
                    C_expected_test = A_test @ B_test

                    C_computed_test = np.zeros((n, p))
                    for i in range(n):
                        for k in range(p):
                            val = 0.0
                            for j in range(m):
                                for rr in range(len(U_test)):
                                    val += A_test[i, j] * B_test[j, k] * U_test[i*m+j, rr] * V_test[j*p+k, rr] * W_test[k*n+i, rr]
                            C_computed_test[i, k] = val

                    if np.max(np.abs(C_computed_test - C_expected_test)) > 1e-8:
                        test_passed = False
                        break

                if test_passed:
                    U = U_test
                    V = V_test
                    W = W_test
                    rank -= 1
                    reduced = True
                    break

            if not reduced:
                break

        return U, V, W, rank

    # Apply rank reduction
    U, V, W, current_rank = try_reduce_rank(U, V, W)

    # Final verification
    A_final = np.random.randn(n, m)
    B_final = np.random.randn(m, p)
    C_expected_final = A_final @ B_final

    C_computed_final = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(current_rank):
                    val += A_final[i, j] * B_final[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_computed_final[i, k] = val

    error_final = np.max(np.abs(C_computed_final - C_expected_final))
    if error_final > 1e-8:
        # Fallback to original Strassen decomposition
        U_vecs2 = []
        V_vecs2 = []
        W_vecs2 = []

        def add_rank1_term2(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            for idx, val in zip(u_indices, u_vals):
                u[idx] += val
            for idx, val in zip(v_indices, v_vals):
                v[idx] += val
            for idx, val in zip(w_indices, w_vals):
                w[idx] += val
            U_vecs2.append(u)
            V_vecs2.append(v)
            W_vecs2.append(w)

        def strassen_222_2(i0, j0, k0):
            i1 = i0 + 1
            j1 = j0 + 1
            k1 = k0 + 1
            add_rank1_term2([i0*m+j0, i1*m+j1], [1, 1], [j0*p+k0, j1*p+k1], [1, 1], [k0*n+i0, k1*n+i1], [1, 1])
            add_rank1_term2([i1*m+j0, i1*m+j1], [1, 1], [j0*p+k0], [1], [k0*n+i1, k1*n+i1], [1, -1])
            add_rank1_term2([i0*m+j0], [1], [j0*p+k1, j1*p+k1], [1, -1], [k1*n+i0, k1*n+i1], [1, 1])
            add_rank1_term2([i1*m+j1], [1], [j1*p+k0, j0*p+k0], [1, -1], [k0*n+i0, k0*n+i1], [1, 1])
            add_rank1_term2([i0*m+j0, i0*m+j1], [1, 1], [j1*p+k1], [1], [k0*n+i0, k1*n+i0], [-1, 1])
            add_rank1_term2([i1*m+j0, i0*m+j0], [1, -1], [j0*p+k0, j0*p+k1], [1, 1], [k1*n+i1], [1])
            add_rank1_term2([i0*m+j1, i1*m+j1], [1, -1], [j1*p+k0, j1*p+k1], [1, 1], [k0*n+i0], [1])

        def m227_decomp_2(i0, j0, k0):
            strassen_222_2(i0, j0, k0)
            strassen_222_2(i0, j0, k0 + 2)
            strassen_222_2(i0, j0, k0 + 4)
            k6 = k0 + 6
            for i in range(2):
                for j in range(2):
                    ii = i0 + i
                    jj = j0 + j
                    add_rank1_term2([ii*m+jj], [1.0], [jj*p+k6], [1.0], [k6*n+ii], [1.0])

        def m127_decomp_2(i0, j0, k0):
            for j in range(2):
                for k in range(7):
                    jj = j0 + j
                    kk = k0 + k
                    add_rank1_term2([i0*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+i0], [1.0])

        def m327_decomp_2(i0, j0, k0):
            m227_decomp_2(i0, j0, k0)
            m127_decomp_2(i0 + 2, j0, k0)

        m327_decomp_2(0, 0, 0)
        m327_decomp_2(0, 2, 0)

        U = np.array(U_vecs2).T
        V = np.array(V_vecs2).T
        W = np.array(W_vecs2).T
        current_rank = U.shape[1]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
