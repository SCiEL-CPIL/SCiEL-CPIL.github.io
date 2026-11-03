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

    def verify_decomp(U, V, W, n_test=10, tol=1e-8):
        """Vectorized verification of decomposition correctness."""
        rank = U.shape[1]
        max_err = 0.0
        for _ in range(n_test):
            A_t = np.random.randn(n, m)
            B_t = np.random.randn(m, p)
            C_expected = A_t @ B_t
            # C[i,k] = sum_j sum_r A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Vectorize: for each r, compute outer product contribution
            C_computed = np.zeros((n, p))
            for r in range(rank):
                # u_r has shape (n*m,), v_r has shape (m*p,), w_r has shape (p*n,)
                # Reshape u_r to (n, m), v_r to (m, p), w_r to (p, n)
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                # Contribution: sum_j A[i,j]*B[j,k]*u_r[i,j]*v_r[j,k] * w_r[k,i]
                # = (A * u_r) @ (B * v_r) * w_r.T  (element-wise in last dim)
                M1 = (A_t * u_r) @ (B_t * v_r)  # (n, p)
                C_computed += M1 * w_r.T
            max_err = max(max_err, np.max(np.abs(C_computed - C_expected)))
        return max_err

    def try_reduce_rank(U, V, W, max_iterations=50):
        """
        Attempt to reduce rank by finding and removing linearly dependent columns.
        Uses vectorized verification for speed.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        start_reduce = time.time()

        for iteration in range(max_iterations):
            if time.time() - start_reduce > 60.0 or time.time() - start_time > deadline - 10.0:
                break

            reduced = False
            for r in range(rank - 1, -1, -1):
                if time.time() - start_reduce > 60.0 or time.time() - start_time > deadline - 10.0:
                    break

                # Temporarily remove column r
                mask = [i for i in range(rank) if i != r]
                U_test = U[:, mask]
                V_test = V[:, mask]
                W_test = W[:, mask]

                # Quick verification with a few random matrices (vectorized)
                err = verify_decomp(U_test, V_test, W_test, n_test=5, tol=1e-8)

                if err < 1e-8:
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

    # Final verification (vectorized)
    error_final = verify_decomp(U, V, W, n_test=10, tol=1e-9)
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

    # One more pass of rank reduction after fallback if we used fallback
    # (skip if we already reduced successfully)
    if error_final <= 1e-8:
        # Try additional reduction with more iterations
        if time.time() - start_time < deadline - 15.0:
            U, V, W, current_rank = try_reduce_rank(U, V, W, max_iterations=30)
            # Re-verify
            error_final2 = verify_decomp(U, V, W, n_test=10, tol=1e-9)
            if error_final2 > 1e-8:
                # Revert - rebuild from scratch
                U_vecs3 = []
                V_vecs3 = []
                W_vecs3 = []

                def add_rank1_term3(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
                    u = [0.0] * (n * m)
                    v = [0.0] * (m * p)
                    w = [0.0] * (p * n)
                    for idx, val in zip(u_indices, u_vals):
                        u[idx] += val
                    for idx, val in zip(v_indices, v_vals):
                        v[idx] += val
                    for idx, val in zip(w_indices, w_vals):
                        w[idx] += val
                    U_vecs3.append(u)
                    V_vecs3.append(v)
                    W_vecs3.append(w)

                def strassen_222_3(i0, j0, k0):
                    i1 = i0 + 1
                    j1 = j0 + 1
                    k1 = k0 + 1
                    add_rank1_term3([i0*m+j0, i1*m+j1], [1, 1], [j0*p+k0, j1*p+k1], [1, 1], [k0*n+i0, k1*n+i1], [1, 1])
                    add_rank1_term3([i1*m+j0, i1*m+j1], [1, 1], [j0*p+k0], [1], [k0*n+i1, k1*n+i1], [1, -1])
                    add_rank1_term3([i0*m+j0], [1], [j0*p+k1, j1*p+k1], [1, -1], [k1*n+i0, k1*n+i1], [1, 1])
                    add_rank1_term3([i1*m+j1], [1], [j1*p+k0, j0*p+k0], [1, -1], [k0*n+i0, k0*n+i1], [1, 1])
                    add_rank1_term3([i0*m+j0, i0*m+j1], [1, 1], [j1*p+k1], [1], [k0*n+i0, k1*n+i0], [-1, 1])
                    add_rank1_term3([i1*m+j0, i0*m+j0], [1, -1], [j0*p+k0, j0*p+k1], [1, 1], [k1*n+i1], [1])
                    add_rank1_term3([i0*m+j1, i1*m+j1], [1, -1], [j1*p+k0, j1*p+k1], [1, 1], [k0*n+i0], [1])

                def m227_decomp_3(i0, j0, k0):
                    strassen_222_3(i0, j0, k0)
                    strassen_222_3(i0, j0, k0 + 2)
                    strassen_222_3(i0, j0, k0 + 4)
                    k6 = k0 + 6
                    for i in range(2):
                        for j in range(2):
                            ii = i0 + i
                            jj = j0 + j
                            add_rank1_term3([ii*m+jj], [1.0], [jj*p+k6], [1.0], [k6*n+ii], [1.0])

                def m127_decomp_3(i0, j0, k0):
                    for j in range(2):
                        for k in range(7):
                            jj = j0 + j
                            kk = k0 + k
                            add_rank1_term3([i0*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+i0], [1.0])

                def m327_decomp_3(i0, j0, k0):
                    m227_decomp_3(i0, j0, k0)
                    m127_decomp_3(i0 + 2, j0, k0)

                m327_decomp_3(0, 0, 0)
                m327_decomp_3(0, 2, 0)

                U = np.array(U_vecs3).T
                V = np.array(V_vecs3).T
                W = np.array(W_vecs3).T
                current_rank = U.shape[1]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
