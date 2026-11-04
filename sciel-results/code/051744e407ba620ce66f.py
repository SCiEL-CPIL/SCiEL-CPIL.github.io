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
    Uses block decomposition with Strassen for 2x2x2 blocks and schoolbook for remainders.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    def build_schoolbook():
        """Build the schoolbook decomposition (rank = n*m*p = 84)."""
        U_vecs = []
        V_vecs = []
        W_vecs = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0.0] * (n * m)
                    v = [0.0] * (m * p)
                    w = [0.0] * (p * n)
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
                    U_vecs.append(u)
                    V_vecs.append(v)
                    W_vecs.append(w)
        U = np.array(U_vecs).T  # shape (n*m, rank)
        V = np.array(V_vecs).T  # shape (m*p, rank)
        W = np.array(W_vecs).T  # shape (p*n, rank)
        return U, V, W

    def verify_decomp(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reconstructs the tensor."""
        rank = U.shape[1]
        # Vectorized verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    total = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def strassen_222(i0, j0, k0, U_vecs, V_vecs, W_vecs):
        """
        Strassen decomposition for 2x2 * 2x2 block starting at (i0, j0, k0).
        Adds 7 rank-1 terms.
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
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

    def build_strassen_decomp():
        """Build the Strassen-based decomposition."""
        U_vecs = []
        V_vecs = []
        W_vecs = []

        def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
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

        def m227_decomp(i0, j0, k0):
            """2x2x7 block: three Strassen 2x2x2 blocks + 4 schoolbook terms for last k"""
            strassen_222(i0, j0, k0, U_vecs, V_vecs, W_vecs)
            strassen_222(i0, j0, k0 + 2, U_vecs, V_vecs, W_vecs)
            strassen_222(i0, j0, k0 + 4, U_vecs, V_vecs, W_vecs)
            k6 = k0 + 6
            for i in range(2):
                for j in range(2):
                    ii = i0 + i
                    jj = j0 + j
                    add_rank1_term([ii*m+jj], [1.0],
                                   [jj*p+k6], [1.0],
                                   [k6*n+ii], [1.0])

        def m127_decomp(i0, j0, k0):
            """1x2x7 block: 14 schoolbook terms"""
            for j in range(2):
                for k in range(7):
                    jj = j0 + j
                    kk = k0 + k
                    add_rank1_term([i0*m+jj], [1.0],
                                   [jj*p+kk], [1.0],
                                   [kk*n+i0], [1.0])

        def m327_decomp(i0, j0, k0):
            """3x2x7 block: 2x2x7 + 1x2x7"""
            m227_decomp(i0, j0, k0)
            m127_decomp(i0 + 2, j0, k0)

        # m347: split m=4 as 2+2
        m327_decomp(0, 0, 0)
        m327_decomp(0, 2, 0)

        U = np.array(U_vecs).T
        V = np.array(V_vecs).T
        W = np.array(W_vecs).T
        return U, V, W

    # Build Strassen decomposition
    U, V, W = build_strassen_decomp()
    current_rank = U.shape[1]

    # Verify
    if not verify_decomp(U, V, W):
        # Fallback to schoolbook
        U, V, W = build_schoolbook()
        current_rank = U.shape[1]

    # Try to reduce rank using joint linear dependence detection
    def mpo_optimize(U, V, W, max_time=100.0):
        """
        Rank reduction via joint linear dependence detection.
        For each column r, check if U[:,r], V[:,r], W[:,r] are each in the
        span of the other columns. If so, remove and verify.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        start_opt = time.time()
        tol = 1e-10

        for iteration in range(300):
            if time.time() - start_opt > max_time or time.time() - start_time > deadline - 15.0:
                break

            reduced = False
            for r in range(rank - 1, -1, -1):
                if time.time() - start_opt > max_time or time.time() - start_time > deadline - 15.0:
                    break

                other = [i for i in range(rank) if i != r]
                if len(other) == 0:
                    break

                U_other = U[:, other]
                V_other = V[:, other]
                W_other = W[:, other]

                # Check U linear dependence
                try:
                    alpha_u, _, _, _ = np.linalg.lstsq(U_other, U[:, r], rcond=None)
                    if np.linalg.norm(U[:, r] - U_other @ alpha_u) > tol:
                        continue
                except:
                    continue

                # Check V linear dependence
                try:
                    alpha_v, _, _, _ = np.linalg.lstsq(V_other, V[:, r], rcond=None)
                    if np.linalg.norm(V[:, r] - V_other @ alpha_v) > tol:
                        continue
                except:
                    continue

                # Check W linear dependence
                try:
                    alpha_w, _, _, _ = np.linalg.lstsq(W_other, W[:, r], rcond=None)
                    if np.linalg.norm(W[:, r] - W_other @ alpha_w) > tol:
                        continue
                except:
                    continue

                # All three linearly dependent: try removal and verify
                U_test = np.delete(U, r, axis=1)
                V_test = np.delete(V, r, axis=1)
                W_test = np.delete(W, r, axis=1)

                if verify_decomp(U_test, V_test, W_test, tol=1e-8):
                    U = U_test
                    V = V_test
                    W = W_test
                    rank -= 1
                    reduced = True
                    break

            if not reduced:
                break

        return U, V, W, rank

    # Apply rank reduction optimization
    U, V, W, current_rank = mpo_optimize(U, V, W, max_time=100.0)

    # Final verification
    if not verify_decomp(U, V, W, tol=1e-8):
        # Fallback to schoolbook
        U, V, W = build_schoolbook()
        current_rank = U.shape[1]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
