"""
Matrix multiplication 2x4x7 tensor decomposition.

Strategy:
  1. Build a rank-56 trivial (identity) decomposition.
  2. Apply Strassen-style block decomposition to the 2x4x6 sub-matrix,
     reducing rank from 48 (trivial) to 42 (Strassen) for that part,
     plus 8 rank-1 terms for the remaining k=6 column -> total rank 50.
  3. Attempt to reduce rank further by:
     a. Checking for linear dependencies among the vectorized rank-1 tensors
        via SVD of the tensor matrix.
     b. Greedy elimination of dependent columns.
     c. A slice-rank perspective: treat the tensor as a 2x28 matrix of 4x7
        slices and look for a smaller spanning set of 4x7 rank-1 matrices.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]


def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    # ---- Step 1: Build the rank-50 Strassen-based decomposition ----
    rank_init = 50
    U = np.zeros((n * m, rank_init), dtype=float)
    V = np.zeros((m * p, rank_init), dtype=float)
    W = np.zeros((p * n, rank_init), dtype=float)

    col = 0

    def add_strassen_block(i0, j0, k0):
        """Add 7 Strassen multiplications for a 2x2x2 block."""
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0 * m + j0, col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[j0 * p + k0, col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1

        # M2: (A21+A22)B11
        U[(i0 + 1) * m + j0, col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[j0 * p + k0, col] += 1
        W[k0 * n + (i0 + 1), col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] -= 1
        col += 1

        # M3: A11(B12-B22)
        U[i0 * m + j0, col] += 1
        V[j0 * p + (k0 + 1), col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] -= 1
        W[(k0 + 1) * n + i0, col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1

        # M4: A22(B21-B11)
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[(j0 + 1) * p + k0, col] += 1
        V[j0 * p + k0, col] -= 1
        W[k0 * n + i0, col] += 1
        W[k0 * n + (i0 + 1), col] += 1
        col += 1

        # M5: (A11+A12)B22
        U[i0 * m + j0, col] += 1
        U[i0 * m + (j0 + 1), col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] -= 1
        W[(k0 + 1) * n + i0, col] += 1
        col += 1

        # M6: (A21-A11)(B11+B12)
        U[(i0 + 1) * m + j0, col] += 1
        U[i0 * m + j0, col] -= 1
        V[j0 * p + k0, col] += 1
        V[j0 * p + (k0 + 1), col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1

        # M7: (A12-A22)(B21+B22)
        U[i0 * m + (j0 + 1), col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] -= 1
        V[(j0 + 1) * p + k0, col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] += 1
        col += 1

    # 6 Strassen blocks covering k=0..5 (j=0..3)
    # The 2x4x6 part: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # We tile as 2x2 blocks in (j,k):
    #   (j0=0,k0=0), (j0=2,k0=0), (j0=0,k0=2), (j0=2,k0=2), (j0=0,k0=4), (j0=2,k0=4)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 4)

    # 8 rank-1 terms for k=6 (all i, j)
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + i, col] = 1
            col += 1

    assert col == rank_init, f"Expected rank {rank_init}, got {col}"

    # ---- Verification ----
    def verify(U_arr, V_arr, W_arr, num_tests=30, atol=1e-8):
        """Verify the decomposition against random integer matrices."""
        np.random.seed(12345)
        for t in range(num_tests):
            A = np.random.randint(-5, 6, size=(n, m)).astype(float)
            B = np.random.randint(-5, 6, size=(m, p)).astype(float)
            C_expected = A @ B

            U_r = U_arr.reshape(n, m, -1)
            V_r = V_arr.reshape(m, p, -1)
            W_r = W_arr.reshape(p, n, -1)
            C_comp = np.einsum('ijr,jkr,kir->ik', U_r, V_r, W_r, optimize=True)

            if not np.allclose(C_comp, C_expected, atol=atol):
                return False
        return True

    if not verify(U, V, W):
        # Fallback: trivial rank-56
        U_t = np.zeros((n * m, n * m * p), dtype=float)
        V_t = np.zeros((m * p, n * m * p), dtype=float)
        W_t = np.zeros((p * n, n * m * p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_t[i * m + j, r] = 1
                    V_t[j * p + k, r] = 1
                    W_t[k * n + i, r] = 1
        return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_init

    # ---- Tensor vectorization ----
    def tensor_vec(U_c, V_c, W_c):
        """Vectorize rank-1 tensor u⊗v⊗w into flat vector of size n*m*p."""
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = U_c[i * m + j] * V_c[j * p + k] * W_c[k * n + i]
        return T.flatten()

    # ---- SVD-based rank reduction ----
    def try_svd_reduce(U_arr, V_arr, W_arr, max_rounds=5):
        """Use SVD of the tensor matrix to find and remove dependent columns."""
        U_arr = U_arr.copy()
        V_arr = V_arr.copy()
        W_arr = W_arr.copy()
        current_rank = U_arr.shape[1]

        for _ in range(max_rounds):
            if time_left() < 30:
                break
            if current_rank <= 1:
                break

            # Build tensor matrix M: (n*m*p, current_rank)
            M = np.zeros((n * m * p, current_rank))
            for r in range(current_rank):
                M[:, r] = tensor_vec(U_arr[:, r], V_arr[:, r], W_arr[:, r])

            # SVD to find numerical rank
            try:
                _, s, Vt = np.linalg.svd(M, full_matrices=False)
            except np.linalg.LinAlgError:
                break

            tol = 1e-10 * (s[0] if len(s) > 0 else 1.0)
            num_nonzero = int(np.sum(s > tol))

            if num_nonzero >= current_rank:
                break

            # num_nonzero < current_rank: there are dependencies
            # Try to remove columns greedily
            removed = False
            # Try removing each column and check if rank drops
            for c in range(current_rank - 1, -1, -1):
                if time_left() < 25:
                    break
                keep = [r for r in range(current_rank) if r != c]
                U_test = U_arr[:, keep]
                V_test = V_arr[:, keep]
                W_test = W_arr[:, keep]

                if verify(U_test, V_test, W_test, num_tests=15, atol=1e-8):
                    U_arr = U_test
                    V_arr = V_test
                    W_arr = W_test
                    current_rank -= 1
                    removed = True
                    if current_rank < best_rank:
                        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
                        best_rank = current_rank
                    break

            if not removed:
                break

        return U_arr, V_arr, W_arr, current_rank

    # ---- Slice-rank approach ----
    def try_slice_rank_reduce(U_arr, V_arr, W_arr):
        """
        Slice-rank perspective: The tensor T[i,j,k] can be viewed as a
        2 x 28 matrix of 4x7 slices (one per (j,k) pair for each i).
        Each slice T[i,:,:] is a 4x7 matrix. The multiplication map says:
        T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        For fixed i, the 4x7 slice is:
        T_i[j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        This is not simply a linear combination of rank-1 4x7 matrices because
        W depends on k. However, we can try a different approach:
        
        Look at the structure: for each r, the contribution to T[i,j,k] is
        U[i,j,r] * V[j,k,r] * W[k,i,r].
        
        We try to find a smaller set of "effective" rank-1 terms by checking
        if any column r can be expressed as a linear combination of others
        in the tensor space. This is the same as the SVD approach but we
        also try random linear combinations to see if we can merge columns.
        """
        U_arr = U_arr.copy()
        V_arr = V_arr.copy()
        W_arr = W_arr.copy()
        current_rank = U_arr.shape[1]

        if current_rank <= 1:
            return U_arr, V_arr, W_arr, current_rank

        # Build tensor matrix
        M = np.zeros((n * m * p, current_rank))
        for r in range(current_rank):
            M[:, r] = tensor_vec(U_arr[:, r], V_arr[:, r], W_arr[:, r])

        # Try to find pairs of columns that are close to linear combinations
        # of each other (nearly dependent)
        np.random.seed(999)
        improved = True
        max_attempts = 20
        attempts = 0

        while improved and attempts < max_attempts and time_left() > 20:
            improved = False
            attempts += 1

            for c1 in range(current_rank - 1):
                for c2 in range(c1 + 1, current_rank):
                    if time_left() < 15:
                        break

                    # Check if column c2 is in span of all others
                    keep = [r for r in range(current_rank) if r != c2]
                    M_other = M[:, keep]

                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(M_other, M[:, c2], rcond=None)
                        residual = np.linalg.norm(M[:, c2] - M_other @ x)
                        norm_c2 = np.linalg.norm(M[:, c2])

                        if norm_c2 > 1e-10 and residual / norm_c2 < 1e-8:
                            # Remove column c2
                            keep_indices = [r for r in range(current_rank) if r != c2]
                            U_arr = U_arr[:, keep_indices]
                            V_arr = V_arr[:, keep_indices]
                            W_arr = W_arr[:, keep_indices]
                            M = M[:, keep_indices]
                            current_rank -= 1
                            improved = True
                            if current_rank < best_rank:
                                best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
                                best_rank = current_rank
                            break
                    except np.linalg.LinAlgError:
                        pass
                if improved:
                    break

        return U_arr, V_arr, W_arr, current_rank

    # ---- Apply reduction methods ----
    # Method 1: SVD-based
    if time_left() > 40:
        U_r, V_r, W_r, rank_r = try_svd_reduce(U, V, W, max_rounds=5)
        if rank_r < best_rank and verify(U_r, V_r, W_r):
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank_r

    # Method 2: Slice-rank / greedy elimination
    if time_left() > 25:
        U_r2, V_r2, W_r2, rank_r2 = try_slice_rank_reduce(best_U, best_V, best_W)
        if rank_r2 < best_rank and verify(U_r2, V_r2, W_r2):
            best_U, best_V, best_W = U_r2, V_r2, W_r2
            best_rank = rank_r2

    # Method 3: Another round of SVD on the reduced result
    if best_rank < rank_init and time_left() > 15:
        U_r3, V_r3, W_r3, rank_r3 = try_svd_reduce(best_U, best_V, best_W, max_rounds=3)
        if rank_r3 < best_rank and verify(U_r3, V_r3, W_r3):
            best_U, best_V, best_W = U_r3, V_r3, W_r3
            best_rank = rank_r3

    # Final verification
    if not verify(best_U, best_V, best_W, num_tests=50, atol=1e-8):
        best_U, best_V, best_W = U, V, W
        best_rank = rank_init

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
