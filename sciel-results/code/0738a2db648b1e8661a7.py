import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        """Check if U, V, W (lists) form a valid decomposition."""
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_approx = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_approx, T, atol=1e-9)

    def build_rank28():
        """Build the known rank-28 decomposition."""
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0

        # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1

        # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1

        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - 7 terms
        # M1
        U[0 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1
        W[1 * n + 1][r] = 1
        r += 1

        # M2
        U[1 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        W[0 * n + 1][r] = 1
        W[1 * n + 1][r] = -1
        r += 1

        # M3
        U[0 * m + 0][r] = 1
        V[0 * p + 1][r] = 1
        V[1 * p + 1][r] = -1
        W[1 * n + 0][r] = 1
        W[1 * n + 1][r] = 1
        r += 1

        # M4
        U[1 * m + 1][r] = 1
        V[1 * p + 0][r] = 1
        V[0 * p + 0][r] = -1
        W[0 * n + 0][r] = 1
        W[0 * n + 1][r] = 1
        r += 1

        # M5
        U[0 * m + 0][r] = 1
        U[0 * m + 1][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = -1
        W[1 * n + 0][r] = 1
        r += 1

        # M6
        U[1 * m + 0][r] = 1
        U[0 * m + 0][r] = -1
        V[0 * p + 0][r] = 1
        V[0 * p + 1][r] = 1
        W[1 * n + 1][r] = 1
        r += 1

        # M7
        U[0 * m + 1][r] = 1
        U[1 * m + 1][r] = -1
        V[1 * p + 0][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1
        r += 1

        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - 7 terms
        # M1
        U[0 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1
        W[3 * n + 1][r] = 1
        r += 1

        # M2
        U[1 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1
        W[2 * n + 1][r] = 1
        W[3 * n + 1][r] = -1
        r += 1

        # M3
        U[0 * m + 0][r] = 1
        V[0 * p + 3][r] = 1
        V[1 * p + 3][r] = -1
        W[3 * n + 0][r] = 1
        W[3 * n + 1][r] = 1
        r += 1

        # M4
        U[1 * m + 1][r] = 1
        V[1 * p + 2][r] = 1
        V[0 * p + 2][r] = -1
        W[2 * n + 0][r] = 1
        W[2 * n + 1][r] = 1
        r += 1

        # M5
        U[0 * m + 0][r] = 1
        U[0 * m + 1][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = -1
        W[3 * n + 0][r] = 1
        r += 1

        # M6
        U[1 * m + 0][r] = 1
        U[0 * m + 0][r] = -1
        V[0 * p + 2][r] = 1
        V[0 * p + 3][r] = 1
        W[3 * n + 1][r] = 1
        r += 1

        # M7
        U[0 * m + 1][r] = 1
        U[1 * m + 1][r] = -1
        V[1 * p + 2][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1
        r += 1

        return U, V, W

    def try_greedy_rank_reduction(target_rank, max_time=80.0):
        """
        Greedy rank-reduction: start from rank-28 solution, try to merge/remove
        terms by finding rank-1 components that can be absorbed into existing terms.
        Uses structured search over the tensor space.
        """
        if time.time() > deadline:
            return None

        U28, V28, W28 = build_rank28()
        U28_arr = np.array(U28, dtype=float)
        V28_arr = np.array(V28, dtype=float)
        W28_arr = np.array(W28, dtype=float)

        # Start with rank 28 and try to reduce
        # Strategy: try to find a rank-27 decomposition by solving for one
        # additional degree of freedom

        # Approach: parameterize rank-27 as rank-28 with one column set to
        # a linear combination of others, then solve
        # This is complex; instead, try direct optimization from rank-28 init

        best_loss = float('inf')
        best_solution = None

        # Multiple restarts with different initializations
        for restart in range(5):
            if time.time() > deadline - max_time * 0.1:
                break

            # Initialize from rank-28, drop one random column, add random perturbation
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))

            # Copy first target_rank columns from rank-28
            for r in range(target_rank):
                U[:, r] = U28_arr[:, r]
                V[:, r] = V28_arr[:, r]
                W[:, r] = W28_arr[:, r]

            # Add small random perturbation to break symmetry
            scale = 0.001 * (restart + 1)
            U += np.random.randn(n * m, target_rank) * scale
            V += np.random.randn(m * p, target_rank) * scale
            W += np.random.randn(p * n, target_rank) * scale

            # Gradient descent with adaptive learning rate
            current_lr = 0.001
            patience = 0
            best_restart_loss = float('inf')
            best_restart_sol = None

            for it in range(5000):
                if time.time() > deadline - 5:
                    break

                # Compute T_approx
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_approx - T
                loss = np.sum(diff ** 2)

                if loss < best_restart_loss:
                    best_restart_loss = loss
                    best_restart_sol = (U.copy(), V.copy(), W.copy())
                    patience = 0
                else:
                    patience += 1

                if loss < 1e-12:
                    break

                if patience > 500:
                    current_lr *= 0.9
                    patience = 0

                # Compute gradients
                grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
                grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
                grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)

                # Update with momentum-like damping
                U -= current_lr * grad_U
                V -= current_lr * grad_V
                W -= current_lr * grad_W

                # Adaptive learning rate decay
                if it % 200 == 0 and it > 0:
                    current_lr *= 0.95

            if best_restart_sol is not None:
                U_sol, V_sol, W_sol = best_restart_sol
                # Round to nearest half-integer
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2

                U_check = [[float(x) for x in row] for row in U_rounded]
                V_check = [[float(x) for x in row] for row in V_rounded]
                W_check = [[float(x) for x in row] for row in W_rounded]

                if verify(U_check, V_check, W_check):
                    return (U_check, V_check, W_check)

                if best_restart_loss < best_loss:
                    best_loss = best_restart_loss
                    best_solution = (U_check, V_check, W_check)

        return None

    def try_structured_merge():
        """
        Try to find a rank-27 solution by systematically trying to merge
        two of the rank-28 terms into one, checking if the resulting
        decomposition still spans the tensor.
        """
        if time.time() > deadline:
            return None

        U28, V28, W28 = build_rank28()
        U28_arr = np.array(U28, dtype=float)
        V28_arr = np.array(V28, dtype=float)
        W28_arr = np.array(W28, dtype=float)

        # For each pair of columns (i, j), check if column j can be expressed
        # as a linear combination of the other 27 columns in a way that
        # preserves the tensor decomposition
        # This is equivalent to checking if the 28 rank-1 tensors are linearly dependent

        # Build the tensor as a matrix: each column r gives a rank-1 tensor
        # T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Flatten each to a vector of length (n*m)*(m*p)*(p*n)
        # Check if these 28 vectors are linearly dependent

        nm = n * m
        mp = m * p
        pn = p * n
        dim = nm * mp * pn

        # Build matrix where column r is the flattened rank-1 tensor
        M = np.zeros((dim, 28))
        for r in range(28):
            tensor_r = np.outer(np.outer(U28_arr[:, r], V28_arr[:, r]), W28_arr[:, r])
            M[:, r] = tensor_r.flatten()

        # Check rank of M
        rank_M = np.linalg.matrix_rank(M, tol=1e-9)
        if rank_M < 28:
            # Linearly dependent - find null space
            _, s, vh = np.linalg.svd(M, full_matrices=True)
            # Null space vectors
            null_vecs = vh[rank_M:]  # shape (28- rank_M, 28)
            # For each null space vector, we can modify the decomposition
            # by adding a multiple of the null vector to the coefficient matrix
            # But this doesn't directly reduce rank...
            # Instead, try to find a sparse null vector
            for nv in null_vecs:
                # Try to make one entry zero by shifting
                for k in range(28):
                    if abs(nv[k]) > 1e-6:
                        # Shift all other columns by -nv[j]/nv[k] * column_k
                        new_U = U28_arr.copy()
                        new_V = V28_arr.copy()
                        new_W = W28_arr.copy()
                        for j in range(28):
                            if j != k:
                                factor = -nv[j] / nv[k]
                                new_U[:, j] += factor * U28_arr[:, k]
                                new_V[:, j] += factor * V28_arr[:, k]
                                new_W[:, j] += factor * W28_arr[:, k]
                        # Now column k should be zero (approximately)
                        # Remove it
                        keep = [i for i in range(28) if i != k]
                        U27 = new_U[:, keep]
                        V27 = new_V[:, keep]
                        W27 = new_W[:, keep]

                        U_check = [[float(x) for x in row] for row in U27]
                        V_check = [[float(x) for x in row] for row in V27]
                        W_check = [[float(x) for x in row] for row in W27]

                        if verify(U_check, V_check, W_check):
                            return (U_check, V_check, W_check)

        return None

    # First, verify the rank-28 solution
    U28, V28, W28 = build_rank28()
    if verify(U28, V28, W28):
        # Try structured merge first (fast, deterministic)
        result = try_structured_merge()
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}

        # Try greedy optimization
        result = try_greedy_rank_reduction(27)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}

        # Return rank 28
        return {"U": U28, "V": V28, "W": W28}
    else:
        # Fallback to schoolbook rank 30
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
