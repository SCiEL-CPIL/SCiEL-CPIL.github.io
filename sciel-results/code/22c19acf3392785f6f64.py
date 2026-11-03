"""Rewritten program: structured tensor decomposition search for 3x3 * 3x5."""
import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit

    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 45
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

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def verify_exact(U, V, W, r, tol=1e-8):
        """Verify that the decomposition is exact (float check)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def to_fraction_list(matrix):
        """Convert numpy array to list of Fractions with half-integer entries."""
        result = []
        for row in matrix:
            row_frac = []
            for x in row:
                if not np.isfinite(x):
                    return None
                x_rounded = round(x * 2) / 2
                row_frac.append(Fraction(x_rounded))
            result.append(row_frac)
        return result

    def check_half_integer(U, V, W, r, tol=1e-10):
        """Check if rounding to half-integers gives exact decomposition."""
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U_h[i * m + j, r_idx] * V_h[j * p + k, r_idx] * W_h[k * n + i, r_idx]
                    if abs(val - 1.0) > tol:
                        return None
        res_U = to_fraction_list(U_h)
        res_V = to_fraction_list(V_h)
        res_W = to_fraction_list(W_h)
        if res_U is not None and res_V is not None and res_W is not None:
            return {"U": res_U, "V": res_V, "W": res_W}
        return None

    def try_als(r, num_trials=30, max_iters=100):
        """ALS (Alternating Least Squares) for rank-r decomposition."""
        t0 = time.time()
        for trial in range(num_trials):
            if time.time() > deadline - 10.0:
                return None

            np.random.seed(trial * 7919 + 104729 + r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)

            for it in range(max_iters):
                if time.time() > deadline - 10.0:
                    return None

                # Update U
                for idx in range(n * m):
                    i, j = divmod(idx, m)
                    vec = np.zeros(r)
                    for k in range(p):
                        vec += V[j * p + k] * W[k * n + i]
                    U[idx] = vec

                # Update V
                for idx in range(m * p):
                    j, k = divmod(idx, p)
                    vec = np.zeros(r)
                    for i in range(n):
                        vec += U[i * m + j] * W[k * n + i]
                    V[idx] = vec

                # Update W
                for idx in range(p * n):
                    k, i = divmod(idx, n)
                    vec = np.zeros(r)
                    for j in range(m):
                        vec += U[i * m + j] * V[j * p + k]
                    W[idx] = vec

            if not (np.all(np.isfinite(U)) and np.all(np.isfinite(V)) and np.all(np.isfinite(W))):
                continue

            # Check error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                        max_err = max(max_err, abs(val - 1.0))

            if max_err < 1e-6:
                result = check_half_integer(U, V, W, r)
                if result is not None:
                    return result

        return None

    def try_gradient_descent(r, num_restarts=8, max_iters=2000):
        """Gradient descent with adaptive learning rate."""
        rng = np.random.RandomState(12345 + r)

        for restart in range(num_restarts):
            if time.time() > deadline - 10.0:
                return None

            U = rng.randn(n * m, r) * 0.3
            V = rng.randn(m * p, r) * 0.3
            W = rng.randn(p * n, r) * 0.3

            lr = 0.005
            prev_loss = float('inf')
            patience = 0

            for it in range(max_iters):
                if time.time() > deadline - 10.0:
                    return None

                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)

                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = U[i * m + j]
                            v = V[j * p + k]
                            w = W[k * n + i]
                            val = np.dot(u * w, v)
                            err = val - 1.0
                            dU[i * m + j] += err * (w * v)
                            dV[j * p + k] += err * (u * w)
                            dW[k * n + i] += err * (u * v)

                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j] * W[k * n + i], V[j * p + k])
                            loss += (val - 1.0) ** 2

                if loss < prev_loss * 0.999:
                    lr = min(lr * 1.08, 0.05)
                    patience = 0
                else:
                    lr *= 0.85
                    patience += 1
                    if patience > 200:
                        break
                prev_loss = loss

                if loss < 1e-12:
                    break

            if loss < 1e-6:
                result = check_half_integer(U, V, W, r)
                if result is not None:
                    return result

        return None

    def try_structured_search(r):
        """
        Structured search: fix W to have a specific pattern.
        For rank-r, we try W matrices where each row has at most one nonzero,
        and the nonzeros are from {-1, 0, 1, 1/2, -1/2}.
        This reduces the search space dramatically.
        """
        if time.time() > deadline - 15.0:
            return None

        # For small ranks, enumerate possible W structures
        # W is p*n x r = 15 x r
        # Each row of W: pick which column is nonzero and what value
        # Values: -1, -1/2, 0, 1/2, 1
        # But this is still huge for r > 5

        # Instead, use a smarter approach: 
        # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For each r, define u_r = U[:, :, r] (3x3), v_r = V[:, :, r] (3x5), w_r = W[:, :, r] (5x3)
        # Then T = sum_r u_r ⊗ v_r ⊗ w_r (outer product)
        # And T[i,j,k] = 1 for all i,j,k

        # Key insight: if we fix W to be diagonal-like, we can solve for U, V
        # Let's try: W[k,i,r] = 1 if r == k*n+i, else 0 (identity in W)
        # Then T[i,j,k] = U[i,j,k*n+i] * V[j,k,k*n+i]
        # This gives rank 45 (schoolbook). Not helpful.

        # Better: try to find a rank-r solution where W has a specific structure.
        # For rank 15-20, use random search with more trials.
        return None

    def try_more_aggressive_als(r, num_trials=50, max_iters=200):
        """More aggressive ALS with different initializations."""
        t0 = time.time()
        for trial in range(num_trials):
            if time.time() > deadline - 10.0:
                return None

            # Different initialization strategies
            if trial % 3 == 0:
                np.random.seed(trial * 31337 + 99991 + r)
                U = np.random.randn(n * m, r)
                V = np.random.randn(m * p, r)
                W = np.random.randn(p * n, r)
            elif trial % 3 == 1:
                np.random.seed(trial * 52429 + 100003 + r)
                U = np.random.randint(-1, 2, (n * m, r)).astype(float)
                V = np.random.randint(-1, 2, (m * p, r)).astype(float)
                W = np.random.randint(-1, 2, (p * n, r)).astype(float)
            else:
                np.random.seed(trial * 73257 + 100007 + r)
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5

            for it in range(max_iters):
                if time.time() > deadline - 10.0:
                    return None

                for idx in range(n * m):
                    i, j = divmod(idx, m)
                    vec = np.zeros(r)
                    for k in range(p):
                        vec += V[j * p + k] * W[k * n + i]
                    U[idx] = vec

                for idx in range(m * p):
                    j, k = divmod(idx, p)
                    vec = np.zeros(r)
                    for i in range(n):
                        vec += U[i * m + j] * W[k * n + i]
                    V[idx] = vec

                for idx in range(p * n):
                    k, i = divmod(idx, n)
                    vec = np.zeros(r)
                    for j in range(m):
                        vec += U[i * m + j] * V[j * p + k]
                    W[idx] = vec

            if not (np.all(np.isfinite(U)) and np.all(np.isfinite(V)) and np.all(np.isfinite(W))):
                continue

            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                        max_err = max(max_err, abs(val - 1.0))

            if max_err < 1e-6:
                result = check_half_integer(U, V, W, r)
                if result is not None:
                    return result

        return None

    # Try ranks from lower bound up
    lower_bound = max(n * m, m * p, p * n)  # 15

    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 20.0:
            break

        # Try structured search first (fast for small r)
        result = try_structured_search(r)
        if result is not None:
            best_rank = r
            best_result = result
            break

        # Try ALS
        result = try_als(r, num_trials=25, max_iters=80)
        if result is not None:
            best_rank = r
            best_result = result
            break

        # Try more aggressive ALS
        result = try_more_aggressive_als(r, num_trials=30, max_iters=150)
        if result is not None:
            best_rank = r
            best_result = result
            break

        # Try gradient descent
        result = try_gradient_descent(r, num_restarts=6, max_iters=1500)
        if result is not None:
            best_rank = r
            best_result = result
            break

    return best_result
# EVOLVE-BLOCK-END
