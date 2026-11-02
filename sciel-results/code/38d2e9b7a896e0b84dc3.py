import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0  # 10s safety margin before 150s kill

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # For matrix multiplication of (n x m) * (m x p), the tensor T is in
    # R^{n*m} x R^{m*p} x R^{n*p}
    # T_{(i,j), (j,k), (i,k)} = 1 for all valid i, j, k
    # The decomposition is: T = sum_r U_r (tensor) V_r (tensor) W_r
    # where U_r is in R^{n*m}, V_r in R^{m*p}, W_r in R^{n*p}
    #
    # In the seed's convention:
    # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    # Note: W uses k*n + i, which means W is indexed as (k, i) not (i, k)
    # So W[k*n+i, r] corresponds to the output element C[i, k]
    #
    # The constraint is: for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a system of n*m*p = 42 equations in R * n*m + R * m*p + R * p*n variables
    # where each equation is cubic (product of three variables from different matrices).
    #
    # For rank R < 42, we need to find if such a decomposition exists.
    #
    # Strategy: Use a randomized approach with local search.
    # Initialize U, V, W randomly for a given rank R, then use gradient descent
    # or random perturbations to try to satisfy the 42 tensor equations.

    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W form a valid decomposition."""
        # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_rank_random_search(R, max_iters=50000):
        """Try to find a rank-R decomposition using random search."""
        if time.time() - start_time > time_limit - 5.0:
            return None

        np.random.seed(42 + R)  # Reproducible
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None

        for attempt in range(5):
            if time.time() - start_time > time_limit - 5.0:
                break

            # Initialize with random values
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5

            err = float('inf')
            for it in range(max_iters):
                if time.time() - start_time > time_limit - 5.0:
                    break

                # Compute error: for each (i,j,k), the product sum should be 1
                # Error = sum over (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            err += (s - 1.0) ** 2

                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()

                if err < 1e-10:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

                # Random perturbation
                mat = np.random.randint(0, 3)
                if mat == 0:
                    row = np.random.randint(0, n * m)
                    col = np.random.randint(0, R)
                    U[row, col] += np.random.randn() * 0.1
                elif mat == 1:
                    row = np.random.randint(0, m * p)
                    col = np.random.randint(0, R)
                    V[row, col] += np.random.randn() * 0.1
                else:
                    row = np.random.randint(0, p * n)
                    col = np.random.randint(0, R)
                    W[row, col] += np.random.randn() * 0.1

        if best_err < 1e-10:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        return None

    # Try ranks from 30 up to 41 (if we find a valid one, great)
    # Start from lower ranks for efficiency
    for R in range(30, 42):
        if time.time() - start_time > time_limit - 10.0:
            break
        result = try_rank_random_search(R)
        if result is not None:
            return result

    # Fall back to schoolbook
    return get_schoolbook()
