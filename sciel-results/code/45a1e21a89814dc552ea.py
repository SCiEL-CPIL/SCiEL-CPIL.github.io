"""Combined approach: structured search for low-rank exact tensor decomposition."""
import numpy as np
import time
import random
import itertools
import scipy.optimize

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin within 150s limit

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    def verify_decomposition_numpy(U_mat, V_mat, W_mat, rank):
        """Vectorized verification using numpy."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)

        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)

        result = np.zeros((n, m, p))
        for c in range(rank):
            result += np.einsum('ij,jk,ki->ijk', U_3d[:,:,c], V_3d[:,:,c], W_3d[:,:,c])

        return np.allclose(result, np.ones((n,m,p)), atol=1e-9)

    def refine_and_check(rank, initial_U, initial_V, initial_W, max_iter=200):
        """Refine using least squares and check for integer/half-integer solution."""
        if time.time() > deadline - 10:
            return None
        
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten(x):
            size_U = n * m * rank
            size_V = m * p * rank
            U = x[:size_U].reshape(n, m, rank)
            V = x[size_U:size_U+size_V].reshape(m, p, rank)
            W = x[size_U+size_V:].reshape(p, n, rank)
            return U, V, W

        def residual(x):
            U, V, W = unflatten(x)
            res = np.zeros((n, m, p))
            for c in range(rank):
                res += np.einsum('ij,jk,ki->ijk', U[:,:,c], V[:,:,c], W[:,:,c])
            return (res - 1.0).flatten()

        x0 = flatten(initial_U, initial_V, initial_W)
        
        try:
            # Use 'lm' (Levenberg-Marquardt) which is robust for this type of problem
            result = scipy.optimize.least_squares(residual, x0, method='lm', max_nfev=max_iter)
            
            # Check if the solution is close to zero residual
            final_res = np.linalg.norm(residual(result.x))
            if final_res < 1e-7:
                U, V, W = unflatten(result.x)
                
                # Check if close to integers or half-integers
                U_check = np.round(U * 2) / 2
                V_check = np.round(V * 2) / 2
                W_check = np.round(W * 2) / 2
                
                if np.allclose(U, U_check, atol=1e-6) and np.allclose(V, V_check, atol=1e-6) and np.allclose(W, W_check, atol=1e-6):
                    if verify_decomposition_numpy(U_check, V_check, W_check, rank):
                        return {"U": U_check.tolist(), "V": V_check.tolist(), "W": W_check.tolist()}
        except Exception:
            pass
        return None

    def try_rank(rank, num_attempts=20):
        """Try to find a valid decomposition at the given rank."""
        if time.time() > deadline - 10:
            return None

        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                return None

            seed = hash((rank, attempt)) % (2**32)
            np_rng = np.random.RandomState(seed)

            # Random initialization
            # Scale matters: if too large, LM might diverge or take too long. If too small, might get stuck.
            # 0.5 is a reasonable start.
            U = np_rng.randn(n*m, rank) * 0.5
            V = np_rng.randn(m*p, rank) * 0.5
            W = np_rng.randn(p*n, rank) * 0.5

            result = refine_and_check(rank, U, V, W)
            if result is not None:
                return result

        return None

    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound for matrix multiplication rank is at least max(n*p, m*p, n*m) = 24
    # But often higher. Let's start from 24.
    for rank in range(24, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break

        if rank >= best_rank:
            continue

        result = try_rank(rank, num_attempts=20)

        if result is not None:
            best_rank = rank
            best_result = result
            # If we find a very low rank, we can stop early as it's likely minimal
            if rank <= 26: 
                break

    return best_result
# EVOLVE-BLOCK-END
