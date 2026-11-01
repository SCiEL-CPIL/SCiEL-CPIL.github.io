"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS  # 2, 3, 3
    U_rows = n * m  # 6
    V_rows = m * p  # 9
    W_rows = p * n  # 6

    deadline = time.time() + 110  # Safety margin

    def schoolbook():
        rank = n * m * p
        U = np.zeros((U_rows, rank))
        V = np.zeros((V_rows, rank))
        W = np.zeros((W_rows, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    def tensor_residual(U, V, W):
        """Compute residual: sum over (i,j,k) of (sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] - 1)^2"""
        rank = U.shape[1]
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    total += (s - 1.0) ** 2
        return total

    def is_half_integer(arr, tol=1e-8):
        """Check if all entries are integer or half-integer."""
        flat = arr.flatten()
        v = flat * 2
        return np.all(np.abs(v - np.round(v)) < tol)

    def round_to_half(arr):
        return np.round(arr * 2) / 2

    def verify_solution(U, V, W, tol=1e-6):
        """Verify that the decomposition is exact and has half-integer entries."""
        if not (is_half_integer(U, 1e-8) and is_half_integer(V, 1e-8) and is_half_integer(W, 1e-8)):
            return False
        res = tensor_residual(U, V, W)
        return res < tol

    def try_rank(target_rank, max_attempts=20):
        """Try to find a rank-minimal decomposition for given target_rank."""
        nonlocal best_U, best_V, best_W, best_rank
        if target_rank >= best_rank:
            return False

        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                return False

            np.random.seed(42 + target_rank * 1000 + attempt * 7)
            
            # Initialize with small random values
            scale = 0.5
            U = np.random.randn(U_rows, target_rank) * scale
            V = np.random.randn(V_rows, target_rank) * scale
            W = np.random.randn(W_rows, target_rank) * scale

            # Define the residual function for least_squares
            def residual_func(x):
                U_flat = x[:U_rows * target_rank].reshape(U_rows, target_rank)
                V_flat = x[U_rows * target_rank : U_rows * target_rank + V_rows * target_rank].reshape(V_rows, target_rank)
                W_flat = x[U_rows * target_rank + V_rows * target_rank:].reshape(W_rows, target_rank)
                
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                            res.append(s - 1.0)
                return np.array(res)

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = least_squares(residual_func, x0, method='lm', max_nfev=5000)
                
                if res.cost < 1e-10:
                    U_sol = res.x[:U_rows * target_rank].reshape(U_rows, target_rank)
                    V_sol = res.x[U_rows * target_rank : U_rows * target_rank + V_rows * target_rank].reshape(V_rows, target_rank)
                    W_sol = res.x[U_rows * target_rank + V_rows * target_rank:].reshape(W_rows, target_rank)
                    
                    # Check if already half-integer or can be rounded
                    if is_half_integer(U_sol) and is_half_integer(V_sol) and is_half_integer(W_sol):
                        if tensor_residual(U_sol, V_sol, W_sol) < 1e-6:
                            best_U, best_V, best_W = U_sol, V_sol, W_sol
                            best_rank = target_rank
                            return True
                    else:
                        U_r = round_to_half(U_sol)
                        V_r = round_to_half(V_sol)
                        W_r = round_to_half(W_sol)
                        if tensor_residual(U_r, V_r, W_r) < 1e-6:
                            best_U, best_V, best_W = U_r, V_r, W_r
                            best_rank = target_rank
                            return True
            except Exception:
                continue

        return False

    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(6, 9, 6) = 9
    lower_bound = max(n * m, m * p, p * n)
    
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 20:
            break
        if try_rank(r):
            break

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
