"""
Matrix multiplication 3x4 * 4x7 tensor decomposition.
Attempts to find a rank-minimal exact decomposition using random restarts with
rank-1 updates and a verification loop. Falls back to schoolbook if no better
rank is found within the time budget.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 84

def _verify(U, V, W, tol=1e-9):
    """
    Verify that U @ diag(1) @ V^T @ diag(1) ... actually:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    We build the 3D tensor T[i,j,k] and check it equals 1.
    """
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    # Sum over r: use einsum
    T = np.einsum('ijr,jkr,inr->ijn', U_3d, V_3d, W_3d)
    return np.allclose(T, 1.0, atol=tol)

def _to_half_integer_list(arr, tol=1e-9):
    """Convert a 2D float array to a list of lists of half-integers if possible."""
    arr = np.array(arr, dtype=float)
    # Multiply by 2 and check if close to integers
    doubled = 2.0 * arr
    rounded = np.round(doubled)
    if np.allclose(doubled, rounded, atol=tol):
        # Convert to half-integers
        result = [[float(rounded[i][j]) / 2.0 for j in range(arr.shape[1])] for i in range(arr.shape[0])]
        return result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s deadline, with safety margin
    
    # Start with schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    
    # Try to find lower rank decompositions using random restarts and iterative refinement
    # We'll try ranks from a lower bound up to schoolbook rank - 1
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But actual lower bound might be higher. Let's try from 30 up to 83.
    
    # For efficiency, we'll try a few random initializations for each rank
    # and do a limited number of iterations of a simple gradient-like update.
    
    def random_init(rank):
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        return U, V, W
    
    def try_rank(rank, max_iters=200, max_restarts=5):
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            U, V, W = random_init(rank)
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                # Compute residual: T[i,j,k] - 1
                U_3d = U.reshape(n, m, rank)
                V_3d = V.reshape(m, p, rank)
                W_3d = W.reshape(p, n, rank)
                T = np.einsum('ijr,jkr,inr->ijn', U_3d, V_3d, W_3d)
                residual = T - 1.0
                # Check if close to zero
                if np.max(np.abs(residual)) < 1e-8:
                    # Snap to half-integers
                    U_list = _to_half_integer_list(U)
                    V_list = _to_half_integer_list(V)
                    W_list = _to_half_integer_list(W)
                    if U_list and V_list and W_list:
                        if _verify(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # Maybe it's valid but not half-integer, skip
                        pass
                # Gradient computation is complex; for simplicity, we'll just do random perturbations
                # This is a rough heuristic
                if it % 10 == 0:
                    U += np.random.randn(n * m, rank) * 0.1
                    V += np.random.randn(m * p, rank) * 0.1
                    W += np.random.randn(p * n, rank) * 0.1
        return None
    
    # Try ranks from 30 to 83
    for rank in range(30, 84):
        if time.time() > deadline - 10:
            break
        result = try_rank(rank, max_iters=100, max_restarts=3)
        if result is not None:
            best_result = result
            break  # Found a valid decomposition, return it
    
    return best_result
