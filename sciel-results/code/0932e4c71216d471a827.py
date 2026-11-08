"""Find a rank-minimal exact tensor decomposition for 3x3 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i==i', j==j', k==k'
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    # Function to snap to half-integers
    def snap_to_half(x):
        return np.round(2.0 * x) / 2.0

    # Function to compute residual
    def compute_residual(params, U, V, W):
        # params shape (nm + mp + pn, rank)
        r = U.shape[1]
        residual = np.zeros((nm, mp, pn))
        for c in range(r):
            residual += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return (T - residual).ravel()

    # Function to extract U, V, W from params
    def extract_UVW(params, rank):
        U = params[:nm, :]
        V = params[nm:nm+mp, :]
        W = params[nm+mp:nm+mp+pn, :]
        return U, V, W

    # Function to verify exactness with snapped values
    def verify_exact(U, V, W):
        # Snap to half-integers
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        # Check if entries are exactly half-integers
        U_ok = np.allclose(U, U_s, atol=1e-9)
        V_ok = np.allclose(V, V_s, atol=1e-9)
        W_ok = np.allclose(W, W_s, atol=1e-9)
        if not (U_ok and V_ok and W_ok):
            return False, None, None, None
        # Compute the product tensor
        prod = np.zeros((nm, mp, pn))
        for c in range(U_s.shape[1]):
            prod += np.einsum('i,j,k->ijk', U_s[:, c], V_s[:, c], W_s[:, c])
        # Check if it matches T
        if np.allclose(prod, T, atol=1e-9):
            return True, U_s, V_s, W_s
        return False, None, None

    # Try to find a decomposition of given rank
    def try_rank(rank, max_restarts=5, max_iter=200):
        best_result = None
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            # Initialize with random values
            np.random.seed(42 + restart)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5

            # Flatten
            params = np.concatenate([U.ravel(), V.ravel(), W.ravel()])

            # Define the objective
            def objective(params):
                U, V, W = extract_UVW(params, rank)
                return compute_residual(params, U, V, W)

            # Optimize
            try:
                result = least_squares(objective, params, method='lm', max_nfev=max_iter)
            except Exception:
                continue

            U, V, W = extract_UVW(result.x, rank)
            residual_norm = np.linalg.norm(result.fun)

            if residual_norm < 1e-7:
                exact, U_s, V_s, W_s = verify_exact(U, V, W)
                if exact:
                    best_result = (rank, U_s, V_s, W_s)
                    break

        return best_result

    # Try ranks from 23 up to 26
    for rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        result = try_rank(rank, max_restarts=5, max_iter=150)
        if result is not None:
            r, U, V, W = result
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook decomposition of rank 27
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
