"""Search for a low-rank exact tensor decomposition of 2x3 * 3x4 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]
N, M, P = DIMENSIONS
# Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
# Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # seconds, safety margin before 150s kill
    MARGIN = 10.0     # keep 10s margin

    n, m, p = N, M, P
    dim_u = n * m  # 6
    dim_v = m * p  # 12
    dim_w = p * n  # 8

    # Target tensor flattened: T[i,j,k] = 1 for all valid (i,j,k)
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1

    # Start with schoolbook rank = n*m*p = 24
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    # Initialize schoolbook
    U_sb = np.zeros((dim_u, best_rank))
    V_sb = np.zeros((dim_v, best_rank))
    W_sb = np.zeros((dim_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U,V,W exactly decompose the all-ones tensor."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def try_rank(r, num_trials=500, seed=42):
        """Try to find a rank-r decomposition via random initialization + optimization."""
        rng = np.random.RandomState(seed)
        for trial in range(num_trials):
            if time.time() > DEADLINE - MARGIN:
                return None, None, None

            # Random initialization with small values
            U = rng.randn(dim_u, r) * 0.1
            V = rng.randn(dim_v, r) * 0.1
            W = rng.randn(dim_w, r) * 0.1

            # Use a simple gradient-free approach: alternating minimization
            # For fixed V,W, solve for U; for fixed U,W, solve for V; for fixed U,V, solve for W
            # This is non-convex, so we use multiple restarts

            for it in range(200):
                if time.time() > DEADLINE - MARGIN:
                    return None, None, None

                # Update U: for each (i,j), U[i*m+j,:] should satisfy
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is a system of p equations in r unknowns (the row of U)
                # Let A[i,j] be a p x r matrix where A[i,j][k,c] = V[j*p+k,c] * W[k*n+i,c]
                # Then A[i,j] @ U[i*m+j,:] = ones(p)

                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        # Solve A @ x = ones(p)
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass

                # Update V: for each j, V[j*p+k,:] for k in range(p)
                # For fixed j, we need: for all i,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # Group by k: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                # Let B[j,k] be an n x r matrix where B[j,k][i,c] = U[i*m+j,c] * W[k*n+i,c]
                # Then B[j,k] @ V[j*p+k,:] = ones(n)
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                B[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                        except:
                            pass

                # Update W: for each k, W[k*n+i,:] for i in range(n)
                # For fixed k, we need: for all i,j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # Group by i: for each i, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                # Let C[k,i] be an m x r matrix where C[k,i][j,c] = U[i*m+j,c] * V[j*p+k,c]
                # Then C[k,i] @ W[k*n+i,:] = ones(m)
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                C[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                        except:
                            pass

                # Check convergence
                if check_decomposition(U, V, W, tol=1e-6):
                    # Verify with tighter tolerance
                    if check_decomposition(U, V, W, tol=1e-10):
                        return U.copy(), V.copy(), W.copy()

            # Try a few more iterations with tighter check
            if check_decomposition(U, V, W, tol=1e-10):
                return U.copy(), V.copy(), W.copy()

        return None, None, None

    # Try ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # But the tensor is all-ones, which has rank 1 in the standard sense...
    # However, the constraint structure makes it more complex.
    # Let's try from rank 12 upward.

    for r in range(12, best_rank + 1):
        if time.time() > DEADLINE - MARGIN:
            break

        U_res, V_res, W_res = try_rank(r, num_trials=300, seed=r * 1000)
        if U_res is not None:
            best_rank = r
            best_U, best_V, best_W = U_res, V_res, W_res
            break

    # If we found a better rank, use it; otherwise fall back to schoolbook
    if best_rank < n * m * p:
        U_out = [[float(best_U[i, c]) for c in range(best_rank)] for i in range(dim_u)]
        V_out = [[float(best_V[i, c]) for c in range(best_rank)] for i in range(dim_v)]
        W_out = [[float(best_W[i, c]) for c in range(best_rank)] for i in range(dim_w)]
    else:
        U_out = [[float(best_U[i, c]) for c in range(best_rank)] for i in range(dim_u)]
        V_out = [[float(best_V[i, c]) for c in range(best_rank)] for i in range(dim_v)]
        W_out = [[float(best_W[i, c]) for c in range(best_rank)] for i in range(dim_w)]

    return {"U": U_out, "V": V_out, "W": W_out}
