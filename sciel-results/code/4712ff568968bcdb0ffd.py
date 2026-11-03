import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145  # Safety margin before 150s kill

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def _verify(U, V, W):
        """Verify exact decomposition"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    def _snap(arr):
        """Snap to nearest half-integer"""
        return np.round(2 * arr) / 2.0

    def _solve_rank(R, max_iters=200, tol=1e-12):
        """
        Try to find a decomposition of rank R using ALS.
        Returns (U, V, W) if found, else None.
        """
        np.random.seed(int(R * 1000 + 42)) # Deterministic seed per rank for reproducibility
        
        for restart in range(50):
            if time.time() - start_time > time_limit:
                return None

            # Initialize
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Pre-compute index mappings for speed
            # We need to solve for U, V, W iteratively.
            
            for it in range(max_iters):
                if time.time() - start_time > time_limit:
                    return None

                # 1. Solve for U
                # Eq: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # For each (i,j), we have R unknowns U[i*m+j, 0..R-1]
                # But the equation couples i,j,k.
                # Let's build the system for U.
                # Unknowns: U_flat (n*m*R)
                # Equations: n*m*p = 24
                A_U = np.zeros((n * m * p, n * m * R))
                b_U = np.ones(n * m * p)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(R):
                                coeff = V[j * p + k, r] * W[k * n + i, r]
                                unk_idx = (i * m + j) * R + r
                                A_U[eq_idx, unk_idx] = coeff
                            eq_idx += 1
                U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                U = U_flat.reshape(n * m, R)

                # 2. Solve for V
                A_V = np.zeros((n * m * p, m * p * R))
                b_V = np.ones(n * m * p)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(R):
                                coeff = U[i * m + j, r] * W[k * n + i, r]
                                unk_idx = (j * p + k) * R + r
                                A_V[eq_idx, unk_idx] = coeff
                            eq_idx += 1
                V_flat, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                V = V_flat.reshape(m * p, R)

                # 3. Solve for W
                A_W = np.zeros((n * m * p, p * n * R))
                b_W = np.ones(n * m * p)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(R):
                                coeff = U[i * m + j, r] * V[j * p + k, r]
                                unk_idx = (k * n + i) * R + r
                                A_W[eq_idx, unk_idx] = coeff
                            eq_idx += 1
                W_flat, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                W = W_flat.reshape(p * n, R)

                # Check error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                            error += (val - 1.0) ** 2
                if error < tol:
                    # Try to snap
                    U_snap = _snap(U)
                    V_snap = _snap(V)
                    W_snap = _snap(W)
                    if _verify(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap
                    # If snapping fails, we might try to refine with integer constraints, 
                    # but for now, if real solution is perfect, maybe snapping is close?
                    # The prompt requires integer/half-integer.
                    # If snapping breaks it, we discard this solution.
                    pass

        return None

    best_rank = rank_schoolbook
    best_result = (U_sb.tolist(), V_sb.tolist(), W_sb.tolist())

    # Try to find lower rank
    # Start from 23 down to 12
    for r in range(23, 11, -1):
        if time.time() - start_time > time_limit:
            break
        
        res = _solve_rank(r)
        if res is not None:
            best_rank = r
            best_result = res
            break

    U, V, W = best_result
    # Ensure they are lists of floats for the interface
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
