"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Uses an Alternating Least Squares (ALS) approach to find lower-rank decompositions.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 145  # 145s deadline with 5s safety margin
    
    # Schoolbook fallback
    def get_schoolbook():
        rank = schoolbook_rank
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

    def verify_decomposition(U, V, W, rank):
        """Verify exact decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integers(matrix):
        """Snap float matrix to nearest half-integer."""
        return [[int(round(val * 2)) / 2.0 for val in row] for row in matrix]

    def run_als(U, V, W, max_iter=100, tol=1e-8):
        """
        Run ALS iterations to minimize error.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        """
        R = U.shape[1]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                return False, U, V, W
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                    # Let A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # We want A @ u = 1
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k] = V[j * p + k] * W[k * n + i]
                    
                    # Solve A u = 1
                    # Use lstsq for stability
                    try:
                        u_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j] = u_sol
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                    # Let B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    # We want B @ v = 1
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i] = U[i * m + j] * W[k * n + i]
                    
                    try:
                        v_sol, residuals, rank_sol, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j * p + k] = v_sol
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                    # Let C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # We want C @ w = 1
                    C = np.zeros((m, R))
                    for j in range(m):
                        C[j] = U[i * m + j] * V[j * p + k]
                    
                    try:
                        w_sol, residuals, rank_sol, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k * n + i] = w_sol
                    except:
                        pass

            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        err += (val - 1.0) ** 2
            
            if err < tol:
                return True, U, V, W

        return False, U, V, W

    # Try to find lower rank decompositions
    # Start from a reasonable upper bound and go down
    # Schoolbook is 36. Try 30, 29, 28...
    
    # We will use numpy arrays for ALS, but return lists
    best_result = None
    
    # Try ranks 30 down to 24
    for candidate_rank in range(30, 23, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for this rank
        # Limit restarts to keep time manageable
        max_restarts = 5 if candidate_rank > 26 else 10
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            # Using small random values to avoid divergence
            U = np.random.randn(n * m, candidate_rank) * 0.1
            V = np.random.randn(m * p, candidate_rank) * 0.1
            W = np.random.randn(p * n, candidate_rank) * 0.1
            
            # Run ALS
            converged, U_final, V_final, W_final = run_als(U, V, W, max_iter=200, tol=1e-10)
            
            if converged:
                # Snap to half-integers
                U_snap = snap_to_half_integers(U_final)
                V_snap = snap_to_half_integers(V_final)
                W_snap = snap_to_half_integers(W_final)
                
                # Verify exactness of snapped version
                if verify_decomposition(U_snap, V_snap, W_snap, candidate_rank):
                    best_result = {
                        "U": U_snap,
                        "V": V_snap,
                        "W": W_snap
                    }
                    break
        
        if best_result is not None:
            break

    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
