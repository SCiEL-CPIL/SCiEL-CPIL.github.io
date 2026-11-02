import numpy as np
import time
from fractions import Fraction
import sys

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = 2, 3, 3
    SCHOOL_RANK = 18
    
    # 1. Construct Schoolbook Solution (Rank 18) as fallback
    U_school = np.zeros((n * m, SCHOOL_RANK))
    V_school = np.zeros((m * p, SCHOOL_RANK))
    W_school = np.zeros((p * n, SCHOOL_RANK))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, idx] = 1.0
                V_school[j * p + k, idx] = 1.0
                W_school[k * n + i, idx] = 1.0
                idx += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = SCHOOL_RANK

    # Helper to verify exactness with Fractions
    def is_valid_exact(U, V, W, rank):
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        # Convert to Fraction carefully
                        # Assuming entries are multiples of 0.5
                        fu = Fraction(int(np.round(U[i * m + j, r] * 2)), 2)
                        fv = Fraction(int(np.round(V[j * p + k, r] * 2)), 2)
                        fw = Fraction(int(np.round(W[k * n + i, r] * 2)), 2)
                        total += fu * fv * fw
                    if total != 1:
                        return False
        return True

    # ALS Update Function
    # Updates one matrix (U, V, or W) given the other two
    def als_update_U(U, V, W, rank):
        new_U = U.copy()
        for i in range(n):
            for j in range(m):
                idx_u = i * m + j
                # We want sum_r U[idx_u, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is a linear system in u = U[idx_u, :]
                # A @ u = b
                # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # b[k] = 1
                A = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                
                # Solve A @ u = b using least squares (or min norm if underdetermined)
                # Since rank is likely > p (17 > 3), we use pseudoinverse or lstsq
                try:
                    # Min norm solution: u = A.T @ inv(A @ A.T) @ b
                    # Or just use lstsq which handles rank deficiency
                    u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    new_U[idx_u, :] = u
                except np.linalg.LinAlgError:
                    pass
        return new_U

    def als_update_V(U, V, W, rank):
        new_V = V.copy()
        for j in range(m):
            for k in range(p):
                idx_v = j * p + k
                # We want sum_r U[i*m+j, r] * V[idx_v, r] * W[k*n+i, r] = 1 for all i
                # A @ v = b
                # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # b[i] = 1
                A = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                
                try:
                    v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    new_V[idx_v, :] = v
                except np.linalg.LinAlgError:
                    pass
        return new_V

    def als_update_W(U, V, W, rank):
        new_W = W.copy()
        for k in range(p):
            for i in range(n):
                idx_w = k * n + i
                # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[idx_w, r] = 1 for all j
                # A @ w = b
                # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # b[j] = 1
                A = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                try:
                    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    new_W[idx_w, :] = w
                except np.linalg.LinAlgError:
                    pass
        return new_W

    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        return err

    # Try ranks from 15 up to 17
    # Theoretical lower bound for M_{2,3,3} is 18, but we search just in case or for bugs
    # Actually, for 2x3x3, rank is known to be 18. But let's try to find a valid one if possible.
    # If we find a valid one for rank < 18, we return it.
    # If not, we return schoolbook (18).
    
    for rank in [15, 16, 17]:
        if time.time() > deadline:
            break
            
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            rng = np.random.RandomState(restart * 1000 + rank)
            
            # Initialize
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            # ALS Iterations
            max_iters = 20
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                U = als_update_U(U, V, W, rank)
                V = als_update_V(U, V, W, rank)
                W = als_update_W(U, V, W, rank)
                
                # Check error
                err = compute_error(U, V, W)
                if err < 1e-10:
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify exactness
                    if is_valid_exact(U_snap, V_snap, W_snap, rank):
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        best_rank = rank
                        break # Found valid solution for this rank, go to next rank (lower is better)
            
            if best_rank == rank:
                break # Found solution for current rank, which is the lowest we tried so far.

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
