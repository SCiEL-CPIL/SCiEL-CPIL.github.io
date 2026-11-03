"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook(n, m, p):
    """Return the standard schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _als_step(U, V, W, n, m, p, rank, deadline):
    """
    Perform one step of Alternating Least Squares.
    Returns updated U, V, W.
    """
    # Update U
    for i in range(n):
        for j in range(m):
            if time.time() > deadline:
                return U, V, W
            A = np.zeros((p, rank))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            # Solve U[i*m+j, :] @ A^T = 1
            # U_row = (A^T)^+ @ 1
            try:
                # Using lstsq is often more stable/faster than pinv for this size
                U[i * m + j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
            except:
                pass

    # Update V
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                return U, V, W
            B = np.zeros((n, rank))
            for i in range(n):
                B[i, :] = U[i * m + j, :] * W[k * n + i, :]
            try:
                V[j * p + k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
            except:
                pass

    # Update W
    for k in range(p):
        for i in range(n):
            if time.time() > deadline:
                return U, V, W
            C = np.zeros((m, rank))
            for j in range(m):
                C[j, :] = U[i * m + j, :] * V[j * p + k, :]
            try:
                W[k * n + i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
            except:
                pass

    return U, V, W

def _compute_error(U, V, W, n, m, p):
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                err += (s - 1.0) ** 2
    return np.sqrt(err)

def _try_rank(rank, n, m, p, deadline, seed_offset=0):
    """
    Try to find a decomposition of the given rank using ALS.
    """
    max_iters = 100
    num_restarts = 3
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            return None
            
        rng = np.random.RandomState(seed_offset + restart * 1000 + rank)
        
        # Initialization: Random
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            U, V, W = _als_step(U, V, W, n, m, p, rank, deadline)
            
            err = _compute_error(U, V, W, n, m, p)
            
            if abs(prev_err - err) < 1e-9:
                break
            prev_err = err
            
            if err < 1e-5:
                # Check for exactness or near-exactness
                if _verify(U, V, W, n, m, p, tol=1e-4):
                    # Try to snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if _verify(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                        return U_snap, V_snap, W_snap
                    elif _verify(U, V, W, n, m, p, tol=1e-8):
                        return U, V, W

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    school_rank = n * m * p
    
    # Fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, school_rank
    
    # Theoretical lower bound is roughly max(n*m, m*p, p*n) = 18.
    # But for matrix mult, it's higher.
    # For 3x3x3, rank is ~23. For 3x3x6, likely ~46.
    # We try ranks from 46 down to 30.
    
    ranks_to_try = list(range(46, 29, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10.0:
            break
            
        result = _try_rank(rank, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            # Found a better rank. We could stop here, or try lower.
            # Given time constraints, stopping at the first found lower rank is safe.
            # But we might find an even lower one.
            # However, the score is seed/rank. Lower rank is better.
            # Let's try to go a bit lower if we have time.
            # But finding rank 45 is hard if 46 works.
            # Let's break to save time for the next evaluation if we found one.
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
