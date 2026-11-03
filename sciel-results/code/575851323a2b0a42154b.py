"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import random
import sys

# EVOLVE-BLOCK-START

def _verify_exact(U, V, W, n, m, p, tol=1e-10):
    """Verify exact decomposition: for all i,k,j1,j2: sum_r W[k*n+i,r]*U[i*m+j1,r]*V[j2*p+k,r] == delta_{j1,j2}"""
    r = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for r_ in range(r):
                        s += W[k*n+i, r_] * U[i*m+j1, r_] * V[j2*p+k, r_]
                    target = 1.0 if j1 == j2 else 0.0
                    if abs(s - target) > tol:
                        return False
    return True

def _snap_to_halfint(M, tol=1e-6):
    """Snap matrix to nearest half-integers and return if all entries are within tol of half-integers."""
    M2 = M * 2
    M_rounded = np.round(M2)
    if np.max(np.abs(M2 - M_rounded)) < tol:
        return M_rounded / 2.0
    return None

def _als_decompose(n, m, p, rank, max_iters=500, tol=1e-12, time_limit=10.0, seed=0):
    """ALS decomposition: alternating least squares for tensor decomposition.
    
    Constraints: for all i,k,j1,j2: sum_r W[k*n+i,r]*U[i*m+j1,r]*V[j2*p+k,r] = delta_{j1,j2}
    """
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n*m, rank) * 0.5
    V = rng.randn(m*p, rank) * 0.5
    W = rng.randn(p*n, rank) * 0.5
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() > start_time + time_limit:
            break
        
        # Update W: for each (k,i), solve A_W @ w = b where A_W[(j1*m+j2), r] = U[i*m+j1,r]*V[j2*p+k,r]
        for k in range(p):
            for i in range(n):
                A_W = np.zeros((m*m, rank))
                for j1 in range(m):
                    for j2 in range(m):
                        A_W[j1*m+j2, :] = U[i*m+j1, :] * V[j2*p+k, :]
                b = np.zeros(m*m)
                for j in range(m):
                    b[j*m+j] = 1.0
                w, residuals, rank_w, sv = np.linalg.lstsq(A_W, b, rcond=None)
                W[k*n+i, :] = w
        
        # Update U: for each (i,j1), solve A_U @ u = b where A_U[(k*m+j2), r] = W[k*n+i,r]*V[j2*p+k,r]
        for i in range(n):
            for j1 in range(m):
                A_U = np.zeros((p*m, rank))
                b = np.zeros(p*m)
                for k in range(p):
                    for j2 in range(m):
                        A_U[k*m+j2, :] = W[k*n+i, :] * V[j2*p+k, :]
                        b[k*m+j2] = 1.0 if j1 == j2 else 0.0
                u, residuals, rank_u, sv = np.linalg.lstsq(A_U, b, rcond=None)
                U[i*m+j1, :] = u
        
        # Update V: for each (j2,k), solve A_V @ v = b where A_V[(i*m+j1), r] = W[k*n+i,r]*U[i*m+j1,r]
        for j2 in range(m):
            for k in range(p):
                A_V = np.zeros((n*m, rank))
                b = np.zeros(n*m)
                for i in range(n):
                    for j1 in range(m):
                        A_V[i*m+j1, :] = W[k*n+i, :] * U[i*m+j1, :]
                        b[i*m+j1] = 1.0 if j1 == j2 else 0.0
                v, residuals, rank_v, sv = np.linalg.lstsq(A_V, b, rcond=None)
                V[j2*p+k, :] = v
        
        # Check convergence periodically
        if it > 0 and it % 20 == 0:
            err = 0.0
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            s = 0.0
                            for r_ in range(rank):
                                s += W[k*n+i, r_] * U[i*m+j1, r_] * V[j2*p+k, r_]
                            target = 1.0 if j1 == j2 else 0.0
                            err += (s - target) ** 2
            if err < tol:
                break
    
    return U, V, W

def _try_rank_als(n, m, p, target_rank, time_limit=15.0, num_restarts=5):
    """Try to find a valid decomposition with target_rank using ALS."""
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() > start_time + time_limit:
            break
        
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2.0:
            break
        
        U, V, W = _als_decompose(n, m, p, target_rank, max_iters=300, 
                                   tol=1e-12, time_limit=remaining * 0.8, 
                                   seed=restart * 1000 + 42)
        
        # Check if solution is close to half-integers
        U_h = _snap_to_halfint(U, tol=1e-4)
        V_h = _snap_to_halfint(V, tol=1e-4)
        W_h = _snap_to_halfint(W, tol=1e-4)
        
        if U_h is not None and V_h is not None and W_h is not None:
            if _verify_exact(U_h, V_h, W_h, n, m, p, tol=1e-9):
                return {
                    "U": U_h.tolist(),
                    "V": V_h.tolist(),
                    "W": W_h.tolist()
                }
        
        # Also try integer snapping
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if _verify_exact(U_i, V_i, W_i, n, m, p, tol=1e-9):
            return {
                "U": U_i.tolist(),
                "V": V_i.tolist(),
                "W": W_i.tolist()
            }
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Schoolbook fallback
    schoolbook = _schoolbook_decomposition(n, m, p)
    schoolbook_rank = n * m * p
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Theoretical lower bound is max(n*m, m*p, n*p) = max(15, 25, 15) = 25
    lower_bound = max(n*m, m*p, n*p)
    
    # Try ranks from lower_bound up to schoolbook_rank - 1
    # Focus on lower ranks first since we want minimal rank
    candidate_ranks = []
    for r in range(lower_bound, schoolbook_rank):
        candidate_ranks.append(r)
    
    # Sort by rank (ascending) to find minimal rank first
    candidate_ranks = sorted(candidate_ranks)
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 15.0:
            break
        
        time_per_rank = min(10.0, deadline - time.time() - 10.0)
        if time_per_rank < 3.0:
            break
        
        result = _try_rank_als(n, m, p, target_rank, time_limit=time_per_rank, num_restarts=6)
        
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Found a valid decomposition at this rank
            # Since we're going from low to high, this is the minimal rank we found
            # We can stop here since we want minimal rank
            break
    
    if best_result is not None:
        return best_result
    
    return schoolbook

# EVOLVE-BLOCK-END
