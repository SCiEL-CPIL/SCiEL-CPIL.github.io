"""Rewritten program: search for rank-minimal exact tensor decomposition of 2x3 * 3x6 matmul."""
import time
import numpy as np
import scipy.linalg as la
from itertools import product

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    
    For exact decomposition: for all i,j,k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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

def _compute_residual(U, V, W, n, m, p):
    """Compute the sum of squared residuals for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                err += (s - 1.0)**2
    return err

def _als_optimize(U, V, W, n, m, p, max_iters=200, tol=1e-12):
    """
    Alternating Least Squares optimization.
    For each factor, fix the other two and solve a linear system.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = U.shape[1]
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        old_err = _compute_residual(U, V, W, n, m, p)
        
        # Update U: for each row i*m+j, solve for U[i*m+j, :]
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is: U[i*m+j, :] @ diag(V[j*p+k, :] * W[k*n+i, :]) = 1 for all k
        # So: U[i*m+j, :] @ M = 1 where M[c,k] = V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                # Build the system: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                # Solve A^T U_row^T = b, i.e., U_row @ A^T = b^T
                # Equivalent to A^T @ U_row^T = b
                try:
                    U_row, _, _, _ = la.lstsq(A.T, b)
                    U[i*m+j, :] = U_row
                except:
                    pass
        
        # Update V: for each row j*p+k
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                try:
                    V_row, _, _, _ = la.lstsq(A.T, b)
                    V[j*p+k, :] = V_row
                except:
                    pass
        
        # Update W: for each row k*n+i
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                try:
                    W_row, _, _, _ = la.lstsq(A.T, b)
                    W[k*n+i, :] = W_row
                except:
                    pass
        
        new_err = _compute_residual(U, V, W, n, m, p)
        if abs(old_err - new_err) < tol:
            break
    
    return U, V, W, _compute_residual(U, V, W, n, m, p)

def _snap_to_half_integers(U, V, W, n, m, p, candidates=None):
    """Snap values to nearest half-integer or candidate value."""
    if candidates is None:
        candidates = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    U_s = np.zeros_like(U)
    V_s = np.zeros_like(V)
    W_s = np.zeros_like(W)
    
    for i in range(U.shape[0]):
        for c in range(U.shape[1]):
            idx = np.argmin(np.abs(candidates - U[i, c]))
            U_s[i, c] = candidates[idx]
    
    for i in range(V.shape[0]):
        for c in range(V.shape[1]):
            idx = np.argmin(np.abs(candidates - V[i, c]))
            V_s[i, c] = candidates[idx]
    
    for i in range(W.shape[0]):
        for c in range(W.shape[1]):
            idx = np.argmin(np.abs(candidates - W[i, c]))
            W_s[i, c] = candidates[idx]
    
    return U_s, V_s, W_s

def _try_rank_r(n, m, p, r, max_restarts=30):
    """
    Try to find a rank-r decomposition using ALS + snap + local search.
    """
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    # Candidate values for final snapping
    snap_candidates = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Initialize with random sparse values
        U0 = np.random.randn(n*m, r) * 0.5
        V0 = np.random.randn(m*p, r) * 0.5
        W0 = np.random.randn(p*n, r) * 0.5
        
        # Run ALS
        U_als, V_als, W_als, err_als = _als_optimize(U0, V0, W0, n, m, p, max_iters=100)
        
        # Snap to half-integers
        U_snap, V_snap, W_snap = _snap_to_half_integers(U_als, V_als, W_als, n, m, p, snap_candidates)
        err_snap = _compute_residual(U_snap, V_snap, W_snap, n, m, p)
        
        # If snap is good, try to improve with local search
        if err_snap < 1e-6:
            # Do a few rounds of ALS starting from snapped solution
            U_pol, V_pol, W_pol, err_pol = _als_optimize(U_snap, V_snap, W_snap, n, m, p, max_iters=50)
            U_pol_s, V_pol_s, W_pol_s = _snap_to_half_integers(U_pol, V_pol, W_pol, n, m, p, snap_candidates)
            err_pol_s = _compute_residual(U_pol_s, V_pol_s, W_pol_s, n, m, p)
            
            if err_pol_s < err_snap:
                U_snap, V_snap, W_snap = U_pol_s, V_pol_s, W_pol_s
                err_snap = err_pol_s
        
        if err_snap < best_err:
            best_err = err_snap
            best_U, best_V, best_W = U_snap, V_snap, W_snap
            
            if err_snap < 1e-12:
                return U_snap, V_snap, W_snap
    
    return None

def construct():
    """Construct the lowest-rank exact tensor decomposition for 2x3 * 3x6 matmul."""
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Lower bound is max(nm, mp, pn) = 18
    # We try ranks from 18 upwards
    for r in range(18, best_rank):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
            
        # Try ALS-based search
        result = _try_rank_r(n, m, p, r, max_restarts=30)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break
            
    # Safety: ensure we have a valid decomposition
    ok, _ = _verify_decomposition(best_U, best_V, best_W, n, m, p)
    if not ok:
        best_U, best_V, best_W = U_school, V_school, W_school
    
    # Convert to lists for the return format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
