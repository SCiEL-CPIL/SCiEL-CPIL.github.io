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

def _compute_residual_vectorized(U, V, W, n, m, p):
    """
    Compute the sum of squared residuals for the decomposition using vectorization.
    For each (i,j,k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    total_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U[i*m+j]  # shape (rank,)
            for k in range(p):
                v_row = V[j*p+k]  # shape (rank,)
                w_row = W[k*n+i]  # shape (rank,)
                s = np.dot(u_row, v_row * w_row)
                total_err += (s - 1.0)**2
    return total_err

def _compute_residual(U, V, W, n, m, p):
    """Compute the sum of squared residuals for the decomposition."""
    return _compute_residual_vectorized(U, V, W, n, m, p)

def _als_step(U, V, W, n, m, p, which='U'):
    """
    Perform one ALS step. Fix two matrices, solve for the third.
    For each column c, we solve a linear system.
    """
    rank = U.shape[1]
    
    if which == 'U':
        # For each row i*m+j, solve: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
        # This is a linear system in U[i*m+j,:]
        new_U = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build system: A @ x = b where A is p x rank, b is p x 1
                A = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    A[k, :] = V[j*p+k] * W[k*n+i]
                # Solve A @ x = b
                try:
                    x, residuals, rank_solve, sv = la.lstsq(A, b)
                    new_U[i*m+j, :] = x
                except:
                    new_U[i*m+j, :] = U[i*m+j, :]
        return new_U
    elif which == 'V':
        new_V = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    A[i, :] = U[i*m+j] * W[k*n+i]
                try:
                    x, residuals, rank_solve, sv = la.lstsq(A, b)
                    new_V[j*p+k, :] = x
                except:
                    new_V[j*p+k, :] = V[j*p+k, :]
        return new_V
    else:  # which == 'W'
        new_W = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j] * V[j*p+k]
                try:
                    x, residuals, rank_solve, sv = la.lstsq(A, b)
                    new_W[k*n+i, :] = x
                except:
                    new_W[k*n+i, :] = W[k*n+i, :]
        return new_W

def _als_optimize(U, V, W, n, m, p, max_iters=100, tol=1e-10):
    """
    Run ALS optimization to minimize the residual.
    """
    current_err = _compute_residual(U, V, W, n, m, p)
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Cycle through U, V, W
        U_new = _als_step(U, V, W, n, m, p, 'U')
        V_new = _als_step(U_new, V, W, n, m, p, 'V')
        W_new = _als_step(U_new, V_new, W, n, m, p, 'W')
        
        new_err = _compute_residual(U_new, V_new, W_new, n, m, p)
        
        if new_err < current_err - tol:
            U, V, W = U_new, V_new, W_new
            current_err = new_err
            if current_err < 1e-12:
                break
        else:
            # No improvement, break
            break
    
    return U, V, W, current_err

def _snap_to_half_integers(U, V, W, n, m, p):
    """
    Snap values to nearest half-integer and verify.
    """
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    err = _compute_residual(U_snap, V_snap, W_snap, n, m, p)
    if err < 1e-9:
        return U_snap, V_snap, W_snap
    return None

def _discrete_local_search(n, m, p, r, U0, V0, W0, max_iters=2000):
    """
    Perform discrete local search on the entries of U, V, W.
    Candidates: [-1, -0.5, 0, 0.5, 1]
    """
    candidates = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    
    current_err = _compute_residual(U, V, W, n, m, p)
    
    if current_err < 1e-12:
        return U, V, W
    
    no_improve_count = 0
    max_no_improve = 50
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Random selection approach
        mat_idx = np.random.randint(0, 3)
        if mat_idx == 0:
            row = np.random.randint(0, n * m)
            col = np.random.randint(0, r)
            mat = U
        elif mat_idx == 1:
            row = np.random.randint(0, m * p)
            col = np.random.randint(0, r)
            mat = V
        else:
            row = np.random.randint(0, p * n)
            col = np.random.randint(0, r)
            mat = W
            
        old_val = mat[row, col]
        
        # Try all other candidates
        best_val = old_val
        best_err = current_err
        
        for new_val in candidates:
            if new_val == old_val:
                continue
            mat[row, col] = new_val
            new_err = _compute_residual(U, V, W, n, m, p)
            if new_err < best_err:
                best_err = new_err
                best_val = new_val
            else:
                mat[row, col] = old_val  # restore
        
        if best_val != old_val:
            mat[row, col] = best_val
            current_err = best_err
            no_improve_count = 0
            
            if current_err < 1e-12:
                return U, V, W
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    return U, V, W

def _try_rank_r(n, m, p, r, max_restarts=30):
    """
    Try to find a rank-r decomposition using ALS + discrete local search.
    """
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    # Candidate values for initialization
    init_candidates = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Initialize with sparse random values from candidates
        # Bias towards 0 to keep sparsity
        probs = [0.15, 0.1, 0.5, 0.1, 0.15]  # -1, -0.5, 0, 0.5, 1
        U0 = np.random.choice(init_candidates, size=(n*m, r), p=probs)
        V0 = np.random.choice(init_candidates, size=(m*p, r), p=probs)
        W0 = np.random.choice(init_candidates, size=(p*n, r), p=probs)
        
        # First try ALS
        U_als, V_als, W_als, err_als = _als_optimize(U0, V0, W0, n, m, p, max_iters=50)
        
        # Try to snap to half-integers
        snapped = _snap_to_half_integers(U_als, V_als, W_als, n, m, p)
        if snapped is not None:
            U_sol, V_sol, W_sol = snapped
            err = _compute_residual(U_sol, V_sol, W_sol, n, m, p)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                if err < 1e-12:
                    return U_sol, V_sol, W_sol
            continue
        
        # If ALS didn't give exact solution, try discrete local search from ALS result
        U_sol, V_sol, W_sol = _discrete_local_search(n, m, p, r, U_als, V_als, W_als, max_iters=500)
        err = _compute_residual(U_sol, V_sol, W_sol, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            
            if err < 1e-12:
                return U_sol, V_sol, W_sol
    
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
            
        # Try ALS + discrete search
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
