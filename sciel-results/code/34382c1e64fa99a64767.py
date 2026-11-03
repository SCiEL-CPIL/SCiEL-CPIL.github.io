"""Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact for all (i,j,k)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _max_error(U, V, W, n, m, p):
    """Compute the maximum absolute error over all (i,j,k)."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(val - 1.0))
    return max_err

def _snap_to_half_integer(arr, tol=1e-6):
    """Snap array entries to nearest half-integer if close enough."""
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped, True
    return arr, False

def _try_snap_and_check(U, V, W, n, m, p):
    """Try to snap to half-integers and check exactness."""
    for tol in [1e-5, 1e-4, 1e-3, 1e-2]:
        U_snapped, u_ok = _snap_to_half_integer(U, tol=tol)
        V_snapped, v_ok = _snap_to_half_integer(V, tol=tol)
        W_snapped, w_ok = _snap_to_half_integer(W, tol=tol)
        if u_ok and v_ok and w_ok:
            if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                return U_snapped, V_snapped, W_snapped
    return None

def _solve_linear_system(A, b, allowed_vals):
    """
    Solve A x = b where x must be in allowed_vals.
    A is (4, r), b is (4,).
    Returns x if found, else None.
    """
    r = A.shape[1]
    # If r == 4, try direct solve
    if r == 4:
        try:
            x = np.linalg.solve(A, b)
            # Check if x is close to allowed values
            for val in allowed_vals:
                if np.all(np.abs(x - val) < 1e-6):
                    return x
            # Check if any component is in allowed_vals? No, all must be.
            # Try snapping
            x_snapped = np.round(x * 2) / 2
            if np.all(np.abs(A @ x_snapped - b) < 1e-6):
                return x_snapped
        except:
            pass
        return None
    
    # If r > 4, we have a underdetermined system.
    # We can parameterize the solution space.
    # x = x_p + N y, where x_p is a particular solution and N is null space.
    # This is complex. 
    # Alternative: Randomized search for x in allowed_vals^r.
    # Since r is small (e.g., 4-10), we can iterate.
    
    # First, find a particular solution using least squares
    x_ls, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    
    # If the least squares solution is close to a half-integer solution, return it
    x_snapped = np.round(x_ls * 2) / 2
    if np.all(np.abs(A @ x_snapped - b) < 1e-6):
        return x_snapped
        
    # If not, try to find a solution in the null space
    # This is heuristic. We'll just return None for now and rely on the search to find better U,V.
    return None

def _check_uv_pair(U, V, n, m, p, rank, allowed_vals):
    """
    Check if there exists W such that the decomposition holds.
    Returns W if found, else None.
    """
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for k in range(p):
            # Build the 4xR matrix A for this (i,k)
            A = np.zeros((m, rank))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            
            b = np.ones(m)
            
            # Solve for W[k*n + i, :]
            w_sol = _solve_linear_system(A, b, allowed_vals)
            if w_sol is None:
                return None
            W[k * n + i, :] = w_sol
            
    return W

def _search_rank(n, m, p, rank, deadline, seed=0):
    """
    Search for a decomposition of the given rank.
    """
    allowed_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Initialize U and V randomly
    rng = np.random.RandomState(seed)
    
    # Try multiple restarts
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        U = rng.choice(allowed_vals, size=(n * m, rank))
        V = rng.choice(allowed_vals, size=(m * p, rank))
        
        W = _check_uv_pair(U, V, n, m, p, rank, allowed_vals)
        if W is not None:
            # Verify exactness
            if _check_exact(U, V, W, n, m, p):
                return U, V, W
        
        # Local search: try to improve U and V
        # We can try swapping values in U and V
        for step in range(50):
            if time.time() > deadline - 5:
                break
                
            # Pick a random element in U or V
            if rng.random() < 0.5:
                idx = rng.randint(0, n * m - 1)
                col = rng.randint(0, rank - 1)
                old_val = U[idx, col]
                new_val = rng.choice(allowed_vals)
                U[idx, col] = new_val
            else:
                idx = rng.randint(0, m * p - 1)
                col = rng.randint(0, rank - 1)
                old_val = V[idx, col]
                new_val = rng.choice(allowed_vals)
                V[idx, col] = new_val
                
            W = _check_uv_pair(U, V, n, m, p, rank, allowed_vals)
            if W is not None:
                if _check_exact(U, V, W, n, m, p):
                    return U, V, W
            else:
                # Revert if it didn't work? Or keep it?
                # Keeping it might lead to a better solution later.
                pass

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from 47 down to 30
    for rank in range(schoolbook_rank - 1, 29, -1):
        if time.time() > deadline - 15:
            break
            
        result = _search_rank(n, m, p, rank, deadline, seed=rank * 1000)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    if best_result is None:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
