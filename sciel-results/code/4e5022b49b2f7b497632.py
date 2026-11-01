"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W form a valid decomposition."""
    n, m, p = int(n), int(m), int(p)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _is_half_integer(val, tol=1e-6):
    """Check if val is close to an integer or half-integer."""
    doubled = val * 2
    nearest = round(doubled)
    return abs(doubled - nearest) < tol

def _snap_to_half_integer(val):
    """Snap val to nearest half-integer."""
    return round(val * 2) / 2.0

def _try_rank(r, n, m, p, max_iters=500, timeout=120.0):
    """Try to find a rank-r decomposition using iterative refinement."""
    start_time = time.time()
    
    # Total number of variables: n*m*r + m*p*r + p*n*r
    nm = n * m
    mp = m * p
    pn = p * n
    total_vars = nm * r + mp * r + pn * r
    
    # Equations: for each (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    n_eqs = n * m * p
    
    # Use a random initialization and iterative approach
    best_solution = None
    best_err = float('inf')
    
    for attempt in range(20):
        if time.time() - start_time > timeout - 5:
            break
        
        # Random initialization with small values
        np.random.seed(42 + attempt)
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        # Scale to make equations more likely to be satisfied
        # Each term is roughly 0.125, so sum of r terms ~ r*0.125
        # We want sum = 1, so scale by 1/(r*0.125) = 8/r
        scale = 8.0 / r
        U *= scale
        V *= scale
        W *= scale
        
        # Iterative refinement: fix two matrices, solve for the third
        for it in range(max_iters):
            if time.time() - start_time > timeout - 5:
                break
            
            # Compute current error
            errors = np.zeros(n_eqs)
            for idx, (i, j, k) in enumerate(itertools.product(range(n), range(m), range(p))):
                errors[idx] = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r)) - 1.0
            
            err = np.max(np.abs(errors))
            if err < best_err:
                best_err = err
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                break
            
            # Update W: for each (k,i), we need sum_c V[j*p+k,c]*W[k*n+i,c] * U[i*m+j,c] = 1 for all j
            # This is linear in W if we fix U and V
            # For fixed (k,i), the equations are: for each j, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Let A[j,c] = U[i*m+j,c] * V[j*p+k,c], then sum_c A[j,c]*W[k*n+i,c] = 1
            # This is a linear system: A @ W_col = 1 for each (k,i)
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # Build linear system for W[k*n+i, :]
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    # Solve A @ w = ones(m)
                    try:
                        w, residuals, rank_sv, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, residuals, rank_sv, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        u, residuals, rank_sv, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
        
        # After iterations, check if we have a good solution
        if best_err < 1e-8:
            break
    
    if best_solution is None or best_err > 1e-6:
        return None
    
    U, V, W = best_solution
    
    # Try to snap to half-integers
    U_snapped = np.vectorize(_snap_to_half_integer)(U)
    V_snapped = np.vectorize(_snap_to_half_integer)(V)
    W_snapped = np.vectorize(_snap_to_half_integer)(W)
    
    # Check if snapped version works
    if _check_decomposition(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p, tol=1e-6):
        return (U_snapped, V_snapped, W_snapped)
    
    # Check original
    if _check_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
        # Try to find a nearby half-integer solution
        return (U, V, W)
    
    return None

def _try_structural_rank(r, n, m, p):
    """Try specific structural patterns for rank r."""
    # Pattern 1: Use Winograd-style factorization
    # For 3-element dot products, we can sometimes save a multiplication
    
    # Try: for each pair of output rows that share the same B row,
    # combine them
    
    # Actually, let's try a specific known structure
    # For 2x3x3, try rank 15, 14, 13, etc.
    
    # One approach: group by k (column of B)
    # For fixed k, we compute C[:,k] = A @ B[:,k]
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k]
    
    # Sum: (A[0,0]+A[1,0])B[0,k] + (A[0,1]+A[1,1])B[1,k] + (A[0,2]+A[1,2])B[2,k]
    # Diff: (A[0,0]-A[1,0])B[0,k] + (A[0,1]-A[1,1])B[1,k] + (A[0,2]-A[1,2])B[2,k]
    
    # So for each k, we can compute sum and diff with 3+3=6 terms, then
    # C[0,k] = (sum+diff)/2, C[1,k] = (sum-diff)/2
    # But this still uses 6 terms per k, total 18. No gain.
    
    # Try another structure: share across k values
    # For fixed i, C[i,:] = A[i,:] @ B
    # C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0] + A[i,2]B[2,0]
    # C[i,1] = A[i,0]B[0,1] + A[i,1]B[1,1] + A[i,2]B[2,1]
    # C[i,2] = A[i,0]B[0,2] + A[i,1]B[1,2] + A[i,2]B[2,2]
    
    # Group by A[i,j]: for fixed i,j, we have A[i,j] * B[j,0] + A[i,j] * B[j,1] + A[i,j] * B[j,2]
    # = A[i,j] * (B[j,0], B[j,1], B[j,2])
    # This contributes to C[i,0], C[i,1], C[i,2]
    
    # No obvious sharing here either.
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
