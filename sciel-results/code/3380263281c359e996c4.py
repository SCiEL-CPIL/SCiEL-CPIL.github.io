"""Matrix multiplication 3x4x5 - rank-minimal decomposition search."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np
import random
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

# EVOLVE-BLOCK-START

def _build_system(V, W, n, m, p, R):
    """
    Build the linear system A * vec(U) = b for fixed V, W.
    U has shape (n*m, R).
    Equation: sum_r U[x,r] V[y,r] W[z,r] = T[x,y,z]
    """
    nx = n * m
    ny = m * p
    nz = p * n
    
    # A is (nx * ny * nz) x (nx * R)
    # We only need to build the non-zeros.
    # Row index: (x, y, z) -> x * ny * nz + y * nz + z
    # Col index: (x, r) -> x * R + r
    
    # We can construct A as a dense matrix if size is small, or sparse.
    # nx*ny*nz = 12*20*15 = 3600
    # nx*R = 12*R. If R=60, 720.
    # 3600x720 is ~2.5M entries. Dense is fine.
    
    A = np.zeros((nx * ny * nz, nx * R))
    b = np.zeros(nx * ny * nz)
    
    # Precompute target T
    T = np.zeros((nx, ny, nz))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                T[x, y, z] = 1.0
    b = T.flatten()
    
    # Fill A
    # A[row, col] = V[y, r] * W[z, r] if x matches
    # row = x * ny * nz + y * nz + z
    # col = x * R + r
    
    # Vectorized fill
    # Create indices
    # For each x, we have a block of rows.
    # Within that block, for each y, z, the row is x*ny*nz + y*nz + z.
    # The non-zeros are in columns x*R .. x*R+R-1.
    # Values are V[y, :] * W[z, :]
    
    for x in range(nx):
        row_start = x * ny * nz
        row_end = row_start + ny * nz
        col_start = x * R
        col_end = col_start + R
        
        # We need to fill A[row_start:row_end, col_start:col_end]
        # This submatrix has shape (ny*nz, R)
        # Entry (y*nz+z, r) = V[y, r] * W[z, r]
        
        # Construct outer product sum?
        # V is (ny, R), W is (nz, R)
        # We want a matrix M of size (ny*nz, R)
        # M[y*nz+z, r] = V[y, r] * W[z, r]
        
        # Reshape V to (ny, 1, R) and W to (1, nz, R)
        # Broadcast to (ny, nz, R)
        # Then reshape to (ny*nz, R)
        
        V_exp = V[:, np.newaxis, :] # (ny, 1, R)
        W_exp = W[np.newaxis, :, :] # (1, nz, R)
        M = (V_exp * W_exp).reshape(ny * nz, R)
        
        A[row_start:row_end, col_start:col_end] = M
        
    return A, b

def _solve_U(A, b, R, nx):
    """Solve A * vec(U) = b using least squares."""
    U_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    U = U_flat.reshape(nx, R)
    return U, np.sum(residuals**2) if len(residuals) > 0 else 0.0

def _round_to_half_int(U):
    """Round U to nearest half-integer."""
    return np.round(U * 2) / 2.0

def _compute_error_int(U, V, W, T):
    """Compute error with integer U."""
    D = np.einsum('xr,yr,zr->xyz', U, V, W)
    return np.sum((D - T) ** 2)

def _search_rank(n, m, p, R, time_limit):
    """
    Search for a valid decomposition of rank R.
    """
    start_time = time.time()
    nx = n * m
    ny = m * p
    nz = p * n
    
    # Target tensor
    T = np.zeros((nx, ny, nz))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                T[x, y, z] = 1.0
                
    # Initialize V, W
    # Start from schoolbook and truncate/perturb
    V = np.zeros((ny, R))
    W = np.zeros((nz, R))
    
    # Fill first min(R, n*m*p) columns with schoolbook pattern
    cols_to_use = min(R, n * m * p)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= cols_to_use:
                    break
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Randomize remaining columns
    for c in range(cols_to_use, R):
        V[:, c] = np.random.choice([-1.0, 0.0, 1.0], size=ny)
        W[:, c] = np.random.choice([-1.0, 0.0, 1.0], size=nz)
        
    # Build initial system
    A, b = _build_system(V, W, n, m, p, R)
    U_real, err_real = _solve_U(A, b, R, nx)
    U_int = _round_to_half_int(U_real)
    err_int = _compute_error_int(U_int, V, W, T)
    
    if err_int < 1e-9:
        return {"U": U_int.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    best_err = err_int
    best_sol = (U_int.copy(), V.copy(), W.copy())
    
    max_iterations = 5000
    no_improvement = 0
    
    while time.time() - start_time < time_limit and no_improvement < 200:
        # Pick a matrix to perturb: V or W
        if random.random() < 0.5:
            mat = V
            mat_name = 'V'
            rows = ny
        else:
            mat = W
            mat_name = 'W'
            rows = nz
            
        r = random.randint(0, rows - 1)
        c = random.randint(0, R - 1)
        
        old_val = mat[r, c]
        # Try a random new value
        new_val = np.random.choice([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        if new_val == old_val:
            continue
            
        mat[r, c] = new_val
        
        # Rebuild system (expensive, but R is small)
        A, b = _build_system(V, W, n, m, p, R)
        U_real, err_real = _solve_U(A, b, R, nx)
        U_int = _round_to_half_int(U_real)
        err_int = _compute_error_int(U_int, V, W, T)
        
        if err_int < best_err - 1e-9:
            best_err = err_int
            best_sol = (U_int.copy(), V.copy(), W.copy())
            no_improvement = 0
            if err_int < 1e-9:
                return {"U": U_int.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Revert if not improved (or accept with prob if simulated annealing)
            if random.random() > 0.01:
                mat[r, c] = old_val
            else:
                # Keep the worse move, update best if it's the best so far? No, just keep state
                pass
            no_improvement += 1
            
    if best_err < 1e-9:
        U_b, V_b, W_b = best_sol
        return {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
        
    return None

def _generate_schoolbook(n, m, p):
    """Generate the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
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

def construct():
    """
    Constructs the minimal rank tensor decomposition for 3x4 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safe margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try ranks from schoolbook-1 downwards
    # Allocate time to each rank
    # We will try 59, 58, 57, 56, 55
    ranks_to_try = [schoolbook_rank - 1, schoolbook_rank - 2, schoolbook_rank - 3, schoolbook_rank - 4, schoolbook_rank - 5]
    
    time_per_rank = 15.0
    
    for R in ranks_to_try:
        if time.time() > deadline - time_per_rank:
            break
        res = _search_rank(n, m, p, R, time_per_rank)
        if res is not None:
            return res
            
    # Fallback to schoolbook
    return _generate_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
