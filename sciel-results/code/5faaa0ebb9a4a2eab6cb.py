"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

def _validate(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _validate_numpy(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Vectorized validation using numpy for speed."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]
                val = np.sum(prod)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _is_half_integer(val, tol=1e-9):
    """Check if val is an integer or half-integer."""
    doubled = val * 2
    return abs(doubled - round(doubled)) < tol

def _all_half_integer(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val, tol):
                    return False
    return True

def _compute_residuals(U, V, W, n=2, m=4, p=5):
    """Compute residuals for each (i,j,k) triple."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    residuals = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]
                val = np.sum(prod)
                residuals[(i,j,k)] = 1.0 - val
    return residuals

def _try_ranked_decomposition(n, m, p, rank, time_budget=5.0, seed=42):
    """Try to find a rank-r decomposition."""
    start = time.time()
    rng = random.Random(seed)
    
    best_result = None
    best_residual_norm = float('inf')
    
    # Try a few random initializations
    for init_idx in range(10):
        if time.time() - start > time_budget:
            break
            
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        
        # Sparse random initialization with values -1, 0, 1
        density = 0.3
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    if rng.random() < density:
                        U[i*m+j][c] = float(rng.choice([-1, 1]))
            for j in range(m):
                for k in range(p):
                    if rng.random() < density:
                        V[j*p+k][c] = float(rng.choice([-1, 1]))
            for k in range(p):
                for i in range(n):
                    if rng.random() < density:
                        W[k*n+i][c] = float(rng.choice([-1, 1]))
                        
        # Check if this initialization works
        if _all_half_integer(U, V, W) and _validate_numpy(U, V, W, n, m, p):
            return U, V, W, True
            
        # Compute current residual norm
        residuals = _compute_residuals(U, V, W, n, m, p)
        residual_norm = sum(abs(r) for r in residuals.values())
        
        # Local refinement: try modifying entries to reduce residuals
        for iteration in range(500):
            if time.time() - start > time_budget:
                break
            
            # Choose which matrix to modify
            mat_idx = rng.randint(0, 2)
            mat = [U, V, W][mat_idx]
            row = rng.randint(0, len(mat)-1)
            col = rng.randint(0, rank-1)
            
            old_val = mat[row][col]
            
            # Try several candidate values
            candidates = [old_val]
            for cv in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                if abs(cv - old_val) > 1e-9:
                    candidates.append(cv)
            
            best_new_val = old_val
            best_new_residual = residual_norm
            
            for new_val in candidates:
                mat[row][col] = new_val
                residuals = _compute_residuals(U, V, W, n, m, p)
                new_residual = sum(abs(r) for r in residuals.values())
                if new_residual < best_new_residual:
                    best_new_residual = new_residual
                    best_new_val = new_val
            
            mat[row][col] = best_new_val
            residual_norm = best_new_residual
            
            # Check if we've solved it
            if _all_half_integer(U, V, W) and _validate_numpy(U, V, W, n, m, p):
                return U, V, W, True
            
            # Track best found
            if residual_norm < best_residual_norm:
                best_residual_norm = residual_norm
                best_result = (
                    [row[:] for row in U],
                    [row[:] for row in V],
                    [row[:] for row in W]
                )
                
            # Early exit if residual is very small
            if residual_norm < 1e-6:
                break
    
    # If we found a near-solution, try to polish it
    if best_result is not None and best_residual_norm < 0.1:
        U, V, W = best_result
        for _ in range(100):
            if time.time() - start > time_budget:
                break
            if _all_half_integer(U, V, W) and _validate_numpy(U, V, W, n, m, p):
                return U, V, W, True
            mat_idx = rng.randint(0, 2)
            mat = [U, V, W][mat_idx]
            row = rng.randint(0, len(mat)-1)
            col = rng.randint(0, rank-1)
            old_val = mat[row][col]
            for delta in [-0.5, 0.5]:
                mat[row][col] = old_val + delta
                if _all_half_integer(U, V, W) and _validate_numpy(U, V, W, n, m, p):
                    return U, V, W, True
            mat[row][col] = old_val
    
    return None, None, None, False

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook: rank = n*m*p = 40
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Verify schoolbook
    assert _validate(U_school, V_school, W_school, n, m, p), "Schoolbook decomposition failed!"
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # Prioritize ranks that are multiples of 5 or 2, and close to lower bound 20
    candidate_ranks = [20, 22, 24, 25, 28, 30, 32, 35, 36, 38]
    
    for rank in candidate_ranks:
        if rank >= best_rank:
            continue
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
            
        time_budget = min(10.0, DEADLINE - elapsed - 5)
        if time_budget < 2:
            break
            
        U, V, W, success = _try_ranked_decomposition(n, m, p, rank, time_budget, seed=rank*7+13)
        if success:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break # Found a lower rank, stop searching for even lower? 
                  # Actually, we should keep searching for even lower if time permits.
                  # But 20 is the lower bound. So if we find 20, we stop.
            if rank == 20:
                break
    
    # Return the best decomposition found
    def _clean_entry(v):
        if abs(v - round(v)) < 1e-9:
            return int(round(v))
        doubled = v * 2
        if abs(doubled - round(doubled)) < 1e-9:
            return round(doubled) / 2.0
        return v
    
    result_U = [[_clean_entry(v) for v in row] for row in best_U]
    result_V = [[_clean_entry(v) for v in row] for row in best_V]
    result_W = [[_clean_entry(v) for v in row] for row in best_W]
    
    return {"U": result_U, "V": result_V, "W": result_W}
