"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    # Build tensor from U, V, W: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Expected: T[i,j,k] = 1 for all valid i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _snap_to_half_integers(matrix):
    """Snap values to nearest half-integer (x.0 or x.5)."""
    snapped = []
    for row in matrix:
        new_row = []
        for val in row:
            # Round to nearest 0.5
            snapped_val = round(val * 2) / 2.0
            new_row.append(snapped_val)
        snapped.append(new_row)
    return snapped

def _compute_error(U, V, W, n, m, p):
    """Compute the maximum absolute error of the decomposition."""
    rank = len(U[0])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_random_decomposition(n, m, p, target_rank, num_trials=100, time_limit=100):
    """Try to find a decomposition of the given rank using random search."""
    start_time = time.time()
    best_error = float('inf')
    best_result = None
    
    for trial in range(num_trials):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization with small values
        U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p*n)]
        
        # Try local search: iteratively adjust one entry at a time
        for iteration in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Compute current error
            err = _compute_error(U, V, W, n, m, p)
            if err < 1e-9:
                # Found exact solution
                U_snapped = _snap_to_half_integers(U)
                V_snapped = _snap_to_half_integers(V)
                W_snapped = _snap_to_half_integers(W)
                if _validate(U_snapped, V_snapped, W_snapped, n, m, p):
                    return U_snapped, V_snapped, W_snapped
                else:
                    # Snapping broke it, try without snapping
                    return U, V, W
            
            if err < best_error:
                best_error = err
                best_result = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
            
            # Pick a random entry to perturb
            mat_choice = random.random()
            if mat_choice < 0.33:
                mat = U
                row_idx = random.randint(0, n*m-1)
            elif mat_choice < 0.66:
                mat = V
                row_idx = random.randint(0, m*p-1)
            else:
                mat = W
                row_idx = random.randint(0, p*n-1)
            
            col_idx = random.randint(0, target_rank-1)
            old_val = mat[row_idx][col_idx]
            
            # Try several perturbations
            best_local_val = old_val
            best_local_err = err
            
            for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                mat[row_idx][col_idx] = old_val + delta
                new_err = _compute_error(U, V, W, n, m, p)
                if new_err < best_local_err:
                    best_local_err = new_err
                    best_local_val = old_val + delta
            
            mat[row_idx][col_idx] = best_local_val
    
    # Return best found (may not be exact)
    if best_result and best_error < 1e-6:
        U, V, W = best_result
        U_snapped = _snap_to_half_integers(U)
        V_snapped = _snap_to_half_integers(V)
        W_snapped = _snap_to_half_integers(W)
        if _validate(U_snapped, V_snapped, W_snapped, n, m, p):
            return U_snapped, V_snapped, W_snapped
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # First, create the schoolbook decomposition as fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Try to find lower rank decompositions
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # But actual minimal rank for 2x3 by 3x4 is unknown; try from 12 upward
    
    for target_rank in range(12, schoolbook_rank):
        if time.time() > deadline:
            break
        
        result = _try_random_decomposition(n, m, p, target_rank, num_trials=50, time_limit=30)
        if result is not None:
            U, V, W = result
            # Verify it's valid
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
