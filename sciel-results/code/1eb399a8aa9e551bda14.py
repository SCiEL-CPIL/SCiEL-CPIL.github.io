import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify_exact(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact using the trace convention."""
    rank = len(U[0]) if U else 0
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if this is the matrix multiplication tensor
    # Actually for matrix multiplication, T[i,j,k] should be 1 when we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k in the multiplication tensor
    
    # Build the full tensor and check
    # This is O(n*m*p*rank) which is manageable for small dimensions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    u_val = U[i*m+j][r]
                    v_val = V[j*p+k][r]
                    w_val = W[k*n+i][r]
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer."""
    x2 = 2 * x
    return abs(x2 - round(x2)) < 1e-9

def _to_fraction(x):
    """Convert a float to a Fraction, assuming it's integer or half-integer."""
    x2 = 2 * x
    rounded = round(x2)
    return Fraction(int(rounded), 2)

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Time limit with safety margin
    start_time = time.time()
    time_limit = 130.0  # 20s safety margin from 150s
    
    # Initialize with schoolbook
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_school
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Local search: try to remove columns that are linear combinations of others
    # or find better decompositions through random perturbations
    
    def try_remove_column(U, V, W, col_idx, n, m, p):
        """Try to remove a column by expressing it as combination of others.
        Returns (success, new_U, new_V, new_W)"""
        rank = len(U[0])
        if rank <= 1:
            return False, U, V, W
        
        # The column represents a rank-1 tensor: u_col * v_col * w_col
        # We want to find coefficients c_r such that:
        # u_col[i] * v_col[j] * w_col[k] = sum_{r != col_idx} c_r * U[i][r] * V[j][r] * W[k][r]
        # for all i,j,k
        
        # This is a bilinear/trilinear system. Let's set up the linear system.
        # For each (i,j,k), we have:
        # U[i*m+j][col] * V[j*p+k][col] * W[k*n+i][col] = sum_{r != col_idx} c_r * U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        
        # Since in schoolbook, each column has U=1, V=1, W=1 for specific (i,j,k),
        # the LHS is 1 only for that specific (i,j,k) and 0 otherwise.
        # So we need: for the specific (i0,j0,k0) of this column,
        # 1 = sum_{r != col_idx} c_r * U[i0*m+j0][r] * V[j0*p+k0][r] * W[k0*n+i0][r]
        # And for all other (i,j,k): 0 = sum_{r != col_idx} c_r * U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        
        # Extract the specific (i0, j0, k0) for this column
        # In schoolbook initialization, col corresponds to (i,j,k) where:
        # col = i*m*p + j*p + k
        i0 = col_idx // (m * p)
        j0 = (col_idx % (m * p)) // p
        k0 = col_idx % p
        
        # Build linear system A * c = b
        # Unknowns: c_r for r != col_idx
        # Equations: one for each (i,j,k)
        
        other_cols = [r for r in range(rank) if r != col_idx]
        num_unknowns = len(other_cols)
        num_equations = n * m * p
        
        # A has shape (num_equations, num_unknowns)
        # b has shape (num_equations,)
        A = np.zeros((num_equations, num_unknowns))
        b = np.zeros(num_equations)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for u_idx, r in enumerate(other_cols):
                        A[eq_idx, u_idx] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if i == i0 and j == j0 and k == k0:
                        b[eq_idx] = 1
                    else:
                        b[eq_idx] = 0
                    eq_idx += 1
        
        # Solve least squares
        try:
            c, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            return False, U, V, W
        
        # Check if solution is exact (within tolerance)
        A_c = A @ c
        if not np.allclose(A_c, b, atol=1e-9):
            return False, U, V, W
        
        # Check if c values are integer or half-integer
        for val in c:
            if not _is_half_integer(val):
                return False, U, V, W
        
        # Create new U, V, W without the column, adding c_r to other columns
        new_rank = rank - 1
        new_U = [[0] * new_rank for _ in range(n * m)]
        new_V = [[0] * new_rank for _ in range(m * p)]
        new_W = [[0] * new_rank for _ in range(p * n)]
        
        new_col_idx = 0
        for r in other_cols:
            for i in range(n * m):
                new_U[i][new_col_idx] = U[i][r] + c[new_col_idx] * U[i][col_idx]
            for j in range(m * p):
                new_V[j][new_col_idx] = V[j][r] + c[new_col_idx] * V[j][col_idx]
            for k in range(p * n):
                new_W[k][new_col_idx] = W[k][r] + c[new_col_idx] * W[k][col_idx]
            new_col_idx += 1
        
        # Verify the new decomposition
        if _verify_exact(new_U, new_V, new_W, n, m, p):
            return True, new_U, new_V, new_W
        
        return False, U, V, W
    
    # Try to remove columns one by one
    current_U = [row[:] for row in best_U]
    current_V = [row[:] for row in best_V]
    current_W = [row[:] for row in best_W]
    current_rank = best_rank
    
    # Greedy: try to remove columns in order
    for attempt in range(current_rank):
        if time.time() - start_time > time_limit:
            break
        
        removed = False
        for col_idx in range(current_rank):
            if time.time() - start_time > time_limit:
                break
            
            success, new_U, new_V, new_W = try_remove_column(current_U, current_V, current_W, col_idx, n, m, p)
            if success:
                current_U = new_U
                current_V = new_V
                current_W = new_W
                current_rank -= 1
                removed = True
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = [row[:] for row in current_U]
                    best_V = [row[:] for row in current_V]
                    best_W = [row[:] for row in current_W]
                break
        
        if not removed:
            break
    
    # Random restarts: try different initial decompositions
    if time.time() - start_time < time_limit - 30:
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Start with schoolbook and try random column removals
            U_r = [[0] * rank_school for _ in range(n * m)]
            V_r = [[0] * rank_school for _ in range(m * p)]
            W_r = [[0] * rank_school for _ in range(p * n)]
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_r[i * m + j][col] = 1
                        V_r[j * p + k][col] = 1
                        W_r[k * n + i][col] = 1
                        col += 1
            
            # Try random order of column removal
            cols = list(range(rank_school))
            random.shuffle(cols)
            
            rank_r = rank_school
            for col_idx in cols:
                if time.time() - start_time > time_limit - 5:
                    break
                if col_idx >= rank_r:
                    continue
                    
                # Map original col_idx to current position
                # This is tricky, so let's just try removing from the end
                success, new_U, new_V, new_W = try_remove_column(U_r, V_r, W_r, rank_r - 1, n, m, p)
                if success:
                    U_r = new_U
                    V_r = new_V
                    W_r = new_W
                    rank_r -= 1
                    if rank_r < best_rank:
                        best_rank = rank_r
                        best_U = [row[:] for row in U_r]
                        best_V = [row[:] for row in V_r]
                        best_W = [row[:] for row in W_r]
    
    return {"U": best_U, "V": best_V, "W": best_W}
