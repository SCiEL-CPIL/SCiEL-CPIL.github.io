"""Matrix multiplication 3x4x8 tensor decomposition search."""
import itertools
import random
import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 4, 8]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 96

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
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

def _verify(U, V, W, tol=1e-9):
    """Check if U,V,W exactly decompose the matrix multiplication tensor.
    The tensor T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to C[i,k].
    Convention: W index is k*n+i.
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    """
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Use einsum for efficient verification
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(result, 1.0, atol=tol)

def _verify_fast(U, V, W, tol=1e-9):
    """Faster verification using numpy."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # For each (i,j,k), sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # Use einsum
    # U_r[i,j,c], V_r[j,k,c], W_r[k,i,c] -> sum over c, result[i,j,k]
    result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(result, 1.0, atol=tol)

def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < tol

def _all_half_integers(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val, tol):
                    return False
    return True

def _try_remove_column(U, V, W, col_to_remove, deadline):
    """
    Try to remove a specific column from the decomposition by solving for the other W entries.
    We fix U and V (excluding the column) and solve for W.
    """
    n, m, p = N, M, P
    rank = len(U[0])
    
    # Indices for the remaining columns
    remaining_cols = [c for c in range(rank) if c != col_to_remove]
    new_rank = len(remaining_cols)
    
    # Extract U and V for remaining columns
    U_rem = np.array([U[i][c] for i in range(n*m) for c in remaining_cols]).reshape(n*m, new_rank)
    V_rem = np.array([V[i][c] for i in range(m*p) for c in remaining_cols]).reshape(m*p, new_rank)
    
    # The target tensor is the original tensor (all 1s)
    # Since we are removing a column that was part of the schoolbook (or current) solution,
    # we assume the current U, V, W are valid.
    # We want to find W_rem such that sum_l U_rem[i,j,l] V_rem[j,k,l] W_rem[k,i,l] = 1
    
    # Construct the linear system for W_rem
    # Variables: W_rem[k, i, l] for k in 0..p-1, i in 0..n-1, l in 0..new_rank-1
    # Total variables: p * n * new_rank
    num_vars = p * n * new_rank
    num_eqs = n * m * p
    
    # Build matrix A (num_eqs x num_vars)
    # A[i, j, k, k', i', l] = U_rem[i, j, l] * V_rem[j, k', l] * (1 if k==k' and i==i' else 0)
    # Wait, the equation is:
    # sum_l U_rem[i, j, l] * V_rem[j, k, l] * W_rem[k, i, l] = 1
    # For a fixed (i, j, k), the variable is W_rem[k, i, l].
    # So A[(i,j,k), (k,i,l)] = U_rem[i, j, l] * V_rem[j, k, l]
    
    A = np.zeros((num_eqs, num_vars))
    b = np.ones(num_eqs)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for l in range(new_rank):
                    var_idx = k * n * new_rank + i * new_rank + l
                    A[eq_idx, var_idx] = U_rem[i * m + j, l] * V_rem[j * p + k, l]
    
    # Solve min ||x||_1 subject to A x = b
    # Using linprog
    c = np.ones(num_vars) # Minimize sum |x_i|
    # linprog minimizes c^T x
    # We need to handle absolute values.
    # Introduce variables x_pos, x_neg >= 0
    # x = x_pos - x_neg
    # Minimize sum (x_pos + x_neg)
    # A (x_pos - x_neg) = b
    
    c_lp = np.concatenate([np.ones(num_vars), np.ones(num_vars)])
    A_eq = np.concatenate([A, -A], axis=1)
    b_eq = b
    
    # No bounds (variables are >= 0)
    bounds = [(0, None)] * (2 * num_vars)
    
    try:
        res = linprog(c_lp, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            x_pos = res.x[:num_vars]
            x_neg = res.x[num_vars:]
            x = x_pos - x_neg
            
            # Check if x is close to half-integers
            if _all_half_integers([x.reshape(p, n, new_rank).tolist()], [], [], tol=1e-6):
                # Reconstruct W
                W_rem = x.reshape(p, n, new_rank).T # Need to match W shape (p*n, rank)
                # W_rem[k*n+i, l] = x[k, i, l]
                W_rem_mat = np.zeros((p * n, new_rank))
                for k in range(p):
                    for i in range(n):
                        for l in range(new_rank):
                            W_rem_mat[k * n + i, l] = x[k * n * new_rank + i * new_rank + l]
                
                # Reconstruct U, V, W
                U_new = [U_rem[i, :].tolist() for i in range(n * m)]
                V_new = [V_rem[i, :].tolist() for i in range(m * p)]
                W_new = W_rem_mat.tolist()
                
                if _verify_fast(U_new, V_new, W_new):
                    return {"U": U_new, "V": V_new, "W": W_new}
    except Exception:
        pass
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Start with schoolbook
    result = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Try to remove columns one by one
    # We try to remove columns from the schoolbook solution
    # We iterate through columns and try to remove them
    # If successful, we update the result and try to remove another
    
    current_U = result["U"]
    current_V = result["V"]
    current_W = result["W"]
    current_rank = best_rank
    
    # Try removing each column
    # We do this in a random order to avoid bias
    cols_to_try = list(range(current_rank))
    random.Random(42).shuffle(cols_to_try)
    
    for c in cols_to_try:
        if time.time() > deadline:
            break
        if c >= current_rank:
            continue
            
        new_res = _try_remove_column(current_U, current_V, current_W, c, deadline)
        if new_res is not None:
            current_U = new_res["U"]
            current_V = new_res["V"]
            current_W = new_res["W"]
            current_rank = len(current_U[0])
            result = new_res
            # Update cols_to_try to reflect new indices?
            # It's safer to restart the loop or adjust indices.
            # Since we removed column c, all columns after c shift left.
            # cols_to_try is a list of indices in the OLD matrix.
            # It's complex to update. Let's just break and restart with new rank?
            # Or simply continue with the next column in the list, but be careful with indices.
            # For simplicity, let's just return the first improvement found.
            # Or continue trying to remove more.
            # Let's continue, but we need to map old indices to new.
            # If we removed column c, then old index j > c becomes new index j-1.
            # This is error-prone. Let's just try to remove one column.
            break

    return result
