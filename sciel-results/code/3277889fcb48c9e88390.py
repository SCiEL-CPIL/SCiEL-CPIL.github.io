"""
Matrix multiplication 3,4,5 - Rank-minimal tensor decomposition search.
Uses a fix-U-V-solve-W linear system approach to find low-rank decompositions.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

def _schoolbook(n, m, p):
    """
    Standard schoolbook matrix multiplication tensor decomposition.
    Rank = n * m * p.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    Indexing:
    U[i*m+j, c]
    V[j*p+k, c]
    W[k*n+i, c]
    """
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

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _is_half_integer(x, tol=1e-9):
    """Check if x is a half-integer (i.e., 2*x is an integer)."""
    y = 2.0 * x
    return abs(y - round(y)) < tol

def _try_solve_for_W(U, V, n, m, p, rank):
    """
    Given fixed U and V, solve the linear system for W.
    The equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k}
    for all i in [0,n), j in [0,m), k in [0,p).
    
    We set up a linear system: for each (i,j,k), we have an equation in the variables W[k*n+i, c].
    But note that W[k*n+i, c] appears with coefficient U[i*m+j, c] * V[j*p+k, c].
    
    Actually, the equation for a specific (i,j,k) is:
    sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
    
    This is linear in the variables W[*, c] but the coefficient for W[k*n+i, c] in the (i,j,k) equation
    is U[i*m+j, c] * V[j*p+k, c].
    
    However, W[k*n+i, c] also appears in other equations. Let's think about this differently.
    
    Let's index the unknowns as W_row[c] where W_row = k*n+i for some k, i.
    For a given (i,j,k), the equation involves W[k*n+i, c] for all c.
    So for each (i,j,k), we have one equation involving the rank variables W[k*n+i, 0], ..., W[k*n+i, rank-1].
    
    But W[k*n+i, c] also appears in equations for other (i', j', k') where k'*n+i' = k*n+i, i.e., k'=k and i'=i.
    So W[k*n+i, c] only appears in equations where the output index is (k,i), which means i'=i and k'=k.
    For those equations, j' can be anything.
    
    So for a fixed (k,i), the equations involving W[k*n+i, c] are:
    For each j in [0,m): sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
    
    This gives m equations for each (k,i) pair, with rank unknowns W[k*n+i, 0..rank-1].
    Since m < rank in general (m=4, rank=59), we have an underdetermined system for each (k,i).
    
    Total equations: n*m*p = 60
    Total unknowns: p*n*rank = 15*59 = 885
    
    This is highly underdetermined. We can use least squares to find a solution, then check if it's half-integer.
    """
    # Build the linear system: A @ x = b
    # where x is the flattened W matrix of size (p*n*rank,)
    # Each equation corresponds to a triplet (i,j,k)
    
    num_eqs = n * m * p
    num_unk = p * n * rank
    
    A = np.zeros((num_eqs, num_unk))
    b = np.ones(num_eqs)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    # The unknown W[k*n+i, c] has index (k*n+i)*rank + c in the flattened array
                    unk_idx = (k * n + i) * rank + c
                    coeff = U[i*m+j][c] * V[j*p+k][c]
                    A[eq_idx, unk_idx] += coeff
                eq_idx += 1
    
    # Solve using least squares
    try:
        W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
    except:
        return None
    
    # Reshape W_flat to (p*n, rank)
    W_sol = W_flat.reshape((p*n, rank))
    
    # Check if all entries are half-integers
    for ii in range(p*n):
        for c in range(rank):
            if not _is_half_integer(W_sol[ii, c]):
                return None
    
    # Convert to list of lists with exact half-integer values
    W_out = [[0] * rank for _ in range(p*n)]
    for ii in range(p*n):
        for c in range(rank):
            W_out[ii][c] = W_sol[ii, c]
    
    return W_out

def _random_sparse_matrix(rows, rank, nonzeros_per_row, val_range):
    """Generate a random sparse matrix with given properties."""
    M = [[0] * rank for _ in range(rows)]
    for r in range(rows):
        nnz = max(1, min(nonzeros_per_row, rank))
        cols = np.random.choice(rank, size=nnz, replace=False)
        for c in cols:
            M[r][c] = np.random.choice(val_range)
    return M

def _try_find_low_rank(n, m, p, start_rank, max_rank, time_limit):
    """
    Try to find a rank-r decomposition for r in [start_rank, max_rank].
    Uses random sparse U, V and solves for W.
    """
    deadline = time.time() + time_limit
    best_result = None
    best_rank = None
    
    for rank in range(start_rank, max_rank + 1):
        if time.time() > deadline - 5:
            break
        
        max_attempts = 2000
        attempts = 0
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Try different sparsity levels
            for nnz_per_row in [1, 2, 3]:
                if time.time() > deadline - 5:
                    break
                
                # Random U and V
                val_range = [-2, -1, 1, 2]
                U = _random_sparse_matrix(n*m, rank, nnz_per_row, val_range)
                V = _random_sparse_matrix(m*p, rank, nnz_per_row, val_range)
                
                # Solve for W
                W = _try_solve_for_W(U, V, n, m, p, rank)
                
                if W is not None:
                    # Verify the full decomposition
                    if _verify_decomposition(U, V, W, n, m, p, rank):
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        return best_result, best_rank
        
        # If no solution found for this rank, try next
        continue
    
    return best_result, best_rank

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill switch
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # 1. Establish the safe baseline (Schoolbook)
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    # 2. Try to find a lower-rank decomposition
    # Known lower bound for 3x4x5 matrix multiplication rank is around 56-58
    # Try from rank 50 up to 59
    
    min_rank_to_try = 50
    max_rank_to_try = base_rank - 1  # 59
    
    result, rank_found = _try_find_low_rank(n, m, p, min_rank_to_try, max_rank_to_try, deadline - start_time - 10)
    
    if result is not None and rank_found < best_rank:
        best = result
        best_rank = rank_found
    
    # Check time to ensure we don't accidentally timeout
    if time.time() > deadline - 5.0:
        return best

    return best
# EVOLVE-BLOCK-END
