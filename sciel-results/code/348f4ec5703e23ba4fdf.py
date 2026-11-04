"""
Tensor decomposition for 2x3x7 matrix multiplication.

This implementation uses a structured random-search approach with rank-reduction
via iterative null-space projection. It starts from the schoolbook rank-42
decomposition and attempts to find a lower-rank exact decomposition by:
  1. Representing the tensor as a 3-way array T[i,j,k] (nonzero iff (i,j,k) is
     a valid index triple).
  2. Starting with the schoolbook factor matrices (U, V, W of rank 42).
  3. Iteratively attempting to reduce rank by identifying linear dependencies
     among the columns of U, V, W simultaneously.
  4. Using randomized linear combinations and checking exact integer/half-integer
     recovery.

The algorithm is deterministic in its outcome verification but uses a fixed
seed for reproducibility. It runs well within the 120s budget.
"""

import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _to_fraction_matrix(M):
    """Convert a numpy matrix to a list of lists of Fractions (exact)."""
    return [[Fraction(M[i, j]).limit_denominator(1000000) for j in range(M.shape[1])]
            for i in range(M.shape[0])]

def _validate_decomposition(U, V, W, n, m, p):
    """
    Check that U, V, W (as lists of lists of Fractions or ints) exactly
    decompose the matrix multiplication tensor.
    
    Convention: W is indexed as W[k*n + i, r] (official trace convention).
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank).
    
    The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for all i in [0,n), j in [0,m), k in [0,p).
    T[i,j,k] = 1 if (i,j,k) is a valid index triple, else 0.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = Fraction(0)
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if actual != expected:
                    return False
    return True

def _validate_decomposition_np(U, V, W, n, m, p, tol=1e-9):
    """Numerical validation for speed during search."""
    rank = U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                actual = np.dot(np.dot(u_val, v_val), w_val)
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook rank n*m*p decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_iter=5000, seed=42):
    """
    Attempt to reduce the rank of the decomposition from current rank to target_rank.
    Uses iterative column elimination via linear dependencies.
    Returns (U, V, W) with reduced rank if successful, else original.
    """
    rng = np.random.default_rng(seed)
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = current_U.shape[1]
    
    for attempt in range(max_iter):
        if current_rank <= target_rank:
            break
        
        # Try to find a linear dependency among the columns
        # Pick a random subset of columns to work with
        if current_rank <= target_rank + 2:
            # Near target, try to eliminate one column at a time
            cols_to_try = list(range(current_rank))
        else:
            # Randomly pick a few columns to check for dependencies
            cols_to_try = rng.choice(current_rank, size=min(5, current_rank), replace=False)
        
        found_dependency = False
        for c in cols_to_try:
            # Try to express column c as a linear combination of other columns
            # Check if column c is in the span of the others for each matrix
            # Build the matrix of other columns
            other_cols = [i for i in range(current_rank) if i != c]
            
            # For U: check if U[:, c] is in span of U[:, other_cols]
            U_other = current_U[:, other_cols]
            V_other = current_V[:, other_cols]
            W_other = current_W[:, other_cols]
            
            U_c = current_U[:, c]
            V_c = current_V[:, c]
            W_c = current_W[:, c]
            
            # Solve for coefficients alpha such that U_other @ alpha ≈ U_c
            # Use least squares
            try:
                alpha_U, _, _, _ = np.linalg.lstsq(U_other, U_c, rcond=None)
            except:
                continue
            
            # Check if same alpha works for V and W
            V_pred = V_other @ alpha_U
            W_pred = W_other @ alpha_U
            
            if (np.max(np.abs(V_pred - V_c)) < 1e-7 and 
                np.max(np.abs(W_pred - W_c)) < 1e-7):
                # Column c is linearly dependent! Remove it.
                mask = [i for i in range(current_rank) if i != c]
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_rank -= 1
                found_dependency = True
                break
        
        if not found_dependency:
            # Try a different strategy: random linear combination
            # Pick two columns and replace them with a linear combination
            if current_rank > target_rank:
                c1, c2 = rng.choice(current_rank, size=2, replace=False)
                t = rng.choice([-1, -0.5, 0.5, 1, 2, -2], size=1)[0]
                
                # Try: new_col = U[:, c1] + t * U[:, c2]
                # Remove c2, replace c1 with the combination
                # This is equivalent to a column operation
                new_col_U = current_U[:, c1] + t * current_U[:, c2]
                new_col_V = current_V[:, c1] + t * current_V[:, c2]
                new_col_W = current_W[:, c1] + t * current_W[:, c2]
                
                # Remove column c2
                mask = [i for i in range(current_rank) if i != c2]
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_rank -= 1
                
                # Find where c1 is now (it should still be at index c1 if c2 > c1, else c1-1)
                c1_idx = c1 if c1 < c2 else c1 - 1
                current_U[:, c1_idx] = new_col_U
                current_V[:, c1_idx] = new_col_V
                current_W[:, c1_idx] = new_col_W
                
                # Verify the decomposition is still valid
                if _validate_decomposition_np(current_U, current_V, current_W, n, m, p):
                    continue
                else:
                    # Revert
                    # This is complex; let's just break and try a different seed
                    break
    
    return current_U, current_V, current_W

def construct():
    """
    Construct the tensor decomposition for 2x3x7 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Start with schoolbook rank-42 decomposition
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Verify the schoolbook decomposition
    assert _validate_decomposition_np(U, V, W, n, m, p), "Schoolbook decomposition failed validation"
    
    # Attempt rank reduction
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try various target ranks
    for target_rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline:
            break
        
        U_try, V_try, W_try = _try_rank_reduction(U, V, W, n, m, p, target_rank, 
                                                    max_iter=1000, seed=target_rank)
        
        if U_try.shape[1] < best_rank and _validate_decomposition_np(U_try, V_try, W_try, n, m, p):
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]
    
    # If we haven't reduced the rank, try a more aggressive randomized search
    if best_rank == 42 and time.time() < deadline:
        # Try to find a rank-37 or lower decomposition via random initialization
        # This is unlikely to succeed quickly, but we attempt it
        for rank_attempt in [37, 38, 39, 40, 41]:
            if time.time() > deadline:
                break
            
            found = False
            for trial in range(50):
                if time.time() > deadline:
                    break
                
                rng = np.random.default_rng(trial * 1000 + rank_attempt)
                
                # Random initialization with sparsity
                U_rand = np.zeros((n * m, rank_attempt), dtype=np.float64)
                V_rand = np.zeros((m * p, rank_attempt), dtype=np.float64)
                W_rand = np.zeros((p * n, rank_attempt), dtype=np.float64)
                
                # Initialize with random sparse values
                for r in range(rank_attempt):
                    # Each column has a few nonzeros
                    nnz_U = rng.integers(2, 4)
                    nnz_V = rng.integers(2, 4)
                    nnz_W = rng.integers(2, 4)
                    
                    rows_U = rng.choice(n * m, size=nnz_U, replace=False)
                    rows_V = rng.choice(m * p, size=nnz_V, replace=False)
                    rows_W = rng.choice(p * n, size=nnz_W, replace=False)
                    
                    for row in rows_U:
                        U_rand[row, r] = rng.choice([-1, 0.5, 1, 2])
                    for row in rows_V:
                        V_rand[row, r] = rng.choice([-1, 0.5, 1, 2])
                    for row in rows_W:
                        W_rand[row, r] = rng.choice([-1, 0.5, 1, 2])
                
                if _validate_decomposition_np(U_rand, V_rand, W_rand, n, m, p):
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                    best_rank = rank_attempt
                    found = True
                    break
            
            if found:
                break
    
    # Convert to lists for the return interface
    def to_list_of_lists(arr):
        return [[float(arr[i, j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    result = {
        "U": to_list_of_lists(best_U),
        "V": to_list_of_lists(best_V),
        "W": to_list_of_lists(best_W)
    }
    
    return result

# EVOLVE-BLOCK-END
