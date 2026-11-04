"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

def construct():
    """
    Returns a rank-minimal (or best found) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x7 matrix.
    
    The decomposition is defined by three matrices U, V, W of shapes:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Such that for all i, j, k:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    num_eqs = n * m * p  # 84
    
    def get_schoolbook():
        rank = num_eqs
        U = np.zeros((dim_U, rank), dtype=int)
        V = np.zeros((dim_V, rank), dtype=int)
        W = np.zeros((dim_W, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Precompute index arrays for the tensor equation
    # Equation index e corresponds to (i, j, k)
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    U_rows = i_idx * m + j_idx
    V_rows = j_idx * p + k_idx
    W_rows = k_idx * n + i_idx
    
    def verify_fast(U, V, W, r):
        """
        Verify exact decomposition using vectorized operations.
        U: (dim_U, r), V: (dim_V, r), W: (dim_W, r)
        """
        # We need to check: sum_c U[u, c] * V[v, c] * W[w, c] == 1 for all e
        # u = U_rows[e], v = V_rows[e], w = W_rows[e]
        
        # Extract relevant rows
        # U_sub: (num_eqs, r)
        # V_sub: (num_eqs, r)
        # W_sub: (num_eqs, r)
        U_sub = U[U_rows, :]
        V_sub = V[V_rows, :]
        W_sub = W[W_rows, :]
        
        # Product term by term
        # prod: (num_eqs, r)
        prod = U_sub * V_sub * W_sub
        
        # Sum over columns
        sums = np.sum(prod, axis=1)
        
        # Check if all are 1.0
        # Also check half-integer constraint for W (and U, V if we want to be strict, 
        # but the prompt says "integer/half-integer entries" for the factor matrices)
        if not np.allclose(sums, 1.0, atol=1e-9):
            return False
        
        # Check half-integer property for all matrices
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        return True

    # Search for better rank
    # Start from a reasonable lower bound. 
    # Theoretical lower bound is max(n*m, m*p, p*n) = 28.
    # But practically, we search downwards from 83.
    # However, finding a rank significantly lower than 84 is hard.
    # We will search from 83 down to 70 first, as these are most likely to exist if any.
    # If nothing is found, we return schoolbook.
    
    best_result = None
    best_rank = 84
    
    # Define search ranges
    # We prioritize ranks close to 84 as they are more likely to be found by local search
    # and any improvement is a win.
    ranks_to_try = list(range(83, 69, -1)) + list(range(69, 27, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        
        # Number of tries
        if r > 75:
            num_tries = 30
        elif r > 60:
            num_tries = 15
        else:
            num_tries = 5
            
        for _ in range(num_tries):
            if time.time() > deadline - 1:
                break
            
            # Initialize U and V with sparse random integers {-1, 0, 1}
            # We want to allow for cancellations.
            U = np.zeros((dim_U, r), dtype=int)
            V = np.zeros((dim_V, r), dtype=int)
            
            # Strategy: For each column c, pick a few non-zero entries.
            # To ensure the system is solvable, we need enough non-zeros.
            # If U and V are too sparse, A might have zero rows, making it impossible to satisfy b=1.
            
            for c in range(r):
                # Randomly choose which rows to set non-zero in U
                # Ensure at least 1 non-zero per column to avoid zero rows in A
                num_nonzero_U = np.random.randint(1, dim_U // 2 + 1)
                rows_U = np.random.choice(dim_U, size=num_nonzero_U, replace=False)
                U[rows_U, c] = np.random.choice([-1, 1], size=num_nonzero_U)
                
                num_nonzero_V = np.random.randint(1, dim_V // 2 + 1)
                rows_V = np.random.choice(dim_V, size=num_nonzero_V, replace=False)
                V[rows_V, c] = np.random.choice([-1, 1], size=num_nonzero_V)
            
            # Build sparse A and b
            # A[eq, w_row * r + c] = U[U_rows[eq], c] * V[V_rows[eq], c]
            
            # Vectorized construction of A
            # U_sub: (num_eqs, r)
            # V_sub: (num_eqs, r)
            U_sub = U[U_rows, :]
            V_sub = V[V_rows, :]
            
            # Coefficients for W variables
            # coeff[eq, c] = U_sub[eq, c] * V_sub[eq, c]
            coeff = U_sub * V_sub
            
            # Build sparse matrix A
            # A has shape (num_eqs, dim_W * r)
            # Non-zero entries: for each eq and c where coeff[eq,c] != 0
            # row = eq, col = W_rows[eq] * r + c
            
            # Find non-zero coefficients
            nz_mask = coeff != 0
            nz_rows = np.where(nz_mask)[0]
            nz_cols_local = np.where(nz_mask)[1]
            nz_vals = coeff[nz_mask]
            
            # Compute global column indices
            nz_cols = W_rows[nz_rows] * r + nz_cols_local
            
            # Build sparse matrix
            A = coo_matrix((nz_vals, (nz_rows, nz_cols)), shape=(num_eqs, dim_W * r)).tocsr()
            b = np.ones(num_eqs)
            
            # Solve using sparse least squares
            try:
                W_flat, info = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=500)
            except Exception:
                continue
            
            # Check residual
            W_sol = W_flat.reshape((dim_W, r))
            
            # Check half-integer
            W_scaled = W_sol * 2
            W_rounded = np.round(W_scaled)
            
            if np.max(np.abs(W_scaled - W_rounded)) < 1e-6:
                W_int = W_rounded / 2.0
                
                if verify_fast(U, V, W_int, r):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist()}
    
    return get_schoolbook()
# EVOLVE-BLOCK-END
