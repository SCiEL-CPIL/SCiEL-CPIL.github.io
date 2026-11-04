"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

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
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
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
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    U_rows = i_idx * m + j_idx
    V_rows = j_idx * p + k_idx
    W_rows = k_idx * n + i_idx
    
    def verify_fast(U, V, W, r):
        """
        Verify exact decomposition using vectorized operations.
        """
        U_sub = U[U_rows, :]
        V_sub = V[V_rows, :]
        W_sub = W[W_rows, :]
        
        prod = U_sub * V_sub * W_sub
        sums = np.sum(prod, axis=1)
        
        if not np.allclose(sums, 1.0, atol=1e-9):
            return False
        
        # Check half-integer property
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        return True
    
    def polish_to_halfint(X):
        """Polish a matrix to nearest half-integers and verify."""
        X2 = np.round(X * 2) / 2.0
        return X2
    
    # Strategy: Use structured random search with exact solving
    # Key insight: Fix U and V, then W is determined by linear equations
    # We search for U, V configurations that yield integer/half-integer W
    
    best_result = None
    best_rank = 84
    
    # Search ranks from 83 down
    ranks_to_try = list(range(83, 59, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        
        # Number of trials based on rank
        if r > 78:
            num_tries = 50
        elif r > 70:
            num_tries = 30
        elif r > 60:
            num_tries = 20
        else:
            num_tries = 10
        
        for trial in range(num_tries):
            if time.time() > deadline - 2:
                break
            
            # Initialize U and V with random ±1 entries
            # Use different density patterns
            if trial % 4 == 0:
                # Dense U, sparse V
                U = np.random.choice([-1.0, 1.0], size=(dim_U, r))
                V = np.zeros((dim_V, r))
                for c in range(r):
                    mask = np.random.random(dim_V) < 0.3
                    V[mask, c] = np.random.choice([-1.0, 1.0], size=np.sum(mask))
            elif trial % 4 == 1:
                # Sparse U, dense V
                U = np.zeros((dim_U, r))
                V = np.random.choice([-1.0, 1.0], size=(dim_V, r))
                for c in range(r):
                    mask = np.random.random(dim_U) < 0.5
                    U[mask, c] = np.random.choice([-1.0, 1.0], size=np.sum(mask))
            elif trial % 4 == 2:
                # Both moderate density
                U = np.zeros((dim_U, r))
                V = np.zeros((dim_V, r))
                for c in range(r):
                    mask_U = np.random.random(dim_U) < 0.4
                    mask_V = np.random.random(dim_V) < 0.3
                    U[mask_U, c] = np.random.choice([-1.0, 1.0], size=np.sum(mask_U))
                    V[mask_V, c] = np.random.choice([-1.0, 1.0], size=np.sum(mask_V))
            else:
                # Structured: ensure coverage
                U = np.zeros((dim_U, r))
                V = np.zeros((dim_V, r))
                for c in range(r):
                    # Each row of U appears in ~r/3 columns
                    rows_U = np.random.choice(dim_U, size=max(2, dim_U // 3), replace=False)
                    U[rows_U, c] = np.random.choice([-1.0, 1.0], size=len(rows_U))
                    
                    rows_V = np.random.choice(dim_V, size=max(2, dim_V // 4), replace=False)
                    V[rows_V, c] = np.random.choice([-1.0, 1.0], size=len(rows_V))
            
            # Build the linear system for W
            # For each equation e: sum_c U[U_rows[e], c] * V[V_rows[e], c] * W[W_rows[e], c] = 1
            U_sub = U[U_rows, :]
            V_sub = V[V_rows, :]
            coeff = U_sub * V_sub  # (num_eqs, r)
            
            # A matrix: A[e, w_idx * r + c] = coeff[e, c] where w_idx = W_rows[e]
            A = np.zeros((num_eqs, dim_W * r))
            b = np.ones(num_eqs)
            
            # Vectorized construction
            col_indices = W_rows[:, None] * r + np.arange(r)[None, :]
            A[np.arange(num_eqs)[:, None], col_indices] = coeff
            
            # Solve least squares
            try:
                W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            # Check if solution is exact (residual close to 0)
            if len(residuals) > 0 and residuals[0] > 1e-7:
                continue
            
            W_sol = W_flat.reshape((dim_W, r))
            
            # Polish to half-integers
            W_polished = polish_to_halfint(W_sol)
            
            # Verify
            if verify_fast(U, V, W_polished, r):
                best_rank = r
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W_polished.tolist()}
                break  # Found a valid solution at this rank
        
        if best_result is not None:
            break  # Found a valid solution, stop searching
    
    if best_result is not None:
        return best_result
    
    return get_schoolbook()
# EVOLVE-BLOCK-END
