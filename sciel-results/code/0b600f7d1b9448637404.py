"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

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
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
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
        if not np.allclose(sums, 1.0, atol=1e-9):
            return False
        
        # Check half-integer property for all matrices
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        return True
    
    def try_polish(W_sol, U, V, r):
        """
        Try to polish W_sol to exact half-integers using rational reconstruction.
        """
        # First check if already close to half-integers
        W_scaled = W_sol * 2
        W_rounded = np.round(W_scaled)
        
        if np.max(np.abs(W_scaled - W_rounded)) < 1e-8:
            W_int = W_rounded / 2.0
            if verify_fast(U, V, W_int, r):
                return W_int
            return None
        
        # Try to use rational reconstruction for entries close to simple fractions
        W_polished = W_sol.copy()
        for i in range(dim_W):
            for j in range(r):
                val = W_sol[i, j]
                # Try denominators 1, 2, 3, 4, 6
                for denom in [1, 2, 3, 4, 6]:
                    num = round(val * denom)
                    candidate = num / denom
                    if abs(val - candidate) < 1e-6:
                        W_polished[i, j] = candidate
                        break
        
        # Check if polished version works
        if verify_fast(U, V, W_polished, r):
            return W_polished
        return None

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
        if time.time() > deadline - 10:
            break
        
        # Number of tries - increased for higher probability of success
        if r > 78:
            num_tries = 30
        elif r > 72:
            num_tries = 20
        elif r > 60:
            num_tries = 12
        else:
            num_tries = 6
            
        for trial in range(num_tries):
            if time.time() > deadline - 2:
                break
            
            # Initialize U and V with structured random integers
            U = np.zeros((dim_U, r), dtype=float)
            V = np.zeros((dim_V, r), dtype=float)
            
            # Strategy: Use different initialization strategies
            if trial % 3 == 0:
                # Dense random initialization with values in {-1, 0, 1}
                for c in range(r):
                    # Ensure enough non-zeros for solvability
                    mask_U = np.random.random(dim_U) < 0.6
                    mask_V = np.random.random(dim_V) < 0.4
                    U[mask_U, c] = np.random.choice([-1.0, 1.0], size=np.sum(mask_U))
                    V[mask_V, c] = np.random.choice([-1.0, 1.0], size=np.sum(mask_V))
            elif trial % 3 == 1:
                # Sparse initialization with guaranteed coverage
                for c in range(r):
                    # Ensure at least 3 non-zeros in U
                    num_nonzero_U = max(3, np.random.randint(2, dim_U // 2 + 1))
                    rows_U = np.random.choice(dim_U, size=num_nonzero_U, replace=False)
                    U[rows_U, c] = np.random.choice([-1.0, 1.0], size=num_nonzero_U)
                    
                    num_nonzero_V = max(3, np.random.randint(2, dim_V // 3 + 1))
                    rows_V = np.random.choice(dim_V, size=num_nonzero_V, replace=False)
                    V[rows_V, c] = np.random.choice([-1.0, 1.0], size=num_nonzero_V)
            else:
                # Semi-structured: each row of U appears in ~r/3 columns
                col_per_row_U = max(1, r // 4)
                col_per_row_V = max(1, r // 6)
                
                for c in range(r):
                    # Pick a subset of rows
                    num_U = np.random.randint(2, dim_U // 2 + 1)
                    rows_U = np.random.choice(dim_U, size=num_U, replace=False)
                    U[rows_U, c] = np.random.choice([-1.0, 1.0], size=num_U)
                    
                    num_V = np.random.randint(2, dim_V // 3 + 1)
                    rows_V = np.random.choice(dim_V, size=num_V, replace=False)
                    V[rows_V, c] = np.random.choice([-1.0, 1.0], size=num_V)
            
            # Build A and b
            U_sub = U[U_rows, :]
            V_sub = V[V_rows, :]
            coeff = U_sub * V_sub
            
            A = np.zeros((num_eqs, dim_W * r))
            b = np.ones(num_eqs)
            
            eq_indices = np.arange(num_eqs)
            col_indices = W_rows[:, None] * r + np.arange(r)[None, :]
            
            A[eq_indices[:, None], col_indices] = coeff
            
            # Solve
            try:
                W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            W_sol = W_flat.reshape((dim_W, r))
            
            # Try to polish to exact half-integers
            W_int = try_polish(W_sol, U, V, r)
            
            if W_int is not None:
                return {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist()}
    
    return get_schoolbook()
# EVOLVE-BLOCK-END
