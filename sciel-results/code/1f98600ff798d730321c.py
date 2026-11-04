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
        U_sub = U[U_rows, :]
        V_sub = V[V_rows, :]
        W_sub = W[W_rows, :]
        
        prod = U_sub * V_sub * W_sub
        sums = np.sum(prod, axis=1)
        
        if not np.allclose(sums, 1.0, atol=1e-9):
            return False
        
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        return True
    
    def try_polish(W_sol, U, V, r):
        """
        Try to polish W_sol to exact half-integers using rational reconstruction.
        """
        W_scaled = W_sol * 2
        W_rounded = np.round(W_scaled)
        
        if np.max(np.abs(W_scaled - W_rounded)) < 1e-8:
            W_int = W_rounded / 2.0
            if verify_fast(U, V, W_int, r):
                return W_int
            return None
        
        W_polished = W_sol.copy()
        for i in range(dim_W):
            for j in range(r):
                val = W_sol[i, j]
                for denom in [1, 2, 3, 4, 6]:
                    num = round(val * denom)
                    candidate = num / denom
                    if abs(val - candidate) < 1e-6:
                        W_polished[i, j] = candidate
                        break
        
        if verify_fast(U, V, W_polished, r):
            return W_polished
        return None

    def try_polish_all(W_sol, U, V, r):
        """
        Polish all three matrices to half-integers if close.
        """
        U_polished = U.copy()
        V_polished = V.copy()
        W_polished = W_sol.copy()
        
        # Check if already close to half-integers
        for mat_name, mat in [('U', U_polished), ('V', V_polished), ('W', W_polished)]:
            scaled = mat * 2
            rounded = np.round(scaled)
            max_diff = np.max(np.abs(scaled - rounded))
            if max_diff < 1e-8:
                pass  # Already close
            else:
                # Try to polish
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        val = mat[i, j]
                        for denom in [1, 2, 3, 4, 6]:
                            num = round(val * denom)
                            candidate = num / denom
                            if abs(val - candidate) < 1e-6:
                                mat[i, j] = candidate
                                break
        
        if verify_fast(U_polished, V_polished, W_polished, r):
            return U_polished, V_polished, W_polished
        return None

    # Search for better rank
    # We prioritize ranks close to 84 as they are more likely to be found
    ranks_to_try = list(range(83, 75, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Number of tries - increased for higher probability of success
        num_tries = 50
        
        for trial in range(num_tries):
            if time.time() > deadline - 2:
                break
            
            # Initialize U and V with structured random integers
            U = np.zeros((dim_U, r), dtype=float)
            V = np.zeros((dim_V, r), dtype=float)
            
            # Strategy: Use different initialization strategies
            if trial % 4 == 0:
                # Dense random initialization with values in {-1, 0, 1}
                for c in range(r):
                    mask_U = np.random.random(dim_U) < 0.6
                    mask_V = np.random.random(dim_V) < 0.4
                    U[mask_U, c] = np.random.choice([-1.0, 1.0], size=np.sum(mask_U))
                    V[mask_V, c] = np.random.choice([-1.0, 1.0], size=np.sum(mask_V))
            elif trial % 4 == 1:
                # Sparse initialization with guaranteed coverage
                for c in range(r):
                    num_nonzero_U = max(3, np.random.randint(2, dim_U // 2 + 1))
                    rows_U = np.random.choice(dim_U, size=num_nonzero_U, replace=False)
                    U[rows_U, c] = np.random.choice([-1.0, 1.0], size=num_nonzero_U)
                    
                    num_nonzero_V = max(3, np.random.randint(2, dim_V // 3 + 1))
                    rows_V = np.random.choice(dim_V, size=num_nonzero_V, replace=False)
                    V[rows_V, c] = np.random.choice([-1.0, 1.0], size=num_nonzero_V)
            elif trial % 4 == 2:
                # Semi-structured: each row of U appears in ~r/3 columns
                for c in range(r):
                    num_U = np.random.randint(2, dim_U // 2 + 1)
                    rows_U = np.random.choice(dim_U, size=num_U, replace=False)
                    U[rows_U, c] = np.random.choice([-1.0, 1.0], size=num_U)
                    
                    num_V = np.random.randint(2, dim_V // 3 + 1)
                    rows_V = np.random.choice(dim_V, size=num_V, replace=False)
                    V[rows_V, c] = np.random.choice([-1.0, 1.0], size=num_V)
            else:
                # Block-structured: group rows into blocks
                block_size_U = max(1, dim_U // 4)
                block_size_V = max(1, dim_V // 7)
                
                for c in range(r):
                    # Pick a block for U
                    block_U = np.random.randint(0, 4)
                    start_U = block_U * block_size_U
                    end_U = min(start_U + block_size_U, dim_U)
                    U[start_U:end_U, c] = np.random.choice([-1.0, 1.0], size=end_U - start_U)
                    
                    # Pick a block for V
                    block_V = np.random.randint(0, 7)
                    start_V = block_V * block_size_V
                    end_V = min(start_V + block_size_V, dim_V)
                    V[start_V:end_V, c] = np.random.choice([-1.0, 1.0], size=end_V - start_V)
            
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
            result = try_polish_all(W_sol, U, V, r)
            
            if result is not None:
                U_pol, V_pol, W_pol = result
                return {"U": U_pol.tolist(), "V": V_pol.tolist(), "W": W_pol.tolist()}
    
    return get_schoolbook()
# EVOLVE-BLOCK-END
