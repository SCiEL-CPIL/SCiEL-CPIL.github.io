"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def _verify_half_integer(U, V, W, n, m, p):
    """Verify that U, V, W (containing integers or half-integers) form a valid decomposition."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check type constraint: entries must be integers or half-integers
    for mat in [U_arr, V_arr, W_arr]:
        for val in mat.flat:
            if not np.isclose(val * 2, round(val * 2), atol=1e-9):
                return False
    
    # Build the target tensor T[i,j,k] = 1
    # Reconstructed: T_rec[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T_rec = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u_col = U_arr[:, c]
        v_col = V_arr[:, c]
        w_col = W_arr[:, c]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i*m+j]
                if abs(u_val) < 1e-15:
                    continue
                for k in range(p):
                    T_rec[i, j, k] += u_val * v_col[j*p+k] * w_col[k*n+i]
    
    return np.allclose(T_rec, 1.0, atol=1e-9)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _structured_init(n, m, p, target_rank, seed):
    """Generate structured initializations that may be closer to valid solutions."""
    np.random.seed(seed)
    num_rows_U = n * m
    num_rows_V = m * p
    num_rows_W = p * n
    
    # Try different structured patterns
    pattern = seed % 5
    
    if pattern == 0:
        # Random but scaled
        U = np.random.randn(num_rows_U, target_rank) * 0.5
        V = np.random.randn(num_rows_V, target_rank) * 0.5
        W = np.random.randn(num_rows_W, target_rank) * 0.5
    elif pattern == 1:
        # Block-structured: each column has nonzeros in specific blocks
        U = np.zeros((num_rows_U, target_rank))
        V = np.zeros((num_rows_V, target_rank))
        W = np.zeros((num_rows_W, target_rank))
        for c in range(target_rank):
            # Randomly assign which (i,j), (j,k), (k,i) indices are active
            i_active = np.random.choice(n, size=max(1, n//2))
            j_active = np.random.choice(m, size=max(1, m//2))
            k_active = np.random.choice(p, size=max(1, p//2))
            for i in i_active:
                for j in j_active:
                    U[i*m+j, c] = np.random.choice([-1, 1]) * 0.5
            for j in j_active:
                for k in k_active:
                    V[j*p+k, c] = np.random.choice([-1, 1]) * 0.5
            for k in k_active:
                for i in i_active:
                    W[k*n+i, c] = np.random.choice([-1, 1]) * 0.5
    elif pattern == 2:
        # Symmetric-like initialization
        U = np.random.randn(num_rows_U, target_rank) * 0.3
        V = np.random.randn(num_rows_V, target_rank) * 0.3
        W = np.random.randn(num_rows_W, target_rank) * 0.3
        # Add some structure: make certain rows similar
        if num_rows_U >= 2:
            U[0, :] = U[1, :] + np.random.randn(target_rank) * 0.1
    elif pattern == 3:
        # Sparse initialization
        U = np.zeros((num_rows_U, target_rank))
        V = np.zeros((num_rows_V, target_rank))
        W = np.zeros((num_rows_W, target_rank))
        for c in range(target_rank):
            nnz_U = max(1, target_rank // 3)
            nnz_V = max(1, target_rank // 3)
            nnz_W = max(1, target_rank // 3)
            idx_U = np.random.choice(num_rows_U, size=nnz_U, replace=False)
            idx_V = np.random.choice(num_rows_V, size=nnz_V, replace=False)
            idx_W = np.random.choice(num_rows_W, size=nnz_W, replace=False)
            U[idx_U, c] = np.random.choice([-1, 1], size=nnz_U) * 0.5
            V[idx_V, c] = np.random.choice([-1, 1], size=nnz_V) * 0.5
            W[idx_W, c] = np.random.choice([-1, 1], size=nnz_W) * 0.5
    else:
        # Mixed: some random, some structured
        U = np.random.randn(num_rows_U, target_rank) * 0.4
        V = np.random.randn(num_rows_V, target_rank) * 0.4
        W = np.random.randn(num_rows_W, target_rank) * 0.4
        # Zero out random entries
        mask_U = np.random.rand(num_rows_U, target_rank) > 0.3
        mask_V = np.random.rand(num_rows_V, target_rank) > 0.3
        mask_W = np.random.rand(num_rows_W, target_rank) > 0.3
        U[~mask_U] = 0
        V[~mask_V] = 0
        W[~mask_W] = 0
    
    return U, V, W

def _refine_decomposition(U, V, W, n, m, p, max_iterations=200, tol=1e-10, time_budget=5.0):
    """Refine a decomposition using alternating least squares with snapping."""
    start_time = time.time()
    num_rows_U = n * m
    num_rows_V = m * p
    num_rows_W = p * n
    rank = U.shape[1]
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_budget:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                try:
                    U[i*m+j, :], residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                try:
                    V[j*p+k, :], residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                try:
                    W[k*n+i, :], residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    pass
        
        # Compute error
        T_rec = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            u_col = U[:, c]
            v_col = V[:, c]
            w_col = W[:, c]
            for i in range(n):
                for j in range(m):
                    u_val = u_col[i*m+j]
                    if abs(u_val) < 1e-15:
                        continue
                    for k in range(p):
                        T_rec[i, j, k] += u_val * v_col[j*p+k] * w_col[k*n+i]
        
        error = np.max(np.abs(T_rec - 1.0))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    # Try to snap to half-integers
    U_snap = np.round(best_U * 2) / 2
    V_snap = np.round(best_V * 2) / 2
    W_snap = np.round(best_W * 2) / 2
    
    if _verify_half_integer(U_snap, V_snap, W_snap, n, m, p):
        U_list = [[float(x) for x in row] for row in U_snap]
        V_list = [[float(x) for x in row] for row in V_snap]
        W_list = [[float(x) for x in row] for row in W_snap]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try local search around the best solution
    if best_error < 0.1:
        # Try small perturbations and re-snap
        for _ in range(50):
            if time.time() - start_time > time_budget:
                break
            U_try = best_U + np.random.randn(*best_U.shape) * 0.01
            V_try = best_V + np.random.randn(*best_V.shape) * 0.01
            W_try = best_W + np.random.randn(*best_W.shape) * 0.01
            
            U_try_snap = np.round(U_try * 2) / 2
            V_try_snap = np.round(V_try * 2) / 2
            W_try_snap = np.round(W_try * 2) / 2
            
            if _verify_half_integer(U_try_snap, V_try_snap, W_try_snap, n, m, p):
                U_list = [[float(x) for x in row] for row in U_try_snap]
                V_list = [[float(x) for x in row] for row in V_try_snap]
                W_list = [[float(x) for x in row] for row in W_try_snap]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as the safe fallback
    best = _schoolbook(n, m, p)
    
    start_time = time.time()
    time_limit = 140  # Leave 10s margin for the 150s limit
    
    # Try to reduce rank. Start from a reasonable lower bound.
    # For 3x4x4, rank 48 is schoolbook.
    # We try ranks from 44 down to 32.
    for target_rank in range(44, 31, -1):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        for seed in range(30):
            if time.time() - start_time > time_limit:
                break
            
            U, V, W = _structured_init(n, m, p, target_rank, seed)
            result = _refine_decomposition(U, V, W, n, m, p, max_iterations=150, time_budget=3.0)
            if result is not None:
                best = result
                found = True
                break
        
        if found:
            break
    
    return best
# EVOLVE-BLOCK-END
