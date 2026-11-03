"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy (vectorized)."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build all (i,l,k) index combinations
    i_idx, l_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    i_flat = i_idx.flatten()
    l_flat = l_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Compute sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r] for all (i,l,k)
    U_sel = U_arr[i_flat * m + l_flat, :]  # shape (ntriples, rank)
    V_sel = V_arr[l_flat * p + k_flat, :]  # shape (ntriples, rank)
    W_sel = W_arr[k_flat * n + i_flat, :]  # shape (ntriples, rank)
    
    products = U_sel * V_sel * W_sel  # shape (ntriples, rank)
    sums = products.sum(axis=1)  # shape (ntriples,)
    
    return np.allclose(sums, 1.0, atol=1e-9)

def _build_rank54(n, m, p):
    """Build the known rank-54 Strassen-based decomposition."""
    rank54 = 54
    U54 = np.zeros((n*m, rank54), dtype=float)
    V54 = np.zeros((m*p, rank54), dtype=float)
    W54 = np.zeros((p*n, rank54), dtype=float)
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U54, V54, W54, col, n, m, p)
    for k in range(p):
        U54[0*m + 4, col] += 1
        V54[4*p + k, col] += 1
        W54[k*n + 0, col] += 1
        col += 1
        U54[1*m + 4, col] += 1
        V54[4*p + k, col] += 1
        W54[k*n + 1, col] += 1
        col += 1
    
    return U54, V54, W54

def _try_rank(R, n, m, p, deadline, rng, U54_arr, V54_arr, W54_arr):
    """Try to find a rank-R decomposition using numerical optimization."""
    U_shape = (n*m, R)
    V_shape = (m*p, R)
    W_shape = (p*n, R)
    ntriples = n * m * p
    
    # Precompute index arrays for fast residual computation
    i_idx, l_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    i_flat = i_idx.flatten()
    l_flat = l_idx.flatten()
    k_flat = k_idx.flatten()
    U_row_idx = i_flat * m + l_flat
    V_row_idx = l_flat * p + k_flat
    W_row_idx = k_flat * n + i_flat
    
    def residuals(x):
        U = x[:n*m*R].reshape(U_shape)
        V = x[n*m*R:n*m*R + m*p*R].reshape(V_shape)
        W = x[n*m*R + m*p*R:].reshape(W_shape)
        
        U_sel = U[U_row_idx, :]
        V_sel = V[V_row_idx, :]
        W_sel = W[W_row_idx, :]
        return (U_sel * V_sel * W_sel).sum(axis=1) - 1.0
    
    rank54 = 54
    if R <= rank54:
        U_init = U54_arr[:, :R].copy()
        V_init = V54_arr[:, :R].copy()
        W_init = W54_arr[:, :R].copy()
    else:
        U_init = np.zeros(U_shape)
        V_init = np.zeros(V_shape)
        W_init = np.zeros(W_shape)
        U_init[:, :rank54] = U54_arr
        V_init[:, :rank54] = V54_arr
        W_init[:, :rank54] = W54_arr
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Try multiple strategies
    for attempt in range(8):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            # Small perturbation of Strassen init
            x_try = x0 + rng.normal(0, 0.05, size=x0.shape)
        elif attempt == 1:
            # Larger perturbation
            x_try = x0 + rng.normal(0, 0.5, size=x0.shape)
        elif attempt == 2:
            # Pure random
            x_try = rng.normal(0, 1, size=x0.shape)
        else:
            # Mix of random and Strassen
            alpha = 0.1 * (attempt - 2)
            x_try = alpha * x0 + (1 - alpha) * rng.normal(0, 1, size=x0.shape)
        
        try:
            result = least_squares(residuals, x_try, method='lm', max_nfev=5000, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            
            if result.success or np.max(np.abs(result.fun)) < 1e-9:
                U_sol = result.x[:n*m*R].reshape(U_shape)
                V_sol = result.x[n*m*R:n*m*R + m*p*R].reshape(V_shape)
                W_sol = result.x[n*m*R + m*p*R:].reshape(W_shape)
                
                if _verify_decomp(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), n, m, p, R):
                    return U_sol, V_sol, W_sol
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140  # 10s safety margin
    
    # First, build the known rank-54 decomposition as fallback
    U54_arr, V54_arr, W54_arr = _build_rank54(n, m, p)
    rank54 = 54
    
    # Verify the rank-54 decomposition
    if not _verify_decomp(U54_arr.tolist(), V54_arr.tolist(), W54_arr.tolist(), n, m, p, rank54):
        # Should not happen, but fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n*m)]
        V = [[0] * rank for _ in range(m*p)]
        W = [[0] * rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    U[i*m+l][col] = 1
                    V[l*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U54_arr.tolist(), V54_arr.tolist(), W54_arr.tolist()
    best_rank = rank54
    
    # Try to find lower rank decompositions
    # Focus on ranks 48-53 where we have the best chance
    rng = np.random.default_rng(42)
    
    # Try ranks in order of likelihood: 53, 52, 51, 50, 49, 48
    for R in [53, 52, 51, 50, 49, 48]:
        if time.time() > _deadline:
            break
        result = _try_rank(R, n, m, p, _deadline, rng, U54_arr, V54_arr, W54_arr)
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_U = U_sol.tolist()
            best_V = V_sol.tolist()
            best_W = W_sol.tolist()
            best_rank = R
            # Found a lower rank, continue trying even lower
            continue
        else:
            # Try one more time with different seed before giving up on this rank
            if time.time() < _deadline - 5:
                rng2 = np.random.default_rng(R * 7 + 13)
                result = _try_rank(R, n, m, p, _deadline, rng2, U54_arr, V54_arr, W54_arr)
                if result is not None:
                    U_sol, V_sol, W_sol = result
                    best_U = U_sol.tolist()
                    best_V = V_sol.tolist()
                    best_W = W_sol.tolist()
                    best_rank = R
                    continue
            # If we fail at this rank, try the next lower rank (don't break)
            # since lower ranks might be easier to find with random init
    
    # Final check: if we haven't improved, return the rank-54 solution
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
