"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _compute_tensor_error(U, V, W, n, m, p):
    """Compute the maximum error of the tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                max_err = max(max_err, abs(s - 1.0))
    
    # Check invalid entries
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                if i == i2 and j == j2 and k == k2:
                    continue
                s = 0.0
                for r in range(rank):
                    s += U[y, r] * V[z, r] * W[x, r]
                max_err = max(max_err, abs(s))
    
    return max_err

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    return _compute_tensor_error(U, V, W, n, m, p) < tol

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Pairs of j indices for Strassen blocks
    j_pairs = [(0, 1), (2, 3)]
    # Pairs of k indices for Strassen blocks
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # 7 products for Strassen 2x2x2 block
            
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A_1,j0 + A_1,j1) B_j0,k0
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A_0,j0 + A_0,j1) B_j1,k1
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
            
    # Remaining j=4
    for k in range(p):
        # C_0,k = A_0,4 B_4,k
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        # C_1,k = A_1,4 B_4,k
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

def _snap_to_half_integer(x, tol=1e-6):
    """Snap a value to the nearest half-integer if close."""
    half = round(x * 2) / 2.0
    if abs(x - half) < tol:
        return half
    return x

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=1000):
    """
    Try to reduce the rank from current to target_rank by driving columns to zero.
    Uses a penalty method: minimize tensor error + lambda * sum of column norms squared.
    """
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return None
    
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    
    # We'll try to zero out columns from the end
    cols_to_zero = current_rank - target_rank
    
    # Penalty weight
    lambda_penalty = 0.1
    
    for iteration in range(max_iters):
        if _elapsed() > _TIME_LIMIT - 5:
            break
        
        # Compute gradient of penalty term
        # For each column r, we want to minimize ||U[:,r]||^2 + ||V[:,r]||^2 + ||W[:,r]||^2
        # while keeping tensor error small
        
        # Simple approach: for each column to zero, try to scale it down
        # and compensate by adjusting other columns
        
        # For now, just try a simple annealing approach
        # Scale down the last cols_to_zero columns
        for c in range(cols_to_zero):
            col_idx = current_rank - 1 - c
            scale = 0.95  # Reduce by 5% each iteration
            U_work[:, col_idx] *= scale
            V_work[:, col_idx] *= scale
            W_work[:, col_idx] *= scale
        
        # Check error
        err = _compute_tensor_error(U_work, V_work, W_work, n, m, p)
        if err < 1e-8:
            # Check if we can now drop the zeroed columns
            # Find columns with very small norm
            col_norms = np.sum(U_work**2, axis=0) + np.sum(V_work**2, axis=0) + np.sum(W_work**2, axis=0)
            small_cols = np.where(col_norms < 1e-6)[0]
            if len(small_cols) >= cols_to_zero:
                # Keep only the non-small columns
                keep_cols = np.setdiff1d(np.arange(current_rank), small_cols[:cols_to_zero])
                U_new = U_work[:, keep_cols]
                V_new = V_work[:, keep_cols]
                W_new = W_work[:, keep_cols]
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    return {"U": U_new, "V": V_new, "W": W_new}
        
        # If error is too large, revert
        if err > 1e-4:
            U_work = U.copy()
            V_work = V.copy()
            W_work = W.copy()
            break
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the known rank 54 solution
    base_sol = _build_54_rank(n, m, p)
    best_result = base_sol
    best_rank = 54
    
    # Verify the base solution
    if not _verify_decomposition(base_sol["U"], base_sol["V"], base_sol["W"], n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    if _time_left() > 20:
        # Try reducing to rank 53
        result_53 = _try_reduce_rank(base_sol["U"], base_sol["V"], base_sol["W"], n, m, p, 53, max_iters=500)
        if result_53 is not None:
            best_result = result_53
            best_rank = 53
        
        # Try reducing to rank 52
        if best_rank == 54 and _time_left() > 15:
            result_52 = _try_reduce_rank(base_sol["U"], base_sol["V"], base_sol["W"], n, m, p, 52, max_iters=300)
            if result_52 is not None:
                best_result = result_52
                best_rank = 52
    
    # Snap to half-integers
    U = best_result["U"]
    V = best_result["V"]
    W = best_result["W"]
    
    U_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in U])
    V_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in V])
    W_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in W])
    
    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
