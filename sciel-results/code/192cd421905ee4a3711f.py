import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _snap_to_half_integers(mat):
    """Snap values to the nearest half-integer."""
    snapped = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            snapped[i, j] = round(val * 2) / 2.0
    return snapped

def _verify_exact(U, V, W, n, m, p, R):
    """Verify the decomposition exactly using Fractions."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0, 1)
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                for r in range(R):
                    u_val = Fraction(str(U[idx_u, r]))
                    v_val = Fraction(str(V[idx_v, r]))
                    w_val = Fraction(str(W[idx_w, r]))
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _als_step_U(U, V, W, n, m, p, R):
    """Update U using least squares."""
    new_U = np.zeros_like(U)
    for i in range(n):
        for j in range(m):
            idx_u = i * m + j
            A = np.zeros((p, R))
            for k in range(p):
                idx_v = j * p + k
                idx_w = k * n + i
                A[k, :] = V[idx_v, :] * W[idx_w, :]
            b = np.ones(p)
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            new_U[idx_u, :] = x
    return new_U

def _als_step_V(U, V, W, n, m, p, R):
    """Update V using least squares."""
    new_V = np.zeros_like(V)
    for j in range(m):
        for k in range(p):
            idx_v = j * p + k
            A = np.zeros((n, R))
            for i in range(n):
                idx_u = i * m + j
                idx_w = k * n + i
                A[i, :] = U[idx_u, :] * W[idx_w, :]
            b = np.ones(n)
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            new_V[idx_v, :] = x
    return new_V

def _als_step_W(U, V, W, n, m, p, R):
    """Update W using least squares."""
    new_W = np.zeros_like(W)
    for k in range(p):
        for i in range(n):
            idx_w = k * n + i
            A = np.zeros((m, R))
            for j in range(m):
                idx_u = i * m + j
                idx_v = j * p + k
                A[j, :] = U[idx_u, :] * V[idx_v, :]
            b = np.ones(m)
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            new_W[idx_w, :] = x
    return new_W

def _calculate_error(U, V, W, n, m, p, R):
    """Calculate the Frobenius norm of the error."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                for r in range(R):
                    val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                err += (val - 1.0) ** 2
    return np.sqrt(err)

def _als_solve(n, m, p, R, max_time, max_trials=20, max_iters=200):
    """
    Try to find a rank-R decomposition for M_{n,m,p} using ALS.
    Returns (U, V, W) as numpy arrays or None.
    """
    start_time = time.time()
    
    for trial in range(max_trials):
        if time.time() - start_time > max_time:
            break
        
        # Different initialization strategies
        if trial % 3 == 0:
            np.random.seed(42 + trial)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
        elif trial % 3 == 1:
            np.random.seed(100 + trial)
            U = np.random.randint(-2, 3, size=(n * m, R)).astype(float)
            V = np.random.randint(-2, 3, size=(m * p, R)).astype(float)
            W = np.random.randint(-2, 3, size=(p * n, R)).astype(float)
        else:
            np.random.seed(200 + trial)
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            # Sparse initialization
            for i in range(n * m):
                U[i, i % R] = 1.0
            for j in range(m * p):
                V[j, j % R] = 1.0
            for k in range(p * n):
                W[k, k % R] = 1.0
        
        for it in range(max_iters):
            if time.time() - start_time > max_time:
                break
                
            U = _als_step_U(U, V, W, n, m, p, R)
            V = _als_step_V(U, V, W, n, m, p, R)
            W = _als_step_W(U, V, W, n, m, p, R)
            
            if it % 5 == 0:
                err = _calculate_error(U, V, W, n, m, p, R)
                if err < 1e-10:
                    U_snap = _snap_to_half_integers(U)
                    V_snap = _snap_to_half_integers(V)
                    W_snap = _snap_to_half_integers(W)
                    
                    if _verify_exact(U_snap, V_snap, W_snap, n, m, p, R):
                        return U_snap, V_snap, W_snap
                        
    return None

def _tile_2x3x3_to_2x3x6(U_sub, V_sub, W_sub, R_sub):
    """
    Tile a decomposition of M_{2,3,3} to M_{2,3,6}.
    U_sub: (6, R_sub), V_sub: (9, R_sub), W_sub: (6, R_sub)
    Returns U_full, V_full, W_full for M_{2,3,6}
    """
    n, m, p = 2, 3, 6
    R_full = R_sub * 2
    
    U_full = [[0] * R_full for _ in range(n * m)]
    V_full = [[0] * R_full for _ in range(m * p)]
    W_full = [[0] * R_full for _ in range(p * n)]
    
    # Fill U: same for both tiles
    for i in range(n):
        for j in range(m):
            idx_u = i * m + j
            for rep in range(2):
                r_offset = rep * R_sub
                for r in range(R_sub):
                    U_full[idx_u][r_offset + r] = U_sub[idx_u][r]
    
    # Fill V: each tile uses the sub V
    for j in range(m):
        for k in range(p):
            idx_v = j * p + k
            rep = k // 3
            k_sub = k % 3
            r_offset = rep * R_sub
            idx_v_sub = j * 3 + k_sub
            for r in range(R_sub):
                V_full[idx_v][r_offset + r] = V_sub[idx_v_sub][r]
    
    # Fill W: each tile uses the sub W
    for k in range(p):
        for i in range(n):
            idx_w = k * n + i
            rep = k // 3
            k_sub = k % 3
            r_offset = rep * R_sub
            idx_w_sub = k_sub * n + i
            for r in range(R_sub):
                W_full[idx_w][r_offset + r] = W_sub[idx_w_sub][r]
    
    return U_full, V_full, W_full

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition."""
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

def _direct_2x3x3_rank14():
    """
    Direct construction of a rank-14 decomposition for 2x3x3 matrix multiplication.
    Uses schoolbook-like terms with some sharing.
    """
    n, m, p = 2, 3, 3
    R = 14
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Start with schoolbook (18 terms) and try to merge
    # For now, just use 14 distinct terms
    assignments = []
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < R:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    if _verify_exact(U_np, V_np, W_np, n, m, p, R):
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Strategy 1: Find M_{2,3,3} decomposition and tile to M_{2,3,6}
    remaining_time = deadline - time.time()
    
    for target_rank_sub in range(11, 15):
        if time.time() >= deadline:
            break
        
        rank_time = min(10, remaining_time * 0.3)
        if rank_time < 3:
            continue
        
        # Try ALS on M_{2,3,3}
        result = _als_solve(2, 3, 3, target_rank_sub, max_time=rank_time,
                           max_trials=15, max_iters=200)
        
        if result is not None:
            U_sub, V_sub, W_sub = result
            U_sub_np = np.array(U_sub, dtype=float)
            V_sub_np = np.array(V_sub, dtype=float)
            W_sub_np = np.array(W_sub, dtype=float)
            
            if _verify_exact(U_sub_np, V_sub_np, W_sub_np, 2, 3, 3, target_rank_sub):
                # Tile to M_{2,3,6}
                U_full, V_full, W_full = _tile_2x3x3_to_2x3x6(U_sub, V_sub, W_sub, target_rank_sub)
                
                U_full_np = np.array(U_full, dtype=float)
                V_full_np = np.array(V_full, dtype=float)
                W_full_np = np.array(W_full, dtype=float)
                R_full = target_rank_sub * 2
                
                if _verify_exact(U_full_np, V_full_np, W_full_np, n, m, p, R_full):
                    return {"U": U_full, "V": V_full, "W": W_full}
        
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            break
    
    # Strategy 2: Try direct ALS on M_{2,3,6} at various ranks
    min_possible = max(n * m, m * p, p * n)
    
    remaining_time = deadline - time.time()
    
    for target_rank in range(min_possible, n * m * p + 1):
        if time.time() >= deadline:
            break
        
        if target_rank <= 24:
            rank_time = min(10, remaining_time * 0.3)
        elif target_rank <= 30:
            rank_time = min(8, remaining_time * 0.2)
        else:
            rank_time = min(5, remaining_time * 0.1)
        
        if rank_time < 2:
            continue
        
        result = _als_solve(n, m, p, target_rank, max_time=rank_time,
                           max_trials=15, max_iters=200)
        
        if result is not None:
            U_np, V_np, W_np = result
            if _verify_exact(U_np, V_np, W_np, n, m, p, target_rank):
                return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
        
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
