import numpy as np
import time

# EVOLVE-BLOCK-START

_START_TIME = time.time()
TIME_LIMIT = 110.0  # 150s limit, keep 40s margin

def _check_deadline():
    if time.time() - _START_TIME > TIME_LIMIT:
        raise TimeoutError("Deadline exceeded")

# The 7 Strassen linear combinations for A, B, and C.
_A_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, 1)],
    [(0, 0, 1)],
    [(1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
    [(1, 0, 1), (0, 0, -1)],
    [(0, 1, 1), (1, 1, -1)]
]

_B_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(0, 0, 1)],
    [(0, 1, 1), (1, 1, -1)],
    [(1, 0, 1), (0, 0, -1)],
    [(1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
    [(1, 0, 1), (1, 1, 1)]
]

_C_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, -1)],
    [(0, 1, 1), (1, 1, 1)],
    [(0, 0, 1), (1, 0, 1)],
    [(0, 1, 1), (0, 0, -1)],
    [(1, 1, 1)],
    [(0, 0, 1)]
]

def _generate_strassen_4x4():
    """
    Generates U, V, W for 4x4 Strassen decomposition directly.
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # Outer Strassen loop
    for t_outer in range(7):
        # Inner Strassen loop
        for t_inner in range(7):
            r_idx = t_outer * 7 + t_inner
            
            # A (U) part
            for (br, bc, sign) in _A_COMBOS[t_outer]:
                for (ir, ik, sign_inner) in _A_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    u_idx = i_global * n + k_global
                    U[u_idx, r_idx] += sign * sign_inner
            
            # B (V) part
            for (br, bc, sign) in _B_COMBOS[t_outer]:
                for (ir, ik, sign_inner) in _B_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    v_idx = i_global * n + k_global
                    V[v_idx, r_idx] += sign * sign_inner
            
            # C (W) part
            for (br, bc, sign) in _C_COMBOS[t_outer]:
                for (ir, ik, sign_inner) in _C_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    w_idx = k_global * n + i_global
                    W[w_idx, r_idx] += sign * sign_inner

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    rank = U.shape[1]
    U_r = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_r = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_r = np.asarray(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))

def _is_half_integer(arr, atol=1e-6):
    return bool(np.allclose(arr * 2, np.round(arr * 2), atol=atol))

def _try_reduce_rank(U, V, W, n, m, p):
    """
    Attempts to reduce the rank by 1 by finding a linear dependence.
    Returns (U_new, V_new, W_new) if successful, else None.
    """
    rank = U.shape[1]
    if rank <= 1:
        return None
        
    # Try removing each term
    for k in range(rank):
        _check_deadline()
        # Indices of terms to keep
        keep = [i for i in range(rank) if i != k]
        
        U_keep = U[:, keep]
        V_keep = V[:, keep]
        W_keep = W[:, keep]
        
        U_target = U[:, k]
        V_target = V[:, k]
        W_target = W[:, k]
        
        # Solve U_keep @ c = U_target
        try:
            c, residuals, rank_u, sv = np.linalg.lstsq(U_keep, U_target, rcond=None)
            # Check if U is exactly in the span
            if not np.allclose(U_keep @ c, U_target, atol=1e-9):
                continue
                
            # Check V
            if not np.allclose(V_keep @ c, V_target, atol=1e-9):
                continue
                
            # Check W
            if not np.allclose(W_keep @ c, W_target, atol=1e-9):
                continue
            
            # Found solution. New coefficients are 1 + c.
            coeffs = 1.0 + c
            
            # Try scaling U, V, or W to get integer/half-integer entries
            # Option 1: Scale W
            W_new = W_keep * coeffs
            if _is_half_integer(W_new):
                U_new = U_keep
                V_new = V_keep
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    return U_new, V_new, W_new
            
            # Option 2: Scale U
            U_new = U_keep * coeffs
            if _is_half_integer(U_new):
                W_new = W_keep
                V_new = V_keep
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    return U_new, V_new, W_new
            
            # Option 3: Scale V
            V_new = V_keep * coeffs
            if _is_half_integer(V_new):
                U_new = U_keep
                W_new = W_keep
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    return U_new, V_new, W_new
            
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = 4, 4, 4
    U, V, W = _generate_strassen_4x4()
    
    # Try to reduce rank
    try:
        reduced = _try_reduce_rank(U, V, W, n, m, p)
        if reduced is not None:
            U, V, W = reduced
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except TimeoutError:
        pass

    # Fallback to rank 49
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
