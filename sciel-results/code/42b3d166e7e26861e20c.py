import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140  # safety margin before 150s kill

def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_target_tensor(n, m, p):
    """Build the target tensor T[a,b,c] for matrix multiplication."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if U,V,W form a valid decomposition with integer/half-integer entries."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_target = _build_target_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _als_decompose(T, R, n, m, p, seed=42, init_type='random', n_iter=300, tol=1e-9):
    """
    Alternating Least Squares for tensor decomposition.
    T: shape (A, B, C)
    R: rank
    Returns (U, V, W) or None if it fails to converge.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(seed)
    
    if init_type == 'random':
        U = rng.randn(A, R) * 0.1
        V = rng.randn(B, R) * 0.1
        W = rng.randn(C, R) * 0.1
    elif init_type == 'sparse_int':
        U = rng.randint(-1, 2, size=(A, R)).astype(float)
        V = rng.randint(-1, 2, size=(B, R)).astype(float)
        W = rng.randint(-1, 2, size=(C, R)).astype(float)
        for r in range(R):
            nu = np.linalg.norm(U[:, r])
            nv = np.linalg.norm(V[:, r])
            nw = np.linalg.norm(W[:, r])
            if nu > 1e-9 and nv > 1e-9 and nw > 1e-9:
                U[:, r] /= nu
                V[:, r] /= nv
                W[:, r] /= nw
    elif init_type == 'schoolbook_trunc':
        U = np.zeros((A, R))
        V = np.zeros((B, R))
        W = np.zeros((C, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < R:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
        for r in range(col, R):
            U[:, r] = rng.randn(A) * 0.3
            V[:, r] = rng.randn(B) * 0.3
            W[:, r] = rng.randn(C) * 0.3
    
    # Precompute target slices
    T_a = [T[a, :, :].flatten() for a in range(A)]
    T_b = [T[:, b, :].flatten() for b in range(B)]
    T_c = [T[:, :, c].flatten() for c in range(C)]
    
    prev_err = float('inf')
    for it in range(n_iter):
        if _remaining_time() < 5:
            break
        
        # Solve for U
        A_U = np.outer(V.flatten(), W.flatten()).reshape(B * C, R)
        for a in range(A):
            U[a, :], _, _, _ = np.linalg.lstsq(A_U, T_a[a], rcond=None)
        
        # Solve for V
        A_V = np.outer(U.flatten(), W.flatten()).reshape(A * C, R)
        for b in range(B):
            V[b, :], _, _, _ = np.linalg.lstsq(A_V, T_b[b], rcond=None)
        
        # Solve for W
        A_W = np.outer(U.flatten(), V.flatten()).reshape(A * B, R)
        for c in range(C):
            W[c, :], _, _, _ = np.linalg.lstsq(A_W, T_c[c], rcond=None)
        
        # Check convergence every 5 iterations
        if it % 5 == 0:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < tol:
                break
            if err >= prev_err * 0.999 and it > 50:
                # Stagnation check
                pass
            prev_err = err
    
    # Final check
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_recon - T)) < 1e-6:
        return U, V, W
    return None

def _try_round_and_verify(U, V, W, n, m, p):
    """Try to round to half-integers or integers and verify."""
    # Try half-integers
    U_h = np.round(U * 2) / 2
    V_h = np.round(V * 2) / 2
    W_h = np.round(W * 2) / 2
    if _verify(U_h, V_h, W_h, n, m, p):
        return U_h.tolist(), V_h.tolist(), W_h.tolist()
    
    # Try integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    if _verify(U_i, V_i, W_i, n, m, p):
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
    
    # Try floor/ceil for near-integer values
    for scale in [1, 2]:
        U_s = np.round(U * scale) / scale
        V_s = np.round(V * scale) / scale
        W_s = np.round(W * scale) / scale
        if _verify(U_s, V_s, W_s, n, m, p):
            return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    return None

def _search_for_decomposition(n, m, p):
    """Search for a low-rank decomposition of the matrix multiplication tensor."""
    T = _build_target_tensor(n, m, p)
    
    # Try ranks from 1 upward
    for rank in range(1, n * m * p + 1):
        if _remaining_time() < 10:
            break
        
        # Try multiple initialization strategies
        for seed in range(5):
            if _remaining_time() < 5:
                break
            
            for init_type in ['random', 'sparse_int', 'schoolbook_trunc']:
                if _remaining_time() < 5:
                    break
                
                result = _als_decompose(T, rank, n, m, p, seed=seed, init_type=init_type)
                if result is not None:
                    U, V, W = result
                    rounded = _try_round_and_verify(U, V, W, n, m, p)
                    if rounded is not None:
                        return {"U": rounded[0], "V": rounded[1], "W": rounded[2]}
    
    return None

def construct():
    n, m, p = 2, 3, 4
    
    # First, try to find a lower-rank decomposition
    result = _search_for_decomposition(n, m, p)
    if result is not None:
        return result
    
    # Fall back to schoolbook decomposition
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
