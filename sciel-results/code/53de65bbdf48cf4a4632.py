# EVOLVE-BLOCK-START
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target = target * 8
    if np.max(np.abs(T_computed - T_target)) > 1e-6:
        return False
    return True

def _build_target(n, m, p):
    """Build the target tensor for matrix multiplication."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m + b, b*p + c, c*n + a] = 1
    return T

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b][col] = 1
                V[b*p + c][col] = 1
                W[c*n + a][col] = 1
                col += 1
    return U, V, W

def _als_search(target_rank, n, m, p, T_flat, deadline, max_iters=500):
    """
    Use ALS to find a valid decomposition of given rank.
    """
    nm, mp, pn = n*m, m*p, p*n
    T = T_flat.reshape(pn, nm, mp).transpose(1, 2, 0) # Shape (nm, mp, pn)
    
    np.random.seed(42)
    
    # Initialize with random integers
    U = np.random.randint(-5, 6, size=(nm, target_rank))
    V = np.random.randint(-5, 6, size=(mp, target_rank))
    W = np.random.randint(-5, 6, size=(pn, target_rank))
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
            
        # Update U
        # T_{i j k} = sum_r U_{i r} V_{j r} W_{k r}
        # For fixed r, we want U_{i r}.
        # T_{i j k} W_{k r} V_{j r} ...
        # Sum over j, k: T_{i j k} V_{j r} W_{k r} = U_{i r} * Sum_{j k} V_{j r}^2 W_{k r}^2
        # This is not quite right because V and W are coupled.
        # Standard ALS:
        # Fix V, W.
        # U_{i r} = (Sum_{j k} T_{i j k} V_{j r} W_{k r}) / (Sum_{j k} V_{j r}^2 W_{k r}^2)
        
        # Compute denominator for U
        # Denom_U[r] = Sum_{j k} V_{j r}^2 W_{k r}^2
        V2 = V ** 2
        W2 = W ** 2
        # We need sum over j,k of T_{i j k} V_{j r} W_{k r}
        # This is (T * V[:, r] * W[:, r])
        
        # Efficient computation using einsum
        # T is (nm, mp, pn)
        # V is (mp, R), W is (pn, R)
        
        # Numerator for U: N_U[i, r] = Sum_{j, k} T[i, j, k] V[j, r] W[k, r]
        # N_U = np.einsum('ijk, jr, kr -> ir', T, V, W)
        # Denominator for U: D_U[r] = Sum_{j, k} V[j, r]^2 W[k, r]^2
        # D_U = np.einsum('jr, kr -> r', V2, W2)
        
        try:
            N_U = np.einsum('ijk, jr, kr -> ir', T, V, W, optimize=True)
            D_U = np.einsum('jr, kr -> r', V2, W2, optimize=True)
            # Avoid division by zero
            mask = D_U != 0
            U_new = np.zeros_like(U, dtype=np.float64)
            U_new[:, mask] = N_U[:, mask] / D_U[mask]
            U = np.round(U_new) # Quantize to integer first for stability
            
            # Update V
            # N_V[j, r] = Sum_{i, k} T[i, j, k] U[i, r] W[k, r]
            # D_V[r] = Sum_{i, k} U[i, r]^2 W[k, r]^2
            U2 = U ** 2
            N_V = np.einsum('ijk, ir, kr -> jr', T, U, W, optimize=True)
            D_V = np.einsum('ir, kr -> r', U2, W2, optimize=True)
            V_new = np.zeros_like(V, dtype=np.float64)
            mask = D_V != 0
            V_new[:, mask] = N_V[:, mask] / D_V[mask]
            V = np.round(V_new)
            
            # Update W
            # N_W[k, r] = Sum_{i, j} T[i, j, k] U[i, r] V[j, r]
            # D_W[r] = Sum_{i, j} U[i, r]^2 V[j, r]^2
            V2 = V ** 2
            N_W = np.einsum('ijk, ir, jr -> kr', T, U, V, optimize=True)
            D_W = np.einsum('ir, jr -> r', U2, V2, optimize=True)
            W_new = np.zeros_like(W, dtype=np.float64)
            mask = D_W != 0
            W_new[:, mask] = N_W[:, mask] / D_W[mask]
            W = np.round(W_new)
            
            # Check error
            # Compute T_computed
            T_comp = np.einsum('ir, jr, kr -> ijk', U, V, W, optimize=True)
            err = np.max(np.abs(T_comp - T))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err == 0:
                # Verify exactness with half-integers?
                # The problem allows half-integers.
                # If integer solution is exact, it's valid.
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, target_rank):
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception as e:
            continue
            
    return None

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    T = _build_target(n, m, p)
    T_flat = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    
    # Try ALS for lower ranks
    # Known upper bound for 5x5 is 125 (schoolbook).
    # Strassen-like improvements might exist around 100-110.
    ranks_to_try = [110, 105, 100, 95, 90]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        result = _als_search(target_rank, n, m, p, T_flat, deadline, max_iters=200)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            print(f"Found valid rank {best_rank}")
            break
    
    return best_result
# EVOLVE-BLOCK-END
