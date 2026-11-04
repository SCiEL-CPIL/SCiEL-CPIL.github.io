import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _reconstruction_error(U, V, W, T):
    """
    Compute the Frobenius norm of the reconstruction error.
    """
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.sum((recon - T) ** 2)

def _snap_to_rational(U, V, W, T, tol=1e-9):
    """
    Try to snap U, V, W to integers or half-integers and verify exact reconstruction.
    """
    # Try half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
    if np.allclose(recon, T, atol=tol):
        return U_snap, V_snap, W_snap
        
    # Try integers
    U_snap = np.round(U)
    V_snap = np.round(V)
    W_snap = np.round(W)
    
    recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
    if np.allclose(recon, T, atol=tol):
        return U_snap, V_snap, W_snap
        
    return None

def _als_decompose(T, n, m, p, r, max_iters=50):
    """
    Alternating Least Squares for tensor decomposition.
    """
    nm, mp, pn = T.shape
    # Initialize with small random integers
    U = np.random.randint(-1, 2, size=(nm, r))
    V = np.random.randint(-1, 2, size=(mp, r))
    W = np.random.randint(-1, 2, size=(pn, r))
    
    for _ in range(max_iters):
        # Solve for W: T_{i,j,k} = sum_a U_{i,a} V_{j,a} W_{k,a}
        # For fixed U, V, this is linear in W.
        # We can solve for each k separately or stack.
        # Stack: For each k, we have a system.
        # T_{i,j,k} = sum_a (U_{i,a} V_{j,a}) W_{k,a}
        # Let M_k be a matrix of size (nm*mp, r) where M_k[(i,j), a] = U_{i,a} V_{j,a}
        # Then T[:, :, k].flatten() = M_k @ W[k]
        
        # To speed up, we can use the property that M_k = (U * V_reshaped) ...
        # Actually, M_k is the Kronecker-like product.
        # Let's just loop over k for simplicity and correctness in this small case.
        for k in range(pn):
            # Build M_k
            # M_k[i*mp + j, a] = U[i, a] * V[j, a]
            M_k = np.zeros((nm * mp, r))
            for i in range(nm):
                for j in range(mp):
                    M_k[i * mp + j, :] = U[i, :] * V[j, :]
            
            b = T[:, :, k].flatten()
            # Solve M_k @ w_k = b
            try:
                W[k, :] = np.linalg.lstsq(M_k, b, rcond=None)[0]
            except:
                pass
                
        # Solve for V
        for j in range(mp):
            M_j = np.zeros((nm * pn, r))
            for i in range(nm):
                for k in range(pn):
                    M_j[i * pn + k, :] = U[i, :] * W[k, :]
            b = T[:, j, :].flatten() # Note: T[i, j, k]
            # Wait, T is indexed [i, j, k].
            # We need T[i, j, k] for all i, k.
            # T[:, j, :] gives shape (nm, pn). Flatten it.
            try:
                V[j, :] = np.linalg.lstsq(M_j, b, rcond=None)[0]
            except:
                pass
                
        # Solve for U
        for i in range(nm):
            M_i = np.zeros((mp * pn, r))
            for j in range(mp):
                for k in range(pn):
                    M_i[j * pn + k, :] = V[j, :] * W[k, :]
            b = T[i, :, :].flatten()
            try:
                U[i, :] = np.linalg.lstsq(M_i, b, rcond=None)[0]
            except:
                pass
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Deadline guard
    total_start = time.time()
    time_budget = 110.0
    
    # Fallback schoolbook rank 24
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Try ranks from 19 to 23
    for r in range(19, 24):
        elapsed = time.time() - total_start
        remaining = time_budget - elapsed
        if remaining < 5.0:
            break
            
        # Allocate time for this rank
        rank_budget = remaining * 0.4 if r == 19 else remaining * 0.25
        rank_start = time.time()
        
        # Multiple restarts of ALS
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() - rank_start > rank_budget:
                break
                
            np.random.seed(hash((n, m, p, r, restart)) % (2**32))
            
            U, V, W = _als_decompose(T, n, m, p, r, max_iters=20)
            
            err = _reconstruction_error(U, V, W, T)
            
            if err < 1e-6:
                snapped = _snap_to_rational(U, V, W, T)
                if snapped is not None:
                    U_snap, V_snap, W_snap = snapped
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
            
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
