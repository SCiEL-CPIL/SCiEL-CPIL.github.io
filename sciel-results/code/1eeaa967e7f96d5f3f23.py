"""Program for matrix multiplication tensor decomposition (3x4, 4x6)."""
import time
import numpy as np

n_dim = 3
m_dim = 4
p_dim = 6

def _als_step(U, V, W, tensor, n, m, p, rank):
    """
    Perform one ALS step.
    U shape: (n, m, rank)
    V shape: (m, p, rank)
    W shape: (p, n, rank)
    tensor shape: (n, m, p)
    """
    # 1. Solve for U
    # A_u has shape (n*m*p, rank)
    # A_u[i*m*p + j*p + k, c] = V[j,k,c] * W[k,i,c]
    A_u = np.zeros((n * m * p, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                A_u[idx, :] = V[j, k, :] * W[k, i, :]
    
    # Target is tensor flattened
    target_flat = tensor.flatten()
    
    # Solve A_u U_flat = target_flat
    try:
        U_flat, _, _, _ = np.linalg.lstsq(A_u, target_flat, rcond=None)
        U = U_flat.reshape(n, m, rank)
    except np.linalg.LinAlgError:
        pass # Keep old U

    # 2. Solve for V
    # A_v has shape (n*m*p, rank)
    # A_v[i*m*p + j*p + k, c] = U[i,j,c] * W[k,i,c]
    A_v = np.zeros((n * m * p, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                A_v[idx, :] = U[i, j, :] * W[k, i, :]
    
    try:
        V_flat, _, _, _ = np.linalg.lstsq(A_v, target_flat, rcond=None)
        V = V_flat.reshape(m, p, rank)
    except np.linalg.LinAlgError:
        pass

    # 3. Solve for W
    # A_w has shape (n*m*p, rank)
    # A_w[i*m*p + j*p + k, c] = U[i,j,c] * V[j,k,c]
    A_w = np.zeros((n * m * p, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                A_w[idx, :] = U[i, j, :] * V[j, k, :]
    
    try:
        W_flat, _, _, _ = np.linalg.lstsq(A_w, target_flat, rcond=None)
        W = W_flat.reshape(p, n, rank)
    except np.linalg.LinAlgError:
        pass

    # Calculate residual
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U, V, W, optimize=True)
    err = np.max(np.abs(reconstructed - tensor))
    
    return U, V, W, err

def _verify_rounded(U, V, W, tensor, n, m, p, rank, tol=1e-6):
    """Round to half-integers and verify."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    err = np.max(np.abs(reconstructed - tensor))
    
    if err < tol:
        return U_r, V_r, W_r, True
    return None, None, None, False

def construct():
    deadline = time.time() + 145
    n, m, p = n_dim, m_dim, p_dim
    tensor = np.ones((n, m, p), dtype=np.float64)
    
    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fb
    
    rng = np.random.default_rng(42)
    
    # Try ranks from low to high
    # Lower bound estimate: max(nm, mp, np) = 24
    # We go up to best_rank - 1
    for rank in range(24, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Try multiple initializations
        found = False
        for init in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize V and W randomly
            V = rng.standard_normal((m, p, rank))
            W = rng.standard_normal((p, n, rank))
            U = rng.standard_normal((n, m, rank))
            
            # Run ALS
            prev_err = np.inf
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                U, V, W, err = _als_step(U, V, W, tensor, n, m, p, rank)
                
                if err < 1e-8:
                    # Try rounding
                    U_r, V_r, W_r, valid = _verify_rounded(U, V, W, tensor, n, m, p, rank)
                    if valid:
                        best_rank = rank
                        best_U = U_r.reshape(n*m, rank)
                        best_V = V_r.reshape(m*p, rank)
                        best_W = W_r.reshape(p*n, rank)
                        found = True
                        break
                elif err > prev_err * 1.1:
                    # Diverging or stuck, break
                    break
                prev_err = err
                
            if found:
                break
                
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
