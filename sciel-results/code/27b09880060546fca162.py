import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor T of size (n*m) x (m*p) x (p*n)."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Verify that U, V, W decompose T."""
    rank = U.shape[1]
    # Compute the reconstructed tensor
    # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # This is a batched matrix product
    # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    residual = np.max(np.abs(T_recon - T))
    return residual < tol, residual

def _try_rank_fixed_UV(U, V, T, tol=1e-6):
    """Given U and V, solve for W such that T = sum_r U[:,r] V[:,r] W[:,r]."""
    n_a = U.shape[0]
    n_b = V.shape[0]
    n_c = T.shape[2]
    rank = U.shape[1]
    
    # We need: for all a, b, c: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is a linear system in W.
    # For fixed a, b: sum_r (U[a,r] V[b,r]) W[c,r] = T[a,b,c] for all c
    # This is: M_{a,b} @ W = T[a,b,:] where M_{a,b}[c, r] = U[a,r] V[b,r]
    # But M_{a,b} is n_c x rank, and we have n_a * n_b such systems.
    # We can combine them into one big system.
    
    # Actually, it's easier to just iterate over a, b and solve for W[c, :] for each c.
    # But that's not quite right either.
    
    # Let me think again. We have:
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # For fixed c, this is: T[:,:,c] = sum_r U[:,r] V[:,r] W[c,r]
    # = sum_r W[c,r] (U[:,r] ⊗ V[:,r])
    # So T[:,:,c] is a linear combination of the rank-1 matrices U[:,r] ⊗ V[:,r].
    
    # Let M_c be the n_a*n_b x rank matrix where M_c[(a,b), r] = U[a,r] V[b,r]
    # Then T[:,:,c].flatten() = M_c @ W[c, :]
    
    # We can solve for W[c, :] for each c.
    
    W = np.zeros((n_c, rank))
    for c in range(n_c):
        M_c = (U * V.T).T.reshape(n_c, -1)  # This is wrong
        # Let me redo this.
        # M_c should be n_a*n_b x rank
        # M_c[a*n_b + b, r] = U[a, r] * V[b, r]
        M_c = np.zeros((n_a * n_b, rank))
        for a in range(n_a):
            for b in range(n_b):
                M_c[a * n_b + b, :] = U[a, :] * V[b, :]
        # Now solve M_c @ W[c, :] = T[:,:,c].flatten()
        try:
            W[c, :], residuals, rank_c, sv = np.linalg.lstsq(M_c, T[:, :, c].flatten(), rcond=None)
        except:
            return None
    
    return W

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    n_a = n * m
    n_b = m * p
    n_c = p * n
    
    deadline = time.time() + 140  # 10s safety margin
    
    best_rank = n_a * n_b  # schoolbook: n*m*p = 9*8 = 72? No, n*m*p = 3*3*8 = 72
    best_result = None
    
    # Try ranks from 72 down to 1
    for R in range(n_a * n_b, 0, -1):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 100 if R > 60 else 500
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random U and V
            U = np.random.randn(n_a, R)
            V = np.random.randn(n_b, R)
            
            W = _try_rank_fixed_UV(U, V, T)
            if W is None:
                continue
            
            valid, residual = _verify_decomposition(U, V, W, T)
            if valid:
                found = True
                best_rank = R
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if found:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        R = n * m * p
        U = np.zeros((n_a, R))
        V = np.zeros((n_b, R))
        W = np.zeros((n_c, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
