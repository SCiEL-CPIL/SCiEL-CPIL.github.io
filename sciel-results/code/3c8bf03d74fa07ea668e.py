"""Matrix multiplication 3,3,5 - find rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, atol=1e-9):
    """Verify that U, V, W give a valid decomposition of the multiplication tensor."""
    T = _build_target_tensor(n, m, p)
    T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=atol):
        return False
    return True

def _snap_to_half_integers(arr, tol=0.05):
    """Snap values close to half-integers to exact half-integers."""
    result = arr.copy()
    for idx in np.ndindex(result.shape):
        val = result[idx]
        closest = round(val * 2) / 2.0
        if abs(val - closest) < tol:
            result[idx] = closest
    return result

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank(n, m, p, target_rank, deadline, max_iter=200):
    """
    Try to find a rank-`target_rank` decomposition using alternating least squares
    with random restarts. Returns (U, V, W) if found, else None.
    """
    T = _build_target_tensor(n, m, p)
    T_reshaped = T.reshape(n * m, m * p * p * n)  # Not quite right for ALS, let's use mode unfolding
    
    # For ALS on a 3-tensor T[a,b,c], we solve:
    # U = argmin ||T - sum_d U[:,d] V[:,d] W[:,d]||_F
    # This is equivalent to, for fixed V,W: U = (T x2 (V^T W^T)) / (V^T V * W^T W) ... 
    # Actually, the standard ALS update for U is:
    # U = (T x2 (V^T W^T)) / (V^T V * W^T W) where x2 is the mode-2 product
    # But T is (n*m, m*p, p*n). Let's denote dimensions A=n*m, B=m*p, C=p*n.
    # T is (A, B, C).
    # U is (A, r), V is (B, r), W is (C, r).
    # T ≈ sum_{d=1}^r U[:,d] ⊗ V[:,d] ⊗ W[:,d]
    # This is not a standard Tucker/ALS setup directly because the target T is not arbitrary,
    # but the ALS update rules are the same as for any 3-tensor.
    
    # Mode-1 unfolding of T is (A, B*C).
    # T_(1) = T.reshape(A, B*C).order('F')? No, standard numpy reshape is C-order.
    # Let's stick to explicit loops for clarity and correctness in the small case, 
    # or use einsum.
    
    # Update U: U[:,d] = sum_{b,c} T[a,b,c] V[b,d] W[c,d] / sum_{b,c} V[b,d]^2 W[c,d]^2
    # Actually, the normal equations for U are:
    # U = (T x2 (V ⊙ W)) / (V^T V * W^T W) where ⊙ is Khatri-Rao product.
    # Let's compute V_kr_W = np.einsum('bd,cd->bc', V, W)  # (B, C)
    # Then T_x2_VkW = np.einsum('abc,bc->a', T, V_kr_W)  # This gives a vector, not a matrix.
    # Wait, T is (A, B, C). V is (B, r), W is (C, r).
    # For each column d of U:
    # U[:,d] = sum_{b,c} T[:,b,c] V[b,d] W[c,d] / sum_{b,c} V[b,d]^2 W[c,d]^2
    # Let's vectorize this.
    # numerator[a,d] = sum_{b,c} T[a,b,c] V[b,d] W[c,d]
    # denominator[d] = sum_{b,c} V[b,d]^2 W[c,d]^2
    
    # numerator = np.einsum('abc,bd,cd->ad', T, V, W)
    # denominator = np.einsum('bd,cd->d', V**2, W**2)
    
    # Similarly for V and W.
    
    T = _build_target_tensor(n, m, p)
    A, B, C = n * m, m * p, p * n
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Initialize
        U = np.random.randn(A, target_rank)
        V = np.random.randn(B, target_rank)
        W = np.random.randn(C, target_rank)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Update U
            num_U = np.einsum('abc,bd,cd->ad', T, V, W)
            den_U = np.einsum('bd,cd->d', V**2, W**2)
            den_U[den_U < 1e-10] = 1e-10
            U = num_U / den_U
            
            # Update V
            num_V = np.einsum('abc,ad,cd->bd', T, U, W)
            den_V = np.einsum('ad,cd->d', U**2, W**2)
            den_V[den_V < 1e-10] = 1e-10
            V = num_V / den_V
            
            # Update W
            num_W = np.einsum('abc,ad,bd->cd', T, U, V)
            den_W = np.einsum('ad,bd->d', U**2, V**2)
            den_W[den_W < 1e-10] = 1e-10
            W = num_W / den_W
            
        # Check error
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        if best_err < 1e-8:
            break
            
    if best_err < 1e-8:
        # Try to snap to half-integers
        best_U = _snap_to_half_integers(best_U)
        best_V = _snap_to_half_integers(best_V)
        best_W = _snap_to_half_integers(best_W)
        
        # Verify after snapping
        if _verify(best_U, best_V, best_W, n, m, p, atol=1e-8):
            return best_U, best_V, best_W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    # Theoretical lower bound is at least max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Known bounds for 3x3x3 are around 23. For 3x3x5, it might be lower than 45.
    # Let's try ranks from 30 down to 15.
    
    for rank in range(30, 14, -1):
        if time.time() > deadline:
            break
            
        result = _try_rank(n, m, p, rank, deadline, max_iter=100)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            # Don't break immediately, might find lower rank? 
            # But ALS is local. Let's try a few more ranks just in case, 
            # but usually if we find one at rank R, lower ranks are harder.
            # However, the loop goes downwards, so if we find rank 30, we try 29, 28...
            # We should probably break if we find a valid one, unless we want to try to go lower.
            # The goal is rank-minimal. So we should continue trying lower ranks.
            pass
            
    return best_result
